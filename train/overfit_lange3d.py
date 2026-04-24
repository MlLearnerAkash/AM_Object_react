"""Overfit-test for LangGeoNetV2 + LangGeoNetLoss.

Goal: definitively answer "can the model rank objects within a frame at all?"
We grab ONE batch of `batch_size` alive samples (filter_dead_samples=True)
from the existing dataset, then optimize on JUST that batch for `n_steps`.

If per-sample spearman → 1.0 and ranking_acc → 1.0 within ~200 steps, the
architecture/loss is capable and the joint-training failure is about lr/grad
balancing or metric mis-reporting.

If those numbers stay near 0/0.5, the architecture cannot learn this task as
configured (likely because frozen-backbone masked pooling makes per-object
features indistinguishable within a frame).
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vint_train.data.vint_dataset import ViNT_Dataset       # noqa: E402
from train import _collate_with_lange3d                     # noqa: E402
from lange3dnet_train.model import LangGeoNetV2             # noqa: E402
from lange3dnet_train.losses import LangGeoNetLoss          # noqa: E402


def per_sample_metrics(preds, gts):
    rho_list, acc_list = [], []
    for p, g in zip(preds, gts):
        p = p.detach().cpu().float().numpy()
        g = g.detach().cpu().float().numpy()
        K = min(len(p), len(g))
        if K < 2:
            continue
        p, g = p[:K], g[:K]

        if K >= 3 and g.std() > 1e-8:
            r, _ = spearmanr(p, g)
            if not np.isnan(r):
                rho_list.append(float(r))

        n_c, n_t = 0, 0
        for i in range(K):
            for j in range(i + 1, K):
                if abs(g[i] - g[j]) < 1e-8:
                    continue
                n_t += 1
                if (p[i] - p[j]) * (g[i] - g[j]) > 0:
                    n_c += 1
        if n_t:
            acc_list.append(n_c / n_t)
    return (
        float(np.mean(rho_list)) if rho_list else float("nan"),
        float(np.mean(acc_list)) if acc_list else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument(
        "--data-folder", type=str,
        default="/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/vint_format/",
    )
    ap.add_argument(
        "--split-folder", type=str,
        default="/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/splits/train/",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[overfit] device={device}")

    # --- Build a dataset that yields the lange3d goal-frame inputs ----------
    ds = ViNT_Dataset(
        data_folder=args.data_folder,
        data_split_folder=args.split_folder,
        dataset_name="object_react",
        image_size=[85, 64],
        waypoint_spacing=1,
        min_dist_cat=0, max_dist_cat=20,
        min_action_distance=2, max_action_distance=20,
        negative_mining=True,
        len_traj_pred=10, learn_angle=True,
        context_size=5, context_type="temporal",
        end_slack=3, goals_per_obs=1, normalize=True,
        obs_type="disabled", goal_type="image_mask_enc",
        return_lange3d_inputs=True,
        # kwargs forwarded:
        max_traj_len=100,
        filter_dead_samples=True,
        dims=8, goal_uses_context=False,
        precomputed_filename=None,
        pl_perturb_ratio=0.0, pl_perturb_type="max_val",
        mask_crop_ratio=1.0, use_mask_grad=False,
        clip_grad_norm=1.0,
    )
    print(f"[overfit] dataset has {len(ds)} samples (alive-filtered)")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate_with_lange3d,
    )
    batch = next(iter(loader))
    lang_inputs = batch[7]

    # Move tensors that we use repeatedly to device once.
    pixel_values = lang_inputs["pixel_values_goal"].to(device)
    masks_list   = [m.to(device) for m in lang_inputs["masks_goal_list"]]
    nai_ids      = lang_inputs["nai_input_ids"].to(device)
    nai_attn     = lang_inputs["nai_attention_mask"].to(device)
    gt_costs_list = [c.to(device) for c in lang_inputs["gt_costs_list"]]

    # Filter the batch down to "alive" samples (defensive: the dataset
    # already filters but a sample may still be K<2 / std<=eps after collation).
    keep = [
        i for i, g in enumerate(gt_costs_list)
        if g.numel() >= 2 and float(g.std()) > 1e-6
    ]
    if not keep:
        print("[overfit] no alive samples in batch; aborting")
        return
    pixel_values = pixel_values[keep]
    masks_list   = [masks_list[i]   for i in keep]
    nai_ids      = nai_ids[keep]
    nai_attn     = nai_attn[keep]
    gt_costs_list = [gt_costs_list[i] for i in keep]
    print(f"[overfit] using {len(keep)} alive samples; "
          f"K per sample: {[int(g.numel()) for g in gt_costs_list]}; "
          f"GT range per sample: "
          f"{[round(float(g.max()-g.min()),2) for g in gt_costs_list]}")

    # --- Model + loss --------------------------------------------------------
    model = LangGeoNetV2(
        d_model=256, n_heads=8, n_layers=2, d_ff=1024, dropout=0.1,
        clip_model_name="openai/clip-vit-base-patch16",
        dino_model_name="facebook/dinov2-small",
        freeze_clip=True, freeze_dino=True,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[overfit] trainable params: {n_train/1e6:.2f}M / {n_total/1e6:.2f}M")

    criterion = LangGeoNetLoss(
        lambda_rank=0.5, lambda_si=0.0, lambda_aux=0.5, lambda_div=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    model.train()
    for step in range(args.n_steps):
        preds, geo_preds, _ = model(
            pixel_values, masks_list, nai_ids, nai_attn, return_geo=True,
        )
        loss, parts = criterion(preds, gt_costs_list, geo_preds=geo_preds)

        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=10.0,
        )
        optimizer.step()

        if step % args.print_every == 0 or step == args.n_steps - 1:
            with torch.no_grad():
                rho, acc = per_sample_metrics(preds, gt_costs_list)
                # Diagnose mode collapse: per-sample std of predictions.
                pred_stds = [float(p.detach().std().item()) for p in preds]
                pred_std_mean = float(np.mean(pred_stds))
            print(
                f"step {step:4d}  loss={loss.item():.4f}  "
                f"reg={parts['loss_regression']:.3f}  "
                f"rank={parts['loss_ranking']:.3f}  "
                f"list={parts['loss_listnet']:.3f}  "
                f"div={parts['loss_diversity']:.3f}  "
                f"aux={parts['loss_aux_geo']:.3f}  "
                f"|grad|={float(gnorm):.2f}  "
                f"pred_std={pred_std_mean:.4f}  "
                f"rho={rho:.4f}  acc={acc:.4f}"
            )


if __name__ == "__main__":
    main()

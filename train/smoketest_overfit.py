"""
smoketest_overfit.py
====================
Diagnostic: overfit a single (obs, goal) sample with the exact same
forward / loss / backward path as ``train.py`` so we can verify that the
GNM + LangGeoNetV2 stack is *capable* of learning to drive towards the goal.

Usage:
    cd /data/ws/VLN-CE/controller/object_react/train
    python smoketest_overfit.py --config config/e3d_obj.yaml \
                                --steps 400 --sample_idx 0
"""
from __future__ import annotations
import argparse, os, sys, yaml, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.gnm.gnm import GNM
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.training.train_utils import (
    _compute_losses, get_obs_image, get_goal_image,
    _maybe_predict_goal_with_lange3d, _log_data,
)
from vint_train.visualizing.action_utils import compare_waypoints_pred_to_label
from vint_train.visualizing.visualize_utils import to_numpy, numpy_to_img
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
import torchvision.transforms.functional as TF

from lange3dnet_train.model import LangGeoNetV2
from lange3dnet_train.losses import LangGeoNetLoss

from train import _collate_with_lange3d, ready_dataloaders  # noqa


class SingleSampleDataset(Dataset):
    def __init__(self, sample, repeat=1):
        self.sample = sample
        self.repeat = repeat

    def __len__(self):
        return self.repeat

    def __getitem__(self, idx):
        return self.sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lange3d_lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", default="./smoketest_out")
    ap.add_argument("--lambda_lange3d", type=float, default=None,
                    help="Override lambda_lange3d from config")
    ap.add_argument(
        "--use_episode_end_goal", action="store_true",
        help="Force the goal to be the LAST frame of the chosen trajectory "
             "(per the user's request)."
    )
    args = ap.parse_args()

    with open("config/defaults.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(args.config) as f:
        cfg.update(yaml.safe_load(f))
    cfg["use_wandb"] = False
    cfg["batch_size"] = 1
    cfg["num_workers"] = 0
    cfg["eval_num_workers"] = 0

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- Build dataset & sample once -------------------------------------
    kwargs = {
        "predict_dists": cfg.get("predict_dists", True),
        "precomputed_filename": cfg.get("precomputed_filename", None),
        "pl_perturb_ratio": cfg.get("pl_perturb_ratio", 0.0),
        "pl_perturb_type": cfg.get("pl_perturb_type", "max_val"),
        "mask_crop_ratio": cfg.get("mask_crop_ratio", 1.0),
        "use_mask_grad": cfg.get("use_mask_grad", False),
        "goal_type": cfg.get("goal_type", "image"),
        "obs_type": cfg.get("obs_type", "image"),
        "dims": cfg.get("dims", None),
        "goal_uses_context": cfg.get("goal_uses_context", False),
        "return_lange3d_inputs": bool(cfg.get("use_lange3d", False)),
        "clip_model_name": cfg.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
        "gnm_mask_h": cfg.get("gnm_mask_h", 60),
        "gnm_mask_w": cfg.get("gnm_mask_w", 80),
    }
    print("Building train ViNT_Dataset...")
    ds_cfg = cfg["datasets"]["object_react"]
    base_ds = ViNT_Dataset(
        data_folder=ds_cfg["data_folder"],
        data_split_folder=ds_cfg["train"],
        dataset_name="object_react",
        image_size=cfg["image_size"],
        waypoint_spacing=ds_cfg["waypoint_spacing"],
        min_dist_cat=cfg["distance"]["min_dist_cat"],
        max_dist_cat=cfg["distance"]["max_dist_cat"],
        min_action_distance=cfg["action"]["min_dist_cat"],
        max_action_distance=cfg["action"]["max_dist_cat"],
        negative_mining=False,
        len_traj_pred=cfg["len_traj_pred"],
        learn_angle=cfg["learn_angle"],
        context_size=cfg["context_size"],
        context_type=cfg["context_type"],
        end_slack=ds_cfg["end_slack"],
        goals_per_obs=1,
        normalize=cfg["normalize"],
        **kwargs,
    )
    print(f"Dataset size = {len(base_ds)}")

    # Optionally force goal=end of trajectory (so we test "navigate to episode end")
    if args.use_episode_end_goal:
        # Pick the (traj, curr_time) at sample_idx; replace goal_time with last valid idx.
        f_curr, curr_time, max_goal_dist = base_ds.index_to_data[args.sample_idx]
        traj = base_ds._get_trajectory(f_curr)
        T = len(traj["position"])
        forced_goal_time = T - 1
        print(f"Forcing goal to end of episode {f_curr}: "
              f"curr_time={curr_time}, goal_time={forced_goal_time}, T={T}")

        # Monkey-patch _sample_goal so it always returns this end-of-episode goal
        def _const_goal(self, trajectory_name, curr_t, max_d):
            return f_curr, forced_goal_time, False
        import types
        base_ds._sample_goal = types.MethodType(_const_goal, base_ds)
        # Also override max_goal_dist in the index entry so action_mask is valid
        base_ds.index_to_data[args.sample_idx] = (f_curr, curr_time, T - curr_time - 1)

    print(f"Loading single sample idx={args.sample_idx}…")
    # The dataset's _sample_goal is non-deterministic — keep retrying / seeding
    # until we get a sample whose action_mask is 1 so the action loss is alive.
    sample = None
    for trial in range(200):
        np.random.seed(trial)
        s = base_ds[args.sample_idx]
        if float(s[6]) > 0.5:
            sample = s
            print(f"  found valid sample after {trial} trials "
                  f"(distance={int(s[3])}, mask={float(s[6])})")
            break
    if sample is None:
        sample = base_ds[args.sample_idx]
        print(f"  WARNING: could not find valid sample, mask={float(sample[6])}")
    print("  shapes:", [s.shape if hasattr(s, 'shape') else type(s).__name__
                        for s in sample[:7]])
    print("  action_label[:5,:2]:\n", sample[2][:, :2])
    print("  goal_pos:", sample[4].numpy())
    print("  distance:", int(sample[3]))
    print("  action_mask:", float(sample[6]))

    sds = SingleSampleDataset(sample, repeat=1)
    loader = DataLoader(sds, batch_size=1, shuffle=False,
                        collate_fn=_collate_with_lange3d if kwargs["return_lange3d_inputs"] else None)

    # ---- Build models ----------------------------------------------------
    print("Building GNM…")
    gnm = GNM(
        cfg["context_size"], cfg["len_traj_pred"], cfg["learn_angle"],
        cfg["obs_encoding_size"], cfg["goal_encoding_size"],
        **kwargs,
    ).to(device)

    lange3d = None
    topopaths = None
    lange3d_loss_fn = None
    if cfg.get("use_lange3d", False):
        print("Building LangGeoNetV2…")
        lange3d = LangGeoNetV2(
            d_model=cfg.get("lange3d_d_model", 256),
            n_heads=cfg.get("lange3d_n_heads", 8),
            n_layers=cfg.get("lange3d_n_layers", 2),
            clip_model_name=cfg.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
            dino_model_name=cfg.get("lange3d_dino_model", "facebook/dinov2-small"),
            freeze_clip=cfg.get("lange3d_freeze_clip", True),
            freeze_dino=cfg.get("lange3d_freeze_dino", True),
        ).to(device)
        topopaths = TopoPaths(
            dims=cfg["dims"], w=cfg.get("mask_w", 160), h=cfg.get("mask_h", 120),
        )
        lange3d_loss_fn = LangGeoNetLoss(
            lambda_rank=float(cfg.get("lange3d_lambda_rank", 1.0)),
            lambda_si=float(cfg.get("lange3d_lambda_si", 1.0)),
        ).to(device)

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---- Optimizer (matches train.py: GNM Adam + lange3d AdamW group) ----
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gnm.parameters()),
                            lr=args.lr)
    if lange3d is not None:
        lange3d_params = [p for p in lange3d.parameters() if p.requires_grad]
        if lange3d_params:
            opt.add_param_group({"params": lange3d_params, "lr": args.lange3d_lr})

    # ---- Train -----------------------------------------------------------
    obs_type = kwargs["obs_type"]
    goal_type_cfg = kwargs["goal_type"]
    alpha   = cfg["alpha"]
    learn_angle = cfg["learn_angle"]
    lambda_lange3d = (
        float(args.lambda_lange3d) if args.lambda_lange3d is not None
        else float(cfg.get("lambda_lange3d", 1.0))
    )
    print(f"lambda_lange3d = {lambda_lange3d}")

    log = []
    gnm.train()
    if lange3d is not None: lange3d.train()

    t0 = time.time()
    for step in range(args.steps):
        for data in loader:
            (obs_image, goal_image, action_label, dist_label, goal_pos,
             dataset_index, action_mask) = data[:7]

            obs_image, viz_obs = get_obs_image(obs_image, obs_type, transform, device)
            goal_image, replaced, lang_preds = _maybe_predict_goal_with_lange3d(
                data, goal_image, device, lange3d, topopaths,
            )
            eff_goal_type = "image_mask_enc" if replaced else goal_type_cfg
            goal_image, viz_goal = get_goal_image(
                goal_image, eff_goal_type, transform, device, obs_image,
            )

            dist_pred, action_pred = gnm(obs_image, goal_image)
            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            losses = _compute_losses(
                dist_label=dist_label, action_label=action_label,
                dist_pred=dist_pred, action_pred=action_pred,
                alpha=alpha, learn_angle=learn_angle, action_mask=action_mask,
            )
            if step == 0:
                import torch.nn.functional as _F
                raw_mse = _F.mse_loss(action_pred, action_label).item()
                xy_mse  = _F.mse_loss(action_pred[..., :2], action_label[..., :2]).item()
                print(f"  DEBUG step0: action_mask={action_mask.cpu().tolist()}  "
                      f"raw mse(all4)={raw_mse:.4f}  xy_mse={xy_mse:.4f}  "
                      f"alpha={alpha}  formula= {alpha*1e-2}*dist + {1-alpha}*action")

            l_lang_val = float("nan")
            if lang_preds is not None and lange3d_loss_fn is not None:
                gt_costs_list = [c.to(device) for c in data[7]["gt_costs_list"]]
                l_lang, _ = lange3d_loss_fn(lang_preds, gt_costs_list)
                losses["total_loss"] = losses["total_loss"] + lambda_lange3d * l_lang
                l_lang_val = float(l_lang.item())

            opt.zero_grad()
            losses["total_loss"].backward()

            # gradient norms (debug)
            gnm_gn = torch.nn.utils.clip_grad_norm_(
                [p for p in gnm.parameters() if p.grad is not None], 1e9,
            )
            lang_gn = float("nan")
            if lange3d is not None:
                lang_gn_val = torch.nn.utils.clip_grad_norm_(
                    [p for p in lange3d.parameters() if p.grad is not None], 1e9,
                )
                lang_gn = float(lang_gn_val)
            opt.step()

            entry = {
                "step": step,
                "total_loss": float(losses["total_loss"].item()),
                "action_loss": float(losses["action_loss"].item()),
                "dist_loss": float(losses["dist_loss"].item()),
                "lange3d_loss": l_lang_val,
                "action_cos_sim": float(losses["action_waypts_cos_sim"].item()),
                "gnm_gn": float(gnm_gn),
                "lang_gn": lang_gn,
            }
            log.append(entry)
            if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
                print(f"[{step:4d}/{args.steps}] "
                      f"L={entry['total_loss']:.4f} "
                      f"act={entry['action_loss']:.4f} "
                      f"dist={entry['dist_loss']:.4f} "
                      f"lang={entry['lange3d_loss']:.4f} "
                      f"cos={entry['action_cos_sim']:.3f} "
                      f"gnmGN={entry['gnm_gn']:.2f} langGN={entry['lang_gn']:.2f}")

            # Save first/last visualization
            if step == 0 or step == args.steps - 1:
                tag = "before" if step == 0 else "after"
                save_path = os.path.join(args.out_dir, f"viz_{tag}.png")
                compare_waypoints_pred_to_label(
                    obs_img=numpy_to_img(to_numpy(viz_obs[0])),
                    goal_img=numpy_to_img(to_numpy(viz_goal[0])),
                    dataset_name="object_react",
                    goal_pos=to_numpy(goal_pos[0]),
                    pred_waypoints=to_numpy(action_pred[0]),
                    label_waypoints=to_numpy(action_label[0]),
                    save_path=save_path,
                )
                print(f"  saved {save_path}")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s ({dt/max(1,args.steps)*1000:.1f} ms/step)")

    # Final summary
    first = log[0]; last = log[-1]
    print("\n=== Summary ===")
    print(f"  step 0   : total={first['total_loss']:.4f} act={first['action_loss']:.4f} "
          f"dist={first['dist_loss']:.4f} cos={first['action_cos_sim']:.3f}")
    print(f"  step {args.steps-1:4d}: total={last['total_loss']:.4f} act={last['action_loss']:.4f} "
          f"dist={last['dist_loss']:.4f} cos={last['action_cos_sim']:.3f}")

    # Print final pred vs label
    with torch.no_grad():
        print("\nFinal action_pred (first 5 wpts, x,y):")
        print(action_pred[0, :, :2].detach().cpu().numpy())
        print("Action label (first 5 wpts, x,y):")
        print(action_label[0, :, :2].detach().cpu().numpy())

    with open(os.path.join(args.out_dir, "log.json"), "w") as f:
        json.dump(log, f)
    print(f"\nLog → {os.path.join(args.out_dir, 'log.json')}")
    print(f"Vis → {args.out_dir}/viz_before.png & viz_after.png")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

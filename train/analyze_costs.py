#!/usr/bin/env python3
"""
analyze_costs.py
================
python analyze_costs.py -c config/eval_only.yaml --cost_source predicted  --joint_checkpoint 
/data/ws/VLN-CE/controller/object_react/train/logs/e3d_object_react/keep_600_epoch_joint_train_2026_04_30_01_43_45_/joint_latest.pth
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from os.path import normpath

import numpy as np
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# ---------------------------------------------------------------------------
# Model / data imports
# ---------------------------------------------------------------------------
from vint_train.models.gnm.gnm import GNM
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.object_react.dataloader import TopoPaths

from lange3dnet_train.model import LangGeoNetV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

try:
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ---------------------------------------------------------------------------
# Collation (matches eval_only.py)
# ---------------------------------------------------------------------------
def _collate_with_lange3d(batch):
    from torch.utils.data._utils.collate import default_collate
    has_extra = len(batch[0]) == 8
    main = [b[:7] for b in batch]
    main_collated = default_collate(main)
    if not has_extra:
        return main_collated
    extras = [b[7] for b in batch]
    K_list = [int(e["K"]) for e in extras]
    K_max = max(K_list) if K_list else 0
    Hm = extras[0]["gnm_masks"].shape[1] if extras[0]["gnm_masks"].ndim == 3 else 60
    Wm = extras[0]["gnm_masks"].shape[2] if extras[0]["gnm_masks"].ndim == 3 else 80
    gnm_masks_padded = torch.zeros(len(batch), max(K_max, 1), Hm, Wm, dtype=torch.float32)
    for i, e in enumerate(extras):
        if K_list[i] > 0:
            gnm_masks_padded[i, :K_list[i]] = e["gnm_masks"]
    lang_inputs = {
        "pixel_values_goal":  torch.stack([e["pixel_values_goal"]  for e in extras]),
        "nai_input_ids":      torch.stack([e["nai_input_ids"]      for e in extras]),
        "nai_attention_mask": torch.stack([e["nai_attention_mask"] for e in extras]),
        "masks_goal_list":    [e["masks_goal"] for e in extras],
        "gnm_masks":          gnm_masks_padded,
        "K_list":             K_list,
        "gt_costs_list":      [e["gt_costs"] for e in extras],
    }
    return tuple(main_collated) + (lang_inputs,)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def per_sample_ranking_accuracy(pred_logits, gt_costs):
    """
    Compute ranking accuracy within a single sample:
      fraction of object pairs (i, j) where the ordering of predicted costs
      matches the ordering of GT costs.

    pred_logits : [K] raw logits
    gt_costs    : [K] GT costs (raw, will be min-max normalised internally)
    """
    K = pred_logits.shape[0]
    if K < 2:
        return float("nan"), float("nan")

    # Min-max normalise both
    p = pred_logits.float()
    p_mm = (p - p.min()) / (p.max() - p.min() + 1e-8)

    g = gt_costs.float()
    g_mm = (g - g.min()) / (g.max() - g.min() + 1e-8)

    # Pairwise comparisons
    gt_i_lt_j = g_mm.unsqueeze(1) < g_mm.unsqueeze(0)   # i < j in GT
    pred_i_lt_j = p_mm.unsqueeze(1) < p_mm.unsqueeze(0)

    matches = (gt_i_lt_j == pred_i_lt_j).float()
    n_pairs = gt_i_lt_j.sum()
    accuracy = (matches * gt_i_lt_j.float()).sum() / n_pairs.clamp(min=1)

    # Spearman ρ
    if _SCIPY_OK and K >= 3 and g_mm.std() > 1e-8:
        rho, _ = _spearmanr(p_mm.cpu().numpy(), g_mm.cpu().numpy())
    else:
        rho = float("nan")

    return accuracy.item(), rho


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def denorm_clip(tensor):
    """[3, H, W] CLIP-normalised tensor → [H, W, 3] uint8."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    return np.clip((img * _CLIP_STD + _CLIP_MEAN) * 255, 0, 255).astype(np.uint8)


def draw_cost_comparison_grid(samples, out_path, max_rows=5):
    """
    Draw a grid of side-by-side bar charts: GT cost vs predicted cost per object.

    samples : list of dicts with keys:
        frame_key, cat_names (or idx labels), gt_costs, pred_costs,
        spearman, rank_acc, rgb (optional), nai_text (optional)
    """
    n = min(len(samples), max_rows)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i in range(n):
        s = samples[i]
        K = len(s["gt_costs"])
        labels = s.get("cat_names") or [f"Obj {j}" for j in range(K)]
        x = np.arange(K)

        # ---- Subplot 1: GT costs (blue) -----------------------------------
        ax = axes[i, 0]
        gt_mm = (s["gt_costs"] - s["gt_costs"].min()) / (s["gt_costs"].ptp() + 1e-8)
        colors_gt = plt.cm.Blues(0.3 + 0.7 * gt_mm)
        bars = ax.bar(x, s["gt_costs"], color=colors_gt, edgecolor="navy", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("GT Cost (raw path length)", fontsize=9)
        ax.set_title(f"Sample {i}: GT Costs  (K={K})", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, ls=":")

        # Annotate values
        for bar, val in zip(bars, s["gt_costs"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6)

        # ---- Subplot 2: Predicted costs (red) -----------------------------
        ax = axes[i, 1]
        pred_mm = (s["pred_costs"] - s["pred_costs"].min()) / (s["pred_costs"].ptp() + 1e-8)
        colors_pred = plt.cm.Reds(0.3 + 0.7 * pred_mm)
        bars = ax.bar(x, s["pred_costs"], color=colors_pred, edgecolor="darkred", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Predicted Cost (raw logit)", fontsize=9)
        ax.set_title(
            f"Sample {i}: Predicted Costs  ρ={s['spearman']:.3f}  "
            f"RankAcc={s['rank_acc']:.3f}",
            fontsize=10, fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, ls=":")

        for bar, val in zip(bars, s["pred_costs"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle("Side-by-Side: GT vs Predicted Costs per Object", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cost comparison grid saved → {out_path}")


def draw_cost_distribution(all_gt_raw, all_gt_norm, out_path):
    """
    Draw a two-panel histogram of GT costs.

    Panel 1: raw GT cost distribution (path lengths).
    Panel 2: min-max normalised GT cost distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Flatten
    gt_raw  = np.concatenate(all_gt_raw)
    gt_norm = np.concatenate(all_gt_norm)

    # ---- Panel 1: Raw GT costs -------------------------------------------
    ax = axes[0]
    ax.hist(gt_raw, bins=80, color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Raw GT Cost (path length, metres)", fontsize=11)
    ax.set_ylabel("Count (objects)", fontsize=11)
    ax.set_title(f"GT Cost Distribution (raw)\nN={len(gt_raw):,} objects  "
                 f"mean={gt_raw.mean():.2f}  σ={gt_raw.std():.2f}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls=":")

    # ---- Panel 2: Normalised GT costs ------------------------------------
    ax = axes[1]
    ax.hist(gt_norm, bins=60, color="#4CAF50", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Normalised GT Cost [0, 1]", fontsize=11)
    ax.set_ylabel("Count (objects)", fontsize=11)
    ax.set_title(f"GT Cost Distribution (min-max norm)\nN={len(gt_norm):,} objects  "
                 f"mean={gt_norm.mean():.3f}  σ={gt_norm.std():.3f}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls=":")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cost distribution saved → {out_path}")


def draw_scatter(all_gt_norm, all_pred_norm, spearman, out_path):
    """Scatter plot: predicted (min-max) vs GT (min-max) with Spearman ρ."""
    gt_flat   = np.concatenate(all_gt_norm)
    pred_flat = np.concatenate(all_pred_norm)

    # Sub-sample for readability
    rng = np.random.default_rng(0)
    if len(gt_flat) > 5000:
        idx = rng.choice(len(gt_flat), 5000, replace=False)
        gt_flat, pred_flat = gt_flat[idx], pred_flat[idx]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(gt_flat, pred_flat, s=6, alpha=0.3, color="#2196F3", edgecolors="none")
    ax.plot([0, 1], [0, 1], "r--", lw=1.5, label="y = x (perfect)")
    ax.set_xlabel("GT Cost  (min-max normalised)", fontsize=11)
    ax.set_ylabel("Predicted Cost  (min-max normalised)", fontsize=11)
    ax.set_title(f"Cost Prediction Quality  —  Spearman ρ = {spearman:.4f}",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Cost scatter plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config, args):
    # ---- Device -----------------------------------------------------------
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in config["gpu_ids"])
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- TopoPaths --------------------------------------------------------
    topopaths = TopoPaths(
        dims=config.get("dims", 8),
        w=config.get("mask_w", 160),
        h=config.get("mask_h", 120),
    )

    # ---- Kwargs for dataset -----------------------------------------------
    kwargs = {
        "predict_dists": config.get("predict_dists", True),
        "precomputed_filename": config.get("precomputed_filename", None),
        "pl_perturb_ratio": config.get("pl_perturb_ratio", 0.0),
        "pl_perturb_type": config.get("pl_perturb_type", "max_val"),
        "mask_crop_ratio": config.get("mask_crop_ratio", 1.0),
        "use_mask_grad": config.get("use_mask_grad", False),
        "goal_type": config.get("goal_type", "image"),
        "obs_type": config.get("obs_type", "image"),
        "dims": config.get("dims", None),
        "goal_uses_context": config.get("goal_uses_context", False),
        "return_lange3d_inputs": True,
        "clip_model_name": config.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
        "gnm_mask_h": config.get("gnm_mask_h", 60),
        "gnm_mask_w": config.get("gnm_mask_w", 80),
        "max_traj_len": config.get("max_traj_len", None),
        "filter_dead_samples": config.get("filter_dead_samples", False),
        "topopaths": topopaths,
    }

    # ---- Dataset (test split) ---------------------------------------------
    print("Loading test dataset...")
    dcfg = config["datasets"]["object_react"]
    dataset = ViNT_Dataset(
        data_folder=dcfg["data_folder"],
        data_split_folder=dcfg["test"],
        dataset_name="object_react",
        image_size=config["image_size"],
        waypoint_spacing=dcfg.get("waypoint_spacing", 1),
        min_dist_cat=config["distance"]["min_dist_cat"],
        max_dist_cat=config["distance"]["max_dist_cat"],
        min_action_distance=config["action"]["min_dist_cat"],
        max_action_distance=config["action"]["max_dist_cat"],
        negative_mining=dcfg.get("negative_mining", True),
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        context_size=config["context_size"],
        context_type=config.get("context_type", "temporal"),
        end_slack=dcfg.get("end_slack", 0),
        goals_per_obs=dcfg.get("goals_per_obs", 1),
        normalize=config["normalize"],
        **kwargs,
    )

    eval_batch_size = config.get("eval_batch_size", config["batch_size"])
    eval_num_workers = config.get("eval_num_workers", config.get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        drop_last=False,
        persistent_workers=(eval_num_workers > 0),
        collate_fn=_collate_with_lange3d,
    )
    print(f"  Dataset: {len(dataset)} samples  (batch_size={eval_batch_size})")

    # ---- Build LangGeoNetV2 -----------------------------------------------
    print("Building LangGeoNetV2 model...")
    lange3d = LangGeoNetV2(
        d_model=config.get("lange3d_d_model", 256),
        n_heads=config.get("lange3d_n_heads", 8),
        n_layers=config.get("lange3d_n_layers", 2),
        clip_model_name=config.get("lange3d_clip_model", "openai/clip-vit-base-patch16"),
        dino_model_name=config.get("lange3d_dino_model", "facebook/dinov2-small"),
        freeze_clip=config.get("lange3d_freeze_clip", True),
        freeze_dino=config.get("lange3d_freeze_dino", True),
    )

    ckpt_path = config["lange3d_checkpoint"]
    print(f"Loading LangGeoNetV2 from: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Joint checkpoint  →  extract "lange3d" sub-dict
    # Standalone ckpt   →  may have "model_state_dict" wrapper or be a bare state dict
    if "lange3d" in ck and "gnm" in ck:
        # Joint checkpoint (saved by joint_train.py / train_eval_loop_joint)
        # Structure: { "epoch": ..., "gnm": {...}, "lange3d": {...}, ... }
        state = ck["lange3d"]
        print(f"  Joint checkpoint (epoch={ck.get('epoch', '?')}) — "
              f"extracted 'lange3d' sub-dict")
    else:
        state = ck.get("model_state_dict", ck)
        # Also handle joint checkpoints where both keys exist after .get fallback
        if "lange3d" in state and "gnm" in state:
            print("  Joint checkpoint detected — extracting 'lange3d' sub-dict")
            state = state["lange3d"]
    miss, unexp = lange3d.load_state_dict(state, strict=False)
    print(f"  missing={len(miss)}  unexpected={len(unexp)}")
    lange3d = lange3d.to(device)
    lange3d.eval()

    # ---- Output directory -------------------------------------------------
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    out_dir = normpath(os.path.join(
        "logs", "analysis", f"cost_analysis_{timestamp}"
    ))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ---- Evaluate ---------------------------------------------------------
    print("\nRunning inference on test set...")
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    all_gt_raw    = []   # list of [K] np arrays (raw GT path lengths)
    all_gt_norm   = []   # list of [K] np arrays (min-max normalised)
    all_pred_raw  = []   # list of [K] np arrays (raw logits)
    all_pred_norm = []   # list of [K] np arrays (min-max normalised)
    spearmans     = []
    rank_accs     = []
    samples_for_viz = []
    n_total = 0
    n_with_objects = 0

    for batch_idx, batch in enumerate(loader):
        obs_img, goal_img, actions, dists, goal_pos, ds_idx, action_mask, lang_inputs = batch

        pv   = lang_inputs["pixel_values_goal"].to(device)   # [B, 3, 224, 224]
        nii  = lang_inputs["nai_input_ids"].to(device)
        niam = lang_inputs["nai_attention_mask"].to(device)
        masks_list = [m.to(device) for m in lang_inputs["masks_goal_list"]]
        K_list     = lang_inputs["K_list"]
        gt_costs_list = lang_inputs["gt_costs_list"]

        with torch.no_grad():
            lang_preds, _ = lange3d(pv, masks_list, nii, niam)

        for b in range(len(K_list)):
            K = K_list[b]
            n_total += 1
            if K == 0:
                continue
            n_with_objects += 1

            gt_raw = gt_costs_list[b].cpu().float().numpy()   # [K]
            pred_logits = lang_preds[b].cpu().float().numpy()  # [K]

            # Min-max normalise both
            gt_mm = (gt_raw - gt_raw.min()) / (gt_raw.ptp() + 1e-8)
            pred_mm = (pred_logits - pred_logits.min()) / (pred_logits.ptp() + 1e-8)

            all_gt_raw.append(gt_raw)
            all_gt_norm.append(gt_mm)
            all_pred_raw.append(pred_logits)
            all_pred_norm.append(pred_mm)

            acc, rho = per_sample_ranking_accuracy(
                torch.from_numpy(pred_logits), torch.from_numpy(gt_raw)
            )
            rank_accs.append(acc)
            spearmans.append(rho)

            # Collect a few samples for visualization
            if len(samples_for_viz) < 20 and K >= 2:
                # Try to get category names from masks_goal ordering — not available
                # directly here, so use indices
                samples_for_viz.append({
                    "frame_key": f"batch{batch_idx}_sample{b}",
                    "cat_names": [f"Obj {j}" for j in range(K)],
                    "gt_costs": gt_raw,
                    "pred_costs": pred_logits,
                    "spearman": rho,
                    "rank_acc": acc,
                })

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx + 1}: processed {n_total} frames, "
                  f"{n_with_objects} with objects")

    # ---- Aggregate metrics ------------------------------------------------
    rank_accs_arr  = np.array([v for v in rank_accs if np.isfinite(v)])
    spearmans_arr  = np.array([v for v in spearmans if np.isfinite(v)])

    # Global Spearman (all objects pooled)
    global_spearman = float("nan")
    if _SCIPY_OK and all_gt_norm:
        gt_all   = np.concatenate(all_gt_norm)
        pred_all = np.concatenate(all_pred_norm)
        if len(gt_all) > 2 and gt_all.std() > 1e-8:
            global_spearman = float(_spearmanr(gt_all, pred_all).correlation)

    # Per-sample mean Spearman
    mean_spearman = float(np.mean(spearmans_arr)) if len(spearmans_arr) > 0 else float("nan")
    mean_rank_acc = float(np.mean(rank_accs_arr)) if len(rank_accs_arr) > 0 else float("nan")

    # ---- Print summary ----------------------------------------------------
    print(f"\n{'='*60}")
    print(f"COST ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Total frames processed     : {n_total}")
    print(f"  Frames with objects (K>0)  : {n_with_objects}")
    print(f"  Total object instances     : {sum(len(a) for a in all_gt_raw)}")
    print(f"  Global Spearman ρ (pooled) : {global_spearman:.6f}")
    print(f"  Per-sample mean Spearman ρ : {mean_spearman:.6f}")
    print(f"  Per-sample mean Rank Acc   : {mean_rank_acc:.6f}")
    print(f"  Median Rank Acc            : {float(np.median(rank_accs_arr)):.6f}")
    print(f"  Median Spearman ρ          : {float(np.median(spearmans_arr)):.6f}")
    print(f"{'='*60}")

    # Save text summary
    summary_path = os.path.join(out_dir, "ranking_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Cost Analysis Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint          : {ckpt_path}\n")
        f.write(f"Total frames        : {n_total}\n")
        f.write(f"Frames w/ objects   : {n_with_objects}\n")
        f.write(f"Total objects       : {sum(len(a) for a in all_gt_raw)}\n")
        f.write(f"Global Spearman ρ   : {global_spearman:.6f}\n")
        f.write(f"Mean Spearman ρ     : {mean_spearman:.6f}\n")
        f.write(f"Median Spearman ρ   : {float(np.median(spearmans_arr)):.6f}\n")
        f.write(f"Mean Rank Acc       : {mean_rank_acc:.6f}\n")
        f.write(f"Median Rank Acc     : {float(np.median(rank_accs_arr)):.6f}\n")
        f.write(f"Std Spearman ρ      : {float(np.std(spearmans_arr)):.6f}\n")
        f.write(f"Std Rank Acc        : {float(np.std(rank_accs_arr)):.6f}\n")
    print(f"  Summary saved → {summary_path}")

    # ---- Draw visualizations ----------------------------------------------
    print("\nDrawing visualizations...")

    # 1. Side-by-side bar chart grid (first 8 samples)
    viz_samples = samples_for_viz[:8]
    if viz_samples:
        draw_cost_comparison_grid(
            viz_samples,
            os.path.join(out_dir, "cost_comparison_samples.png"),
            max_rows=8,
        )

    # 2. GT cost distribution histogram
    draw_cost_distribution(
        all_gt_raw, all_gt_norm,
        os.path.join(out_dir, "cost_distribution_gt.png"),
    )

    # 3. Scatter: predicted vs GT (min-max normalised)
    draw_scatter(
        all_gt_norm, all_pred_norm, global_spearman,
        os.path.join(out_dir, "cost_scatter.png"),
    )

    print(f"\nAll outputs written to {out_dir}/")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Analyze LangGeoNetV2 cost predictions")
    parser.add_argument("--config", "-c", default="config/eval_only.yaml", type=str)
    parser.add_argument("--cost_source", default="predicted", choices=["gt", "predicted"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of frames to process (for quick testing)")
    parser.add_argument("--joint_checkpoint", type=str, default=None,
                        help="Path to a joint checkpoint (contains 'lange3d' + 'gnm' keys). "
                             "If provided, extracts LangGeoNetV2 weights from it. "
                             "Otherwise uses 'lange3d_checkpoint' from config.")
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)

    # Prefer explicit --joint_checkpoint over config's lange3d_checkpoint
    if args.joint_checkpoint is not None:
        config["lange3d_checkpoint"] = args.joint_checkpoint

    if args.device is not None:
        config["gpu_ids"] = [int(args.device.split(":")[-1])]

    config["cost_source"] = args.cost_source
    if args.cost_source == "gt":
        config["use_lange3d"] = False
        config["goal_type"] = "image_mask_enc"
        config["obs_type"] = "disabled"
    else:
        config["use_lange3d"] = True
        config["goal_type"] = "image_mask_enc"
        config["obs_type"] = "disabled"

    print("Config:")
    print(yaml.dump(config, default_flow_style=False))

    main(config, args)

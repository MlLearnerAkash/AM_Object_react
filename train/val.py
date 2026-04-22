#!/usr/bin/env python3
"""
val.py
======
Quantitative evaluation of the joint LangGeoNetV2 + GNM checkpoint on the
same validation split used by train_eval_loop_joint (frame-level Subset split,
same seed and val_split fraction).

Metrics (computed only for frames where action_mask = 1, i.e. a target exists):
  • Navigation Error (NE, metres) — L2 between predicted and GT final waypoint
  • Success Rate (SR)  at r = 1.0, 1.5, 2.0 m
  • SPL                at r = 1.0, 1.5, 2.0 m
      SPL  = SR × L_shortest / max(L_pred, L_shortest)
      where L_shortest = dist_label (ground-truth Euclidean distance to target)
  • SSPL               at r = 1.0, 1.5, 2.0 m
      SSPL = SR × L_gt_traj / max(L_pred, L_gt_traj)
      where L_gt_traj is the cumulative length of the GT waypoint trajectory
  • Spearman ρ  (per-object cost-prediction quality, all frames with K > 0)

Results broken down by:
  all | direction-only NAI | object-reference NAI
  └─ OGCL-grounded (ref class = global Dijkstra min)
  └─ OGCL-not-grounded (cheaper non-ref object exists)

Coordinate system (Habitat Y-up, agent local frame):
  action_label[:, 0] = right   (+X)
  action_label[:, 1] = backward (+Z, i.e. -forward)
  Display convention:
    gt_fwd,   gt_right   = -gt_wp[:, 1],   gt_wp[:, 0]
    pred_fwd, pred_right = -pred_wp[:, 1], pred_wp[:, 0]

Usage (from the train/ directory):
    python val.py
    python val.py --checkpoint checkpoints/joint/best_joint.pth
    python val.py --save_viz --max_viz 300
    python val.py --out_dir results/run_42
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — mirrors joint_train.py
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "lange3dnet_train"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image

try:
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ---------------------------------------------------------------------------
# Defaults — must match joint_train.py / joint_val_app.py
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = os.path.join(_SCRIPT_DIR, "checkpoints/joint/latest.pth")
DEFAULT_H5         = "/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_test.h5"
DEFAULT_VAL_SPLIT  = 0.1
DEFAULT_SEED       = 42
DEFAULT_OUT_DIR    = os.path.join(_SCRIPT_DIR, "val_results")

TARGET_RADII = [1.0, 1.5, 2.0]   # metres

MODEL_CFG = dict(
    d_model=256, n_heads=8, n_layers=2,
    clip_model="openai/clip-vit-base-patch16",
    dino_model="facebook/dinov2-small",
    freeze_clip=True, freeze_dino=True,
    context_size=5, len_traj_pred=10, learn_angle=True,
    obs_encoding_size=1024, goal_encoding_size=1024,
    goal_type="image_mask_enc", obs_type="disabled",
    dims=8, use_mask_grad=False, goal_uses_context=False,
    mask_h=120, mask_w=160,
)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grn_red", ["#00cc44", "#ffff00", "#ff2200"]
)

_GROUP_LABELS = {
    "all":               "All frames",
    "direction_only":    "Direction-only NAI",
    "obj_ref":           "Object-reference NAI",
    "ogcl_grounded":     "OGCL grounded  (ref = Dijkstra min)",
    "ogcl_not_grounded": "OGCL not fired  (ref ≠ Dijkstra min)",
}
_GROUPS = list(_GROUP_LABELS.keys())


# ---------------------------------------------------------------------------
# Image / mask helpers
# ---------------------------------------------------------------------------

def _denorm_clip(tensor) -> np.ndarray:
    """[3, H, W] CLIP-normalised tensor → [H, W, 3] uint8."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    return np.clip((img * _CLIP_STD + _CLIP_MEAN) * 255, 0, 255).astype(np.uint8)


def _resize_masks(masks_np: np.ndarray, H: int, W: int) -> np.ndarray:
    """[K, Hm, Wm] bool → [K, H, W] bool (nearest-neighbour)."""
    K = masks_np.shape[0]
    if K == 0 or (masks_np.shape[1] == H and masks_np.shape[2] == W):
        return masks_np
    return np.stack([
        np.array(
            Image.fromarray(masks_np[k].astype(np.uint8)).resize((W, H), Image.NEAREST)
        ).astype(bool)
        for k in range(K)
    ])


def _cost_overlay(rgb: np.ndarray, masks: np.ndarray, costs: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """Return [H, W, 3] float32 [0,1] — RGB blended with cost heat-map."""
    H, W = rgb.shape[:2]
    img = rgb.astype(np.float32) / 255.0
    canvas = np.full((H, W), np.nan, dtype=np.float32)
    seg = np.zeros((H, W), bool)
    for k in range(masks.shape[0]):
        m = masks[k]
        if not m.any():
            continue
        c = float(costs[k])
        canvas[m] = np.clip(c, 0, 1) if np.isfinite(c) else 0.5
        seg |= m
    if seg.any():
        filled  = np.where(np.isnan(canvas), 0.5, canvas)
        heat    = _CMAP(filled)[:, :, :3]
        blended = (1 - alpha) * img + alpha * heat
        img     = np.where(seg[:, :, None], blended, img)
    # White contour outlines
    for k in range(masks.shape[0]):
        m = masks[k]
        if not m.any():
            continue
        pad  = np.pad(m.astype(np.uint8), 1)
        nb   = pad[:-2,1:-1] + pad[2:,1:-1] + pad[1:-1,:-2] + pad[1:-1,2:]
        cont = m & (nb < 4)
        img  = np.where(cont[:, :, None], 1.0, img)
    return np.clip(img, 0, 1)


def _annotate_costs(ax, masks: np.ndarray, costs: np.ndarray) -> None:
    for k in range(masks.shape[0]):
        m = masks[k]
        if not m.any():
            continue
        ys, xs = np.where(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        lbl = f"{float(costs[k]):.2f}" if np.isfinite(costs[k]) else "∞"
        ax.text(cx, cy, lbl, fontsize=5, color="white", ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))


# ---------------------------------------------------------------------------
# Navigation metric helpers
# ---------------------------------------------------------------------------

def _path_length(waypoints: np.ndarray) -> float:
    """Cumulative L2 path length of [T, 2] waypoints starting from origin [0,0]."""
    if waypoints.shape[0] == 0:
        return 0.0
    pts = np.vstack([[0.0, 0.0], waypoints])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _nav_error(pred_wp: np.ndarray, gt_wp: np.ndarray) -> float:
    """L2 distance between final predicted and GT positions [T,2]."""
    return float(np.linalg.norm(pred_wp[-1] - gt_wp[-1]))


def _to_display(wp: np.ndarray):
    """[T,2] [right,bwd] → (right_arr, fwd_arr) for 2-D top-down plot."""
    return wp[:, 0], -wp[:, 1]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save_viz(
    out_path: str,
    rgb: np.ndarray,
    masks_np: np.ndarray,
    gt_costs: np.ndarray,
    pred_costs: np.ndarray,
    gt_wp: np.ndarray,
    pred_wp: np.ndarray,
    nai_text: str,
    ep_id: str,
    frame_key: str,
    ne: float,
    dist_label: float,
    group_tag: str,
) -> None:
    """
    4-panel canvas:
      [0] GT cost overlay      [1] Pred cost overlay
      [2] Top-down trajectory  [3] RGB + direction arrows
      [text row spanning all columns]

    Trajectory panel:
      - GT  waypoints: green  circles connected by arrows → target marked ★
      - Pred waypoints: red squares connected by dashed arrows
      - NE error circle drawn around the GT target
      - Both trajectories are plotted on top of each other so
        the deviation from target is immediately visible.
    """
    gt_right,   gt_fwd   = _to_display(gt_wp)
    pred_right, pred_fwd = _to_display(pred_wp)

    gt_ov   = _cost_overlay(rgb, masks_np, gt_costs)
    pred_ov = _cost_overlay(rgb, masks_np, pred_costs)

    fig = plt.figure(figsize=(22, 8))
    gs  = fig.add_gridspec(2, 4, wspace=0.12, hspace=0.2, height_ratios=[1, 0.13])

    # ---- Panel 0: GT cost overlay ----------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(gt_ov)
    _annotate_costs(ax0, masks_np, gt_costs)
    ax0.set_title(f"GT cost  [{ep_id} / {frame_key}]", fontsize=9, fontweight="bold")
    ax0.axis("off")
    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax0, fraction=0.025, pad=0.02,
                 label="cost  (green=low  red=high)")

    # ---- Panel 1: Predicted cost overlay ---------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(pred_ov)
    _annotate_costs(ax1, masks_np, pred_costs)
    ax1.set_title(f"Pred cost  [{group_tag}]", fontsize=9, fontweight="bold")
    ax1.axis("off")
    fig.colorbar(sm, ax=ax1, fraction=0.025, pad=0.02)

    # ---- Panel 2: Top-down trajectory ------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])

    # GT trajectory: circles + arrows
    gt_pts_r = np.concatenate([[0.0], gt_right])
    gt_pts_f = np.concatenate([[0.0], gt_fwd])
    ax2.plot(gt_pts_r, gt_pts_f, "o-", color="#00cc44", lw=2.0, ms=5, label="GT traj")
    for t in range(1, len(gt_pts_r)):
        ax2.annotate(
            "", xy=(gt_pts_r[t], gt_pts_f[t]),
            xytext=(gt_pts_r[t-1], gt_pts_f[t-1]),
            arrowprops=dict(arrowstyle="-|>", color="#00cc44", lw=1.4),
        )

    # Pred trajectory: squares + dashed arrows
    pr_pts_r = np.concatenate([[0.0], pred_right])
    pr_pts_f = np.concatenate([[0.0], pred_fwd])
    ax2.plot(pr_pts_r, pr_pts_f, "s--", color="#ff2200", lw=2.0, ms=5, label="Pred traj")
    for t in range(1, len(pr_pts_r)):
        ax2.annotate(
            "", xy=(pr_pts_r[t], pr_pts_f[t]),
            xytext=(pr_pts_r[t-1], pr_pts_f[t-1]),
            arrowprops=dict(arrowstyle="-|>", color="#ff2200", lw=1.4,
                            connectionstyle="arc3,rad=0.0"),
        )

    # Target (GT endpoint) — gold star
    ax2.scatter([gt_right[-1]], [gt_fwd[-1]], marker="*", s=350,
                color="#FFD700", edgecolors="black", linewidths=0.8,
                zorder=10, label="Target ★")

    # NE circle around GT target
    circ = plt.Circle((gt_right[-1], gt_fwd[-1]), ne,
                       color="orange", fill=False, ls=":", lw=1.5,
                       label=f"NE = {ne:.2f} m")
    ax2.add_patch(circ)

    # Draw success-radius rings (dashed grey)
    for r_thresh in TARGET_RADII:
        ring = plt.Circle((gt_right[-1], gt_fwd[-1]), r_thresh,
                           color="grey", fill=False, ls="--", lw=0.8, alpha=0.5)
        ax2.add_patch(ring)
        ax2.text(gt_right[-1] + r_thresh * 0.7, gt_fwd[-1] + r_thresh * 0.7,
                 f"{r_thresh}m", fontsize=6, color="grey", alpha=0.7)

    # Robot marker
    ax2.plot(0, 0, "k^", ms=12, zorder=10, label="Robot")

    all_r = np.concatenate([[0], gt_right, pred_right])
    all_f = np.concatenate([[0], gt_fwd,   pred_fwd])
    pad   = max(1.0, float(np.abs(np.concatenate([all_r, all_f])).max()) * 0.25)
    ax2.set_xlim(all_r.min() - pad, all_r.max() + pad)
    ax2.set_ylim(all_f.min() - pad, all_f.max() + pad)
    ax2.set_xlabel("right (+) / left (−)", fontsize=8)
    ax2.set_ylabel("forward (+)", fontsize=8)
    ax2.set_title(
        f"Trajectory  NE={ne:.2f}m  dist_GT={dist_label:.2f}m",
        fontsize=9, fontweight="bold",
    )
    ax2.set_aspect("equal")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, ls=":", alpha=0.4)

    # ---- Panel 3: RGB + direction arrows (first waypoint heading) --------
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(rgb)

    H_img, W_img = rgb.shape[:2]
    cx, cy = W_img // 2, H_img // 2
    arrow_len = min(H_img, W_img) * 0.28   # pixels

    def _draw_dir_arrow(right_val, fwd_val, color, style):
        mag = np.sqrt(right_val ** 2 + fwd_val ** 2)
        if mag < 1e-3:
            return
        dx =  right_val / mag * arrow_len
        dy = -fwd_val   / mag * arrow_len   # image y-axis is inverted
        ax3.annotate(
            "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=3.0,
                            mutation_scale=18, linestyle=style),
        )

    # GT first-waypoint direction
    if len(gt_right) > 0:
        _draw_dir_arrow(float(gt_right[0]), float(gt_fwd[0]), "#00cc44", "solid")
    # Pred first-waypoint direction
    if len(pred_right) > 0:
        _draw_dir_arrow(float(pred_right[0]), float(pred_fwd[0]), "#ff2200", "dashed")

    # GT final-waypoint direction (target direction from robot)
    if len(gt_right) > 0:
        _draw_dir_arrow(float(gt_right[-1]), float(gt_fwd[-1]), "#009900", "dotted")

    ax3.set_title("RGB + motion direction", fontsize=9, fontweight="bold")
    ax3.axis("off")
    patches = [
        mpatches.Patch(color="#00cc44", label="GT 1st waypoint"),
        mpatches.Patch(color="#009900", label="GT target dir"),
        mpatches.Patch(color="#ff2200", label="Pred 1st waypoint"),
    ]
    ax3.legend(handles=patches, fontsize=7, loc="lower right")

    # ---- Instruction text row --------------------------------------------
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")
    wrapped = textwrap.fill(nai_text, width=130)
    ax_txt.text(0.5, 0.5, wrapped, ha="center", va="center",
                fontsize=13, color="black", transform=ax_txt.transAxes)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out_path)


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def _save_summary_plot(results: dict, path: str) -> None:
    groups  = _GROUPS
    g_short = {
        "all":               "All",
        "direction_only":    "Dir-only",
        "obj_ref":           "Obj-ref",
        "ogcl_grounded":     "OGCL-on",
        "ogcl_not_grounded": "OGCL-off",
    }
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]
    metrics = ["SR", "SPL", "SSPL"]
    x = np.arange(len(groups))
    width = 0.23

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False)
    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i]
        for r_i, radius in enumerate(TARGET_RADII):
            vals = [
                results[g].get(f"{metric}@{radius}m", float("nan"))
                for g in groups
            ]
            ys = [0.0 if np.isnan(v) else v for v in vals]
            offset = (r_i - 1) * width
            bars = ax.bar(x + offset, ys, width,
                          label=f"@{radius}m", color=colors[r_i], alpha=0.82)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=6, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([g_short[g] for g in groups], fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, ls=":")

    fig.suptitle("Validation Metrics by Frame Group", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary bar chart saved to {path}")


# ---------------------------------------------------------------------------
# Spearman / scatter plot
# ---------------------------------------------------------------------------

def _save_cost_scatter(gt_costs_all: list, pred_costs_all: list,
                       spearman: float, path: str) -> None:
    if not gt_costs_all:
        return
    gt_flat   = np.concatenate(gt_costs_all)
    pred_flat = np.concatenate(pred_costs_all)
    # Sub-sample for scatter readability
    rng = np.random.default_rng(0)
    if len(gt_flat) > 5000:
        idx = rng.choice(len(gt_flat), 5000, replace=False)
        gt_flat, pred_flat = gt_flat[idx], pred_flat[idx]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_flat, pred_flat, s=4, alpha=0.35, color="#2196F3")
    ax.plot([0, 1], [0, 1], "r--", lw=1.2, label="y = x")
    ax.set_xlabel("GT cost  (min-max normalised)", fontsize=11)
    ax.set_ylabel("Pred cost  (min-max normalised)", fontsize=11)
    ax.set_title(f"Cost Prediction Quality  —  Spearman ρ = {spearman:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Cost scatter plot saved to {path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _save_markdown(results: dict, path: str, checkpoint_path: str,
                   h5_path: str, epoch: int) -> None:
    def _f(v):
        return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.4f}"

    lines = [
        "# Joint Validation Results",
        "",
        f"| Item | Value |",
        f"|---|---|",
        f"| **Checkpoint** | `{checkpoint_path}` |",
        f"| **Epoch** | {epoch} |",
        f"| **H5 dataset** | `{h5_path}` |",
        f"| **Total val samples** | {results['n_val_total']} |",
        f"| **Spearman ρ** (cost quality) | **{_f(results['spearman_rho'])}** |",
        "",
        "---",
        "",
        "## Navigation Metrics",
        "",
    ]

    # Build header row
    metric_cols = []
    for r in TARGET_RADII:
        metric_cols += [f"SR@{r}m", f"SPL@{r}m", f"SSPL@{r}m"]
    headers = ["Group", "N", "NE mean (m)", "NE med (m)"] + metric_cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for g in _GROUPS:
        r = results[g]
        n = r["N"]
        if n == 0:
            row = [_GROUP_LABELS[g], "0"] + ["—"] * (len(headers) - 2)
        else:
            row = [
                _GROUP_LABELS[g], str(n),
                _f(r["NE_mean"]), _f(r["NE_median"]),
            ]
            for radius in TARGET_RADII:
                row += [
                    _f(r[f"SR@{radius}m"]),
                    _f(r[f"SPL@{radius}m"]),
                    _f(r[f"SSPL@{radius}m"]),
                ]
        lines.append("| " + " | ".join(row) + " |")

    # Comparative section: obj-ref vs direction-only
    lines += [
        "",
        "---",
        "",
        "## Comparative: Object-Reference vs Direction-Only",
        "",
    ]
    comp_headers = ["Metric"] + [f"r={r}m" for r in TARGET_RADII]
    comp_groups  = ["direction_only", "obj_ref", "ogcl_grounded", "ogcl_not_grounded"]
    for metric in ["SR", "SPL", "SSPL"]:
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Group | " + " | ".join(f"@{r}m" for r in TARGET_RADII) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(TARGET_RADII)) + " |")
        for g in comp_groups:
            r_dict = results[g]
            row = [_GROUP_LABELS[g]] + [
                _f(r_dict.get(f"{metric}@{r}m", float("nan")))
                for r in TARGET_RADII
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines += [
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Definition |",
        "|---|---|",
        "| **NE** | L2 distance (metres) between predicted final waypoint and GT final waypoint in agent-local frame |",
        "| **SR@r** | Fraction of frames where NE < r |",
        "| **SPL@r** | `SR × L_shortest / max(L_pred, L_shortest)` — L_shortest = dist_label (GT Euclidean distance to target) |",
        "| **SSPL@r** | `SR × L_gt_traj / max(L_pred, L_gt_traj)` — L_gt_traj = cumulative GT waypoint path length |",
        "| **Spearman ρ** | Rank correlation between GT (min-max) and predicted (min-max) per-object costs |",
        "",
        "## Group Definitions",
        "",
        "| Group | Condition |",
        "|---|---|",
        "| Direction-only | NAI contains no MP3D object keyword — OGCL never fires |",
        "| Object-reference | NAI contains an MP3D object keyword (`nai_matched = True`) |",
        "| OGCL grounded | Object-reference **and** referenced class is the global Dijkstra minimum (`nai_is_global_min = True`) |",
        "| OGCL not fired | Object-reference **but** a cheaper non-referenced object exists (`nai_is_global_min = False`) |",
        "",
        "## Coordinate System",
        "",
        "Waypoints are in agent-local frame `[right, backward]` (Habitat Y-up, metres).",
        "",
        "```",
        "gt_fwd,   gt_right   = -gt_wp[:, 1],   gt_wp[:, 0]",
        "pred_fwd, pred_right = -pred_wp[:, 1],  pred_wp[:, 0]",
        "```",
        "",
    ]

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Markdown report saved to {path}")


# ---------------------------------------------------------------------------
# Console print
# ---------------------------------------------------------------------------

def _print_results(results: dict) -> None:
    sep = "-" * 110
    print()
    print("=" * 110)
    print("  VALIDATION RESULTS")
    print("=" * 110)
    print(f"  Total val samples : {results['n_val_total']}")
    print(f"  Spearman ρ        : {results['spearman_rho']:.4f}")
    print(sep)

    col_w = 12
    header_cols = ["Group", "N", "NE_mean", "NE_med"] + \
                  [f"SR@{r}" for r in TARGET_RADII] + \
                  [f"SPL@{r}" for r in TARGET_RADII] + \
                  [f"SSPL@{r}" for r in TARGET_RADII]
    print("  " + "  ".join(c.ljust(col_w) for c in header_cols))
    print(sep)

    short_labels = {
        "all":               "All",
        "direction_only":    "Dir-only",
        "obj_ref":           "Obj-ref",
        "ogcl_grounded":     "OGCL-on",
        "ogcl_not_grounded": "OGCL-off",
    }

    def _f(v):
        return "   —   " if (v is None or np.isnan(v)) else f"{v:.4f}"

    for g in _GROUPS:
        r = results[g]
        n = r["N"]
        row = [short_labels[g].ljust(col_w), str(n).ljust(col_w)]
        if n == 0:
            row += ["—".ljust(col_w)] * (len(header_cols) - 2)
        else:
            row += [_f(r["NE_mean"]).ljust(col_w), _f(r["NE_median"]).ljust(col_w)]
            for radius in TARGET_RADII:
                row.append(_f(r[f"SR@{radius}m"]).ljust(col_w))
            for radius in TARGET_RADII:
                row.append(_f(r[f"SPL@{radius}m"]).ljust(col_w))
            for radius in TARGET_RADII:
                row.append(_f(r[f"SSPL@{radius}m"]).ljust(col_w))
        print("  " + "  ".join(row))

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    h5_path: str,
    device_str: str,
    out_dir: str,
    save_viz: bool = False,
    max_viz: int = 200,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if save_viz:
        viz_dir = os.path.join(out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)

    device = torch.device(device_str)

    # ---- Load models -------------------------------------------------------
    from vint_train.models.gnm.gnm import GNM
    from vint_train.models.object_react.dataloader import TopoPaths
    from lange3dnet_train.model import LangGeoNetV2
    from lange3dnet_train.joint_dataset import create_joint_dataloaders

    lange3d = LangGeoNetV2(
        d_model=MODEL_CFG["d_model"], n_heads=MODEL_CFG["n_heads"],
        n_layers=MODEL_CFG["n_layers"],
        clip_model_name=MODEL_CFG["clip_model"],
        dino_model_name=MODEL_CFG["dino_model"],
        freeze_clip=MODEL_CFG["freeze_clip"],
        freeze_dino=MODEL_CFG["freeze_dino"],
    ).to(device)

    gnm = GNM(
        context_size=MODEL_CFG["context_size"],
        len_traj_pred=MODEL_CFG["len_traj_pred"],
        learn_angle=MODEL_CFG["learn_angle"],
        obs_encoding_size=MODEL_CFG["obs_encoding_size"],
        goal_encoding_size=MODEL_CFG["goal_encoding_size"],
        goal_type=MODEL_CFG["goal_type"],
        obs_type=MODEL_CFG["obs_type"],
        dims=MODEL_CFG["dims"],
        use_mask_grad=MODEL_CFG["use_mask_grad"],
        goal_uses_context=MODEL_CFG["goal_uses_context"],
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    lange3d.load_state_dict(ckpt["lange3d"], strict=False)
    gnm.load_state_dict(ckpt["gnm"], strict=False)
    lange3d.eval()
    gnm.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"[val] Loaded checkpoint epoch={epoch}  from {checkpoint_path}")

    topopaths = TopoPaths(
        dims=MODEL_CFG["dims"], w=MODEL_CFG["mask_w"], h=MODEL_CFG["mask_h"]
    )

    # ---- Validation dataset ------------------------------------------------
    print(f"[val] Indexing H5: {h5_path} …")
    _, val_loader = create_joint_dataloaders(
        h5_path=h5_path, batch_size=1, num_workers=0,
        val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED,
        len_traj_pred=MODEL_CFG["len_traj_pred"],
        gnm_mask_h=MODEL_CFG["mask_h"] // 2,
        gnm_mask_w=MODEL_CFG["mask_w"] // 2,
    )
    val_samples = list(val_loader.dataset)
    n_val = len(val_samples)
    print(f"[val] {n_val} validation samples")

    # ---- Metric accumulators -----------------------------------------------
    # Records: per group → {ne, success, spl, sspl} lists
    records: dict[str, dict] = {
        g: {
            "ne":      [],
            "success": {r: [] for r in TARGET_RADII},
            "spl":     {r: [] for r in TARGET_RADII},
            "sspl":    {r: [] for r in TARGET_RADII},
        }
        for g in _GROUPS
    }

    # For Spearman ρ — collect all frames with K > 0
    gt_costs_all:   list[np.ndarray] = []
    pred_costs_all: list[np.ndarray] = []

    viz_count = 0

    # ---- Loop --------------------------------------------------------------
    for idx, sample in enumerate(val_samples):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  [{idx+1}/{n_val}]")

        action_mask       = float(sample["action_mask"].item())
        nai_matched       = bool(sample.get("nai_matched", False))
        nai_is_global_min = bool(sample.get("nai_is_global_min", False))
        nai_text          = sample.get("nai_text", "")
        ep_id             = str(sample.get("ep_id", "?"))
        frame_key         = str(sample.get("frame_key", "?"))

        masks    = sample["masks"]          # [K, H, W] bool CPU
        K        = masks.shape[0]
        gt_costs = sample["gt_costs"].cpu().float().numpy()   # [K]

        # Group membership flags
        is_dir_only  = not nai_matched
        is_obj_ref   = nai_matched
        is_ogcl_on   = nai_matched and nai_is_global_min
        is_ogcl_off  = nai_matched and not nai_is_global_min

        # ---- Forward pass --------------------------------------------------
        pv   = sample["pixel_values"].unsqueeze(0).to(device)       # [1,3,224,224]
        nii  = sample["nai_input_ids"].unsqueeze(0).to(device)      # [1,77]
        niam = sample["nai_attn_mask"].unsqueeze(0).to(device)      # [1,77]
        gm   = sample["gnm_masks"].unsqueeze(0).to(device)          # [1,K,Hh,Wh]

        with torch.no_grad():
            lang_preds, _ = lange3d(pv, [masks.to(device)], nii, niam)
            goal_enc = topopaths.build_differentiable_goal(lang_preds, gm, [K], device)
            _, goal_img = goal_enc.split([3, goal_enc.shape[1] - 3], dim=1)
            obs_img = torch.zeros(1, 3, 120, 160, device=device)
            _, action_pred = gnm(obs_img, goal_img)

        raw_logits = lang_preds[0].cpu().float().numpy()   # [K]
        pred_mm    = (raw_logits - raw_logits.min()) / (np.ptp(raw_logits) + 1e-8)

        if K > 0:
            gt_costs_all.append(gt_costs)
            pred_costs_all.append(pred_mm)

        # ---- Navigation metrics (only when there is a labelled target) ----
        if action_mask > 0.5:
            gt_wp   = sample["action_label"].cpu().float().numpy()[:, :2]   # [T,2]
            pred_wp = action_pred[0].cpu().float().numpy()[:, :2]           # [T,2]

            dist_label  = float(sample["dist_label"].item())   # metres
            ne          = _nav_error(pred_wp, gt_wp)
            L_gt_traj   = _path_length(gt_wp)
            L_pred_traj = _path_length(pred_wp)
            L_shortest  = dist_label if dist_label > 1e-3 else L_gt_traj

            def _update(g: str) -> None:
                records[g]["ne"].append(ne)
                for r in TARGET_RADII:
                    succ  = float(ne < r)
                    spl_v = succ * L_shortest / max(L_pred_traj, L_shortest, 1e-6)
                    sspl_v = succ * L_gt_traj / max(L_pred_traj, L_gt_traj, 1e-6)
                    records[g]["success"][r].append(succ)
                    records[g]["spl"][r].append(spl_v)
                    records[g]["sspl"][r].append(sspl_v)

            _update("all")
            if is_dir_only:  _update("direction_only")
            if is_obj_ref:   _update("obj_ref")
            if is_ogcl_on:   _update("ogcl_grounded")
            if is_ogcl_off:  _update("ogcl_not_grounded")

            # ---- Visualisation -------------------------------------------
            if save_viz and viz_count < max_viz:
                rgb      = _denorm_clip(sample["pixel_values"])
                H_img, W_img = rgb.shape[:2]
                masks_np = _resize_masks(masks.cpu().numpy().astype(bool), H_img, W_img)
                gt_ov    = _cost_overlay(rgb, masks_np, gt_costs)
                pred_ov  = _cost_overlay(rgb, masks_np, pred_mm)
                group_tag = (
                    "OGCL-on"   if is_ogcl_on  else
                    "Obj-ref"   if is_obj_ref  else
                    "Dir-only"
                )
                fname = f"{idx:05d}_ep{ep_id}_f{frame_key}_NE{ne:.2f}.png"
                _save_viz(
                    out_path    = os.path.join(viz_dir, fname),
                    rgb         = rgb,
                    masks_np    = masks_np,
                    gt_costs    = gt_costs,
                    pred_costs  = pred_mm,
                    gt_wp       = gt_wp,
                    pred_wp     = pred_wp,
                    nai_text    = nai_text,
                    ep_id       = ep_id,
                    frame_key   = frame_key,
                    ne          = ne,
                    dist_label  = dist_label,
                    group_tag   = group_tag,
                )
                viz_count += 1

    # ---- Spearman ρ --------------------------------------------------------
    spearman = float("nan")
    if _SCIPY_OK and gt_costs_all:
        try:
            gt_flat   = np.concatenate(gt_costs_all)
            pred_flat = np.concatenate(pred_costs_all)
            if len(gt_flat) > 2 and gt_flat.std() > 1e-8:
                spearman = float(_spearmanr(gt_flat, pred_flat).correlation)
        except Exception:
            pass

    # ---- Aggregate ---------------------------------------------------------
    def _agg(g: str) -> dict:
        rc = records[g]
        n  = len(rc["ne"])
        if n == 0:
            base = {"N": 0, "NE_mean": float("nan"), "NE_median": float("nan")}
            for radius in TARGET_RADII:
                for pfx in ("SR", "SPL", "SSPL"):
                    base[f"{pfx}@{radius}m"] = float("nan")
            return base
        res = {
            "N":         n,
            "NE_mean":   float(np.mean(rc["ne"])),
            "NE_median": float(np.median(rc["ne"])),
        }
        for radius in TARGET_RADII:
            res[f"SR@{radius}m"]   = float(np.mean(rc["success"][radius]))
            res[f"SPL@{radius}m"]  = float(np.mean(rc["spl"][radius]))
            res[f"SSPL@{radius}m"] = float(np.mean(rc["sspl"][radius]))
        return res

    final: dict = {g: _agg(g) for g in _GROUPS}
    final["spearman_rho"] = spearman
    final["n_val_total"]  = n_val

    # ---- Output ------------------------------------------------------------
    _print_results(final)
    _save_markdown(final, os.path.join(out_dir, "val_results.md"),
                   checkpoint_path, h5_path, epoch)
    _save_summary_plot(final, os.path.join(out_dir, "summary_bar.png"))
    _save_cost_scatter(gt_costs_all, pred_costs_all, spearman,
                       os.path.join(out_dir, "cost_scatter.png"))

    with open(os.path.join(out_dir, "val_results.json"), "w") as fh:
        json.dump(final, fh, indent=2)
    print(f"[val] All outputs written to {out_dir}/")
    if save_viz:
        print(f"[val] {viz_count} visualisations saved to {out_dir}/viz/")

    return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation on the joint training val split"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to best_joint.pth")
    parser.add_argument("--h5", default=DEFAULT_H5,
                        help="Path to H5 dataset file")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu  (default: auto)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="Directory to write results to")
    parser.add_argument("--save_viz", action="store_true",
                        help="Save per-sample visualisation PNGs")
    parser.add_argument("--max_viz", type=int, default=200,
                        help="Maximum number of visualisations to save")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate(
        checkpoint_path = args.checkpoint,
        h5_path         = args.h5,
        device_str      = args.device,
        out_dir         = args.out_dir,
        save_viz        = args.save_viz,
        max_viz         = args.max_viz,
    )

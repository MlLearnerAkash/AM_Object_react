"""
LangGeoNet Gradio App

Interactive demo: upload an image, enter a navigation instruction,
and get a costmap visualization showing predicted geodesic distances.

Usage:
    python app.py --checkpoint checkpoints/best_model.pt
    python app.py --checkpoint checkpoints/best_model.pt --share
"""

import os
import sys
import argparse
import io

from typing import Optional

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Mask loading  (mirrors inference.py --masks behaviour)
# ---------------------------------------------------------------------------

def load_masks(masks_path: str) -> np.ndarray:
    """
    Load a masks .npy file [K, H, W] and filter out empty masks,
    exactly as predict_frame() does in inference.py.

    Args:
        masks_path: path to masks.npy with shape [K, H, W]

    Returns:
        masks: np.ndarray [K_valid, H, W] bool
    """
    masks = np.load(masks_path)          # [K, H, W]
    valid = masks.any(axis=(1, 2))       # remove all-zero masks
    masks = masks[valid]
    return masks


def load_gt_distances(masks_path: str) -> Optional[np.ndarray]:
    """
    Look for ``geodesic_distances.npy`` in the same directory as *masks_path*
    and return a [K_valid] float array aligned to the non-empty masks.

    The GT file is expected to hold one distance value per mask entry [K_all].
    Empty-mask rows are dropped with the same boolean filter used in
    :func:`load_masks` so the result aligns with the filtered masks array.

    Returns ``None`` when the file does not exist.
    """
    gt_path = os.path.join(os.path.dirname(os.path.abspath(masks_path)),
                           "geodesic_distances.npy")
    if not os.path.isfile(gt_path):
        return None

    # Re-load the *unfiltered* masks to reproduce the same valid boolean mask
    raw_masks = np.load(masks_path)              # [K_all, H, W]
    valid     = raw_masks.any(axis=(1, 2))       # [K_all] bool

    gt_all = np.load(gt_path).astype(np.float32)  # [K_all] or [K_valid]
    if gt_all.shape[0] == valid.sum():
        # Already filtered (K_valid entries)
        return gt_all
    if gt_all.shape[0] == raw_masks.shape[0]:
        # Full array — apply the same valid filter
        return gt_all[valid]

    # Unexpected shape — return as-is and let the caller decide
    return gt_all


# ---------------------------------------------------------------------------
# Predictor (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_predictor = None

def get_predictor(checkpoint_path: str, device: str = None):
    global _predictor
    if _predictor is None:
        from inference import LangGeoNetPredictor
        _predictor = LangGeoNetPredictor(checkpoint_path, device=device)
    return _predictor


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def render_costmap_figure(
    image_np: np.ndarray,
    costmap: np.ndarray,
    instruction: str,
    n_objects: int,
) -> np.ndarray:
    """Render a 3-panel costmap figure and return as RGB numpy array."""
    colormap = cm.get_cmap("RdYlGn_r")
    normalized = (costmap - costmap.min()) / (costmap.max() - costmap.min() + 1e-8)
    colored = colormap(normalized)[:, :, :3]
    overlay = np.clip(0.5 * (image_np / 255.0) + 0.5 * colored, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Instruction: "{instruction}"\n{n_objects} objects detected',
        fontsize=11,
        wrap=True,
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(normalized, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[1].set_title("Predicted Costmap\n(green = near goal, red = far)")
    axes[1].axis("off")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1)),
        ax=axes[1],
        fraction=0.046,
        label="Normalized distance",
    )

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_per_object_figure(
    image_np: np.ndarray,
    masks: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Render bar chart + closest/farthest overlay as RGB numpy array."""
    K = len(distances)
    order = np.argsort(distances)
    labels = [f"obj_{k}" for k in range(K)]

    colormap = cm.get_cmap("RdYlGn_r")
    norm_d = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bar_colors = colormap(norm_d[order])
    axes[0].barh(range(K), distances[order], color=bar_colors)
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels([labels[i] for i in order], fontsize=max(5, 9 - K // 10))
    axes[0].set_xlabel("Predicted Geodesic Distance")
    axes[0].set_title("Per-Object Distances (sorted)")
    axes[0].invert_yaxis()

    overlay = image_np.astype(np.float32) / 255.0
    closest_k = order[0]
    farthest_k = order[-1]
    green = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for k, color in [(closest_k, green), (farthest_k, red)]:
        mask = masks[k] > 0
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * color

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(
        f"Closest: {labels[closest_k]} ({distances[closest_k]:.3f})  |  "
        f"Farthest: {labels[farthest_k]} ({distances[farthest_k]:.3f})"
    )
    axes[1].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_gt_diff_figure(
    image_np: np.ndarray,
    masks: np.ndarray,
    pred_distances: np.ndarray,
    gt_distances: np.ndarray,
) -> np.ndarray:
    """
    Four-panel figure comparing GT and predicted geodesic distances.

    Panels
    ------
    1. GT costmap               – masks painted with GT geodesic distances
    2. Predicted costmap        – same colour scale as GT
    3. |pred − GT| costmap      – absolute error per pixel (hot colourmap)
    4. Abs-error overlay        – absolute error heatmap blended onto the
                                  input image so the changing regions are
                                  spatially visible; per-object abs-error
                                  bar chart inset via a second sub-figure row
    """
    K, H, W = masks.shape

    # Build per-pixel costmaps (background = NaN → renders as white/transparent)
    gt_map   = np.full((H, W), np.nan, dtype=np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)
    abs_map  = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(K):
        px = masks[k] > 0
        gt_map[px]   = gt_distances[k]
        pred_map[px] = pred_distances[k]
        abs_map[px]  = abs(pred_distances[k] - gt_distances[k])

    # Shared colour scale for GT / pred panels
    all_vals = np.concatenate([gt_distances, pred_distances])
    vmin_shared, vmax_shared = float(all_vals.min()), float(all_vals.max())

    # Absolute error scale (0 → max)
    abs_vals = np.abs(pred_distances - gt_distances)
    abs_max  = max(float(abs_vals.max()), 1e-6)

    # --- Abs-error overlay on the input image ---
    # Normalise abs_map to [0,1], colourise, blend with image
    abs_map_norm = np.where(np.isnan(abs_map), 0.0, abs_map / abs_max)  # [H,W]
    hot_cmap  = cm.get_cmap("hot")
    abs_color = hot_cmap(abs_map_norm)[:, :, :3]                        # [H,W,3]
    base      = image_np.astype(np.float32) / 255.0
    # Only colour pixels that belong to at least one mask
    any_mask = (~np.isnan(abs_map))
    overlay  = base.copy()
    overlay[any_mask] = (0.45 * base[any_mask] +
                         0.55 * abs_color[any_mask])

    mae = float(abs_vals.mean())

    fig = plt.figure(figsize=(26, 12))
    # Top row: 4 image panels
    ax0 = fig.add_subplot(2, 4, 1)
    ax1 = fig.add_subplot(2, 4, 2)
    ax2 = fig.add_subplot(2, 4, 3)
    ax3 = fig.add_subplot(2, 4, 4)
    # Bottom row: per-object bar chart spanning all columns
    ax4 = fig.add_subplot(2, 1, 2)

    fig.suptitle(f"GT vs Predicted — MAE = {mae:.4f}", fontsize=13)

    # --- Panel 1: GT costmap ---
    im0 = ax0.imshow(gt_map, cmap="RdYlGn_r",
                     vmin=vmin_shared, vmax=vmax_shared)
    ax0.set_title("GT Geodesic Distances")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, label="Distance")

    # --- Panel 2: Predicted costmap ---
    im1 = ax1.imshow(pred_map, cmap="RdYlGn_r",
                     vmin=vmin_shared, vmax=vmax_shared)
    ax1.set_title("Predicted Geodesic Distances")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, label="Distance")

    # --- Panel 3: Absolute difference |pred − GT| ---
    im2 = ax2.imshow(abs_map, cmap="hot", vmin=0, vmax=abs_max)
    ax2.set_title("|pred − GT|  Absolute Error")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, label="|Δ Distance|")

    # --- Panel 4: Overlay – changing regions on image ---
    ax3.imshow(np.clip(overlay, 0, 1))
    ax3.set_title("Changing Regions  (hot = higher error)")
    ax3.axis("off")

    # --- Bottom: per-object absolute error bar chart ---
    labels  = [f"obj_{k}" for k in range(K)]
    order   = np.argsort(abs_vals)[::-1]          # largest error first
    bar_colors = cm.get_cmap("hot")(
        abs_vals[order] / abs_max * 0.85 + 0.05   # avoid pitch-black bars
    )
    ax4.bar(range(K), abs_vals[order], color=bar_colors)
    ax4.set_xticks(range(K))
    ax4.set_xticklabels([labels[i] for i in order], rotation=45, ha="right",
                         fontsize=max(5, 9 - K // 10))
    ax4.set_ylabel("|pred − GT|")
    ax4.set_title("Per-Object Absolute Error  (sorted, largest first)")
    # Annotate each bar with its raw GT / pred values
    for rank, k in enumerate(order):
        ax4.text(rank, abs_vals[k] + abs_max * 0.01,
                 f"GT={gt_distances[k]:.2f}\nP={pred_distances[k]:.2f}",
                 ha="center", va="bottom", fontsize=max(5, 7 - K // 15))

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_attention_figure(
    instruction: str,
    attn_weights: np.ndarray,
    n_objects: int,
) -> np.ndarray:
    """
    Heatmap: which instruction tokens each object attends to.
    - Skips the CLIP SOT token (position 0) — handled upstream in inference.py.
    - Applies row-wise min-max normalization so per-object patterns are visible
      even when absolute attention values are small.
    """
    from transformers import CLIPTokenizer  # type: ignore[import]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    # tokenize returns content tokens only (no SOT/EOS), matching the sliced attn_weights
    tokens = tokenizer.tokenize(instruction)

    K, L = attn_weights.shape
    L_show = min(L, len(tokens))
    labels = [f"obj_{k}" for k in range(K)]

    # Row-wise min-max normalization: makes per-object token preferences visible
    # even when all absolute values are similar (flat raw attention)
    row_min = attn_weights[:, :L_show].min(axis=1, keepdims=True)
    row_max = attn_weights[:, :L_show].max(axis=1, keepdims=True)
    attn_norm = (attn_weights[:, :L_show] - row_min) / (row_max - row_min + 1e-8)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, L_show * 0.55), max(3, K * 0.4)),
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # Left: raw attention
    im0 = axes[0].imshow(attn_weights[:, :L_show], cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(L_show))
    axes[0].set_xticklabels(tokens[:L_show], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_title("Raw Cross-Attention")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Right: row-normalised attention (highlights relative preference per object)
    im1 = axes[1].imshow(attn_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(L_show))
    axes[1].set_xticklabels(tokens[:L_show], rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(K))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_title("Row-Normalised Attention\n(per-object token preference)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.suptitle("Cross-Attention: Object × Instruction Token", fontsize=11)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_ranking_comparison_figure(
    image_np: np.ndarray,
    masks: np.ndarray,
    pred_distances: np.ndarray,
    gt_distances: np.ndarray,
) -> np.ndarray:
    """
    Four-panel ranking comparison figure.

    Panel 1 — GT ranked list with rank-change annotations.
    Panel 2 — Predicted ranked list with GT rank reference.
    Panel 3 — Per-object rank-change bar chart.
    Panel 4 — Scene overlay: each object coloured by rank change
               (green = ranked closer in pred, red = ranked farther, grey = same).
    """
    K = len(gt_distances)
    labels = [f"obj_{k}" for k in range(K)]

    # rank of each object (0 = closest to goal = smallest distance)
    gt_rank   = np.argsort(np.argsort(gt_distances)).astype(int)
    pred_rank = np.argsort(np.argsort(pred_distances)).astype(int)
    rank_change = gt_rank - pred_rank          # positive → closer in pred
    gt_order    = np.argsort(gt_distances)
    pred_order  = np.argsort(pred_distances)

    max_change = max(int(np.abs(rank_change).max()), 1)
    norm_change = rank_change / max_change     # [-1, 1]

    def _change_color(v):
        if v > 0:
            return (0.2, 0.4 + 0.5 * v, 0.2)
        elif v < 0:
            return (0.5 + 0.4 * abs(v), 0.2, 0.2)
        return (0.6, 0.6, 0.6)

    fig = plt.figure(figsize=(20, max(10, K * 0.55 + 4)))
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[max(K * 0.45, 4), 4],
        hspace=0.40, wspace=0.30,
    )
    ax_gt    = fig.add_subplot(gs[0, 0])
    ax_pred  = fig.add_subplot(gs[0, 1])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_scene = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "GT vs Predicted Object Ranking Comparison\n"
        "(rank #1 = closest to navigation goal)",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: GT ranked list ---
    ax_gt.set_xlim(0, 1)
    ax_gt.set_ylim(-0.5, K - 0.5)
    ax_gt.invert_yaxis()
    ax_gt.axis("off")
    ax_gt.set_title("GT Ranking  (ground truth)", fontsize=11, fontweight="bold")

    for pos, obj_idx in enumerate(gt_order):
        color = _change_color(norm_change[obj_idx])
        ax_gt.add_patch(plt.Rectangle(
            (0, pos - 0.45), 1, 0.9,
            color=color, alpha=0.25, transform=ax_gt.transData,
        ))
        ax_gt.text(0.04, pos, f"#{pos + 1}",
                   va="center", ha="left", fontsize=10, fontweight="bold")
        ax_gt.text(0.20, pos, labels[obj_idx],
                   va="center", ha="left", fontsize=10)
        ax_gt.text(0.55, pos, f"d={gt_distances[obj_idx]:.3f}",
                   va="center", ha="left", fontsize=9, color="dimgray")
        delta = rank_change[obj_idx]
        if delta > 0:
            arrow, a_col = f"▲ +{delta}", "#1a7a1a"
        elif delta < 0:
            arrow, a_col = f"▼ {delta}", "#aa1111"
        else:
            arrow, a_col = "  —", "gray"
        ax_gt.text(0.82, pos, arrow,
                   va="center", ha="left", fontsize=9,
                   fontweight="bold", color=a_col)

    # --- Panel 2: Predicted ranked list ---
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(-0.5, K - 0.5)
    ax_pred.invert_yaxis()
    ax_pred.axis("off")
    ax_pred.set_title("Predicted Ranking", fontsize=11, fontweight="bold")

    for pos, obj_idx in enumerate(pred_order):
        color = _change_color(norm_change[obj_idx])
        ax_pred.add_patch(plt.Rectangle(
            (0, pos - 0.45), 1, 0.9,
            color=color, alpha=0.25, transform=ax_pred.transData,
        ))
        ax_pred.text(0.04, pos, f"#{pos + 1}",
                     va="center", ha="left", fontsize=10, fontweight="bold")
        ax_pred.text(0.20, pos, labels[obj_idx],
                     va="center", ha="left", fontsize=10)
        ax_pred.text(0.55, pos, f"d={pred_distances[obj_idx]:.3f}",
                     va="center", ha="left", fontsize=9, color="dimgray")
        ax_pred.text(0.82, pos, f"GT#{gt_rank[obj_idx] + 1}",
                     va="center", ha="left", fontsize=9, color="steelblue")

    # --- Panel 3: Rank-change bar chart ---
    obj_colors = [_change_color(norm_change[k]) for k in range(K)]
    bars = ax_bar.barh(range(K), rank_change, color=obj_colors,
                       edgecolor="black", linewidth=0.5)
    ax_bar.set_yticks(range(K))
    ax_bar.set_yticklabels(labels, fontsize=max(6, 9 - K // 10))
    ax_bar.axvline(0, color="black", linewidth=1.2)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Rank change  (positive = ranked closer in prediction)", fontsize=9)
    ax_bar.set_title("Per-Object Rank Change  (GT rank − Pred rank)", fontsize=10)
    for k, bar in enumerate(bars):
        val = rank_change[k]
        xoff = 0.1 if val >= 0 else -0.1
        ax_bar.text(val + xoff, k, f"{val:+d}",
                    va="center",
                    ha="left" if val >= 0 else "right",
                    fontsize=8, fontweight="bold")

    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[
            Patch(facecolor=_change_color(1.0),  alpha=0.7, label="Ranked closer (↑ in pred)"),
            Patch(facecolor=_change_color(-1.0), alpha=0.7, label="Ranked farther (↓ in pred)"),
            Patch(facecolor=_change_color(0.0),  alpha=0.7, label="No rank change"),
        ],
        loc="lower right", fontsize=8,
    )

    # --- Panel 4: Scene overlay coloured by rank change ---
    overlay = image_np.astype(np.float32) / 255.0
    for k in range(K):
        mask  = masks[k] > 0
        color = np.array(_change_color(norm_change[k]), dtype=np.float32)
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * color
        ys, xs = np.where(mask)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            ax_scene.text(
                cx, cy,
                f"G{gt_rank[k]+1}→P{pred_rank[k]+1}",
                fontsize=max(5, 8 - K // 8),
                ha="center", va="center", color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5),
            )
    ax_scene.imshow(np.clip(overlay, 0, 1))
    ax_scene.set_title(
        "Scene Overlay — colour = rank change\nLabel: GT rank → Pred rank",
        fontsize=10,
    )
    ax_scene.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ---------------------------------------------------------------------------
# H5-mode visualization
# ---------------------------------------------------------------------------

def render_h5_result_figure(
    image_np: np.ndarray,
    masks: np.ndarray,
    pred_distances: np.ndarray,
    gt_distances: np.ndarray,
    instruction: str,
) -> np.ndarray:
    """
    Four-panel figure for H5 demo:
      1. Input frame
      2. GT costmap  (normalised PL scores painted onto masks)
      3. Predicted costmap
      4. Ranking accuracy breakdown (bar chart sorted by GT rank,
         colour-coded by agreement with predicted rank)
    """
    K, H, W = masks.shape

    # Build per-pixel costmap maps (background stays NaN → white)
    gt_map   = np.full((H, W), np.nan, dtype=np.float32)
    pred_map = np.full((H, W), np.nan, dtype=np.float32)
    for k in range(K):
        px           = masks[k] > 0
        gt_map[px]   = gt_distances[k]
        pred_map[px] = pred_distances[k]

    # Ranking accuracy: fraction of pairs with the same relative order
    n_pairs = n_correct = 0
    for i in range(K):
        for j in range(i + 1, K):
            n_pairs += 1
            if (gt_distances[i] < gt_distances[j]) == (pred_distances[i] < pred_distances[j]):
                n_correct += 1
    ranking_acc = n_correct / n_pairs if n_pairs > 0 else 1.0

    # Shared colour scale (0 → 1 since both are normalised)
    cmap_shared = "RdYlGn_r"

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle(
        f'Instruction: "{instruction}"\n'
        f'{K} objects  |  Ranking Accuracy: {ranking_acc:.3f} '
        f'({n_correct}/{n_pairs} pairs correct)',
        fontsize=11,
    )

    # --- Panel 1: Input frame ---
    axes[0].imshow(image_np)
    axes[0].set_title("Input Frame")
    axes[0].axis("off")

    # --- Panel 2: GT costmap ---
    im1 = axes[1].imshow(gt_map, cmap=cmap_shared, vmin=0, vmax=1)
    axes[1].set_title("GT Costmap\n(normalised PL score)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="GT distance")

    # --- Panel 3: Predicted costmap ---
    im2 = axes[2].imshow(pred_map, cmap=cmap_shared, vmin=0, vmax=1)
    axes[2].set_title("Predicted Costmap")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="Pred distance")

    # --- Panel 4: Ranking accuracy breakdown ---
    ax4 = axes[3]
    gt_order   = np.argsort(gt_distances)               # [K] object indices sorted GT
    pred_ranks = np.argsort(np.argsort(pred_distances)) # [K] 0-based predicted rank per object

    bar_colors = []
    for pos, k in enumerate(gt_order):
        gt_r   = int(pos)
        pred_r = int(pred_ranks[k])
        diff   = abs(gt_r - pred_r)
        if diff == 0:
            bar_colors.append("#2ca02c")    # green  — exact match
        elif diff == 1:
            bar_colors.append("#ff7f0e")    # orange — off by one
        else:
            bar_colors.append("#d62728")    # red    — larger error

    ax4.barh(range(K), gt_distances[gt_order], color=bar_colors, alpha=0.80,
             edgecolor="black", linewidth=0.4)
    ax4.set_yticks(range(K))
    ax4.set_yticklabels(
        [f"GT#{p+1}: obj_{k}  →  pred#{pred_ranks[k]+1}"
         for p, k in enumerate(gt_order)],
        fontsize=max(5, 8 - K // 8),
    )
    ax4.invert_yaxis()
    ax4.set_xlabel("GT distance (normalised)")
    ax4.set_title(
        f"Ranking  (Acc = {ranking_acc:.3f})\n"
        "green = exact, orange = ±1, red = larger",
        fontsize=9,
    )
    ax4.set_xlim(0, 1.15)

    from matplotlib.patches import Patch
    ax4.legend(
        handles=[
            Patch(facecolor="#2ca02c", alpha=0.8, label="Exact rank match"),
            Patch(facecolor="#ff7f0e", alpha=0.8, label="Off by 1"),
            Patch(facecolor="#d62728", alpha=0.8, label="Larger error"),
        ],
        loc="lower right",
        fontsize=7,
    )

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ---------------------------------------------------------------------------
# H5 inference function called by Gradio
# ---------------------------------------------------------------------------

def run_h5_inference(
    h5_path: str,
    base_dir: str,
    episode_id: str,
    frame_idx: int,
    instruction: str,
    checkpoint_path: str,
    device: str,
):
    """Load one H5 frame and produce the 4-panel H5 result figure."""
    h5_path    = (h5_path or "").strip()
    base_dir   = (base_dir or "").strip() or None
    episode_id = (episode_id or "").strip()
    instruction = (instruction or "").strip()

    if not h5_path:
        return None, "Please enter the path to the HDF5 file."
    if not os.path.isfile(h5_path):
        return None, f"HDF5 file not found: {h5_path}"
    if base_dir and not os.path.isdir(base_dir):
        return None, f"Base directory not found: {base_dir}"
    if not episode_id:
        return None, "Please enter an episode folder name (e.g. '1S7LAXRdDqK_0000000_plant_42_')."
    if not instruction:
        return None, "Please enter a navigation instruction."

    try:
        predictor = get_predictor(checkpoint_path, device)
        image_pil, masks, pred_dist, gt_dist, costmap, attn = predictor.predict_from_h5(
            h5_path=h5_path,
            episode_id=episode_id,
            frame_idx=int(frame_idx),
            instruction=instruction,
            base_dir=base_dir,
        )
    except (KeyError, FileNotFoundError) as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Inference error: {exc}"

    if len(pred_dist) == 0:
        return None, "No valid objects in this frame after mask filtering."

    image_np  = np.array(image_pil)
    result_fig = render_h5_result_figure(image_np, masks, pred_dist, gt_dist, instruction)

    n_pairs    = len(pred_dist) * (len(pred_dist) - 1) // 2
    n_correct  = sum(
        1 for i in range(len(pred_dist))
        for j in range(i + 1, len(pred_dist))
        if (gt_dist[i] < gt_dist[j]) == (pred_dist[i] < pred_dist[j])
    )
    ranking_acc = n_correct / n_pairs if n_pairs > 0 else 1.0
    mae         = float(np.abs(pred_dist - gt_dist).mean())

    status = (
        f"Done.  {len(pred_dist)} objects  |  "
        f"Ranking Accuracy: {ranking_acc:.4f}  |  MAE vs GT: {mae:.4f}"
    )
    return result_fig, status


# ---------------------------------------------------------------------------
# Main inference function called by Gradio
# ---------------------------------------------------------------------------

def run_inference(
    image_pil: Image.Image,
    masks_path: str,
    instruction: str,
    checkpoint_path: str,
    device: str,
):
    if image_pil is None:
        return None, None, None, None, "Please upload an image."
    if not masks_path or not masks_path.strip():
        return None, None, None, None, "Please enter the path to a masks .npy file ([K, H, W])."
    if not instruction or not instruction.strip():
        return None, None, None, None, "Please enter a navigation instruction."

    masks_path = masks_path.strip()
    if not os.path.isfile(masks_path):
        return None, None, None, None, f"Masks file not found: {masks_path}"

    # Load masks from the .npy file — same as inference.py CLI
    try:
        masks = load_masks(masks_path)   # [K_valid, H, W]
    except Exception as e:
        return None, None, None, None, f"Failed to load masks: {e}"

    if masks.shape[0] == 0:
        return None, None, None, None, "All masks are empty after filtering."

    predictor = get_predictor(checkpoint_path, device)
    image_np = np.array(image_pil.convert("RGB"))

    # predict_frame handles the rest identically to inference.py
    distances, costmap, attn = predictor.predict_frame(image_pil, masks, instruction)

    if len(distances) == 0:
        return None, None, None, None, "No valid objects detected in the image."

    costmap_fig = render_costmap_figure(image_np, costmap, instruction, len(distances))
    per_obj_fig = render_per_object_figure(image_np, masks, distances)
    attn_fig    = render_attention_figure(instruction, attn, len(distances)) if attn is not None else None

    # --- GT difference figure (optional) ---
    gt_diff_fig    = None
    ranking_fig    = None
    gt_distances   = load_gt_distances(masks_path)
    if gt_distances is not None:
        if len(gt_distances) == len(distances):
            gt_diff_fig = render_gt_diff_figure(image_np, masks, distances, gt_distances)
            ranking_fig = render_ranking_comparison_figure(image_np, masks, distances, gt_distances)
        else:
            print(
                f"[app] GT distances length {len(gt_distances)} != "
                f"predicted {len(distances)} — skipping diff panel."
            )

    status = (
        f"Done.  {len(distances)} objects | "
        f"min dist={distances.min():.4f}  max dist={distances.max():.4f}"
    )
    if gt_distances is not None and gt_diff_fig is not None:
        mae = float(np.abs(distances - gt_distances).mean())
        status += f"  |  MAE vs GT={mae:.4f}"
    elif gt_distances is None:
        status += "  |  geodesic_distances.npy not found (no GT diff)"

    return costmap_fig, per_obj_fig, attn_fig, gt_diff_fig, ranking_fig, status


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_app(checkpoint_path: str, device: str = None, share: bool = False):
    import gradio as gr
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Pre-loading model from {checkpoint_path} on {device} …")
    get_predictor(checkpoint_path, device)

    with gr.Blocks(title="LangGeoNet — Costmap Predictor", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🗺️ LangGeoNet — Language-Guided Geodesic Costmap
            Upload a scene image and describe where you want to go.
            The model will predict geodesic distances to every object and render a navigation costmap.
            """
        )

        with gr.Tabs():


            # ----------------------------------------------------------------
            # Tab 1 — H5 dataset inference
            # ----------------------------------------------------------------
            with gr.Tab("IIN Dataset Demo"):
                gr.Markdown(
                    """
                    ### IIN Dataset Demo
                    Select an episode and frame from the HDF5 costmap file.
                    The model will run inference with your custom instruction
                    and compare the prediction against the stored GT path-length scores.

                    **HDF5 key format**: `ep_{episode_id}_{frame_idx}`  
                    **RGB image path**: `<h5_dir>/ep_{episode_id}/images/{frame_idx:05d}.png`
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        h5_path_input = gr.Textbox(
                            label="HDF5 file path  (*.h5)",
                            placeholder="/path/to/bigger_bot_0.3-sh_0.4_gt_topometric_e3d_masks_pls.h5",
                            lines=1,
                        )
                        h5_base_dir_input = gr.Textbox(
                            label="Base directory  (contains trajectories/ sub-folder)",
                            placeholder="/path/to/training/bigger_bot_0.3-sh_0.4",
                            lines=1,
                        )
                        episode_input = gr.Textbox(
                            label="Episode folder name  (as in H5 key)",
                            placeholder='e.g. "1S7LAXRdDqK_0000000_plant_42_"',
                            lines=1,
                        )
                        frame_input = gr.Number(
                            label="Frame index  (0-based)",
                            value=0,
                            precision=0,
                            minimum=0,
                        )
                        h5_instruction_input = gr.Textbox(
                            label="Navigation Instruction",
                            placeholder='e.g. "Go to the chair near the window"',
                            lines=2,
                        )
                        h5_run_btn   = gr.Button("Run H5 Inference", variant="primary")
                        h5_status    = gr.Textbox(label="Status", interactive=False, lines=2)

                    with gr.Column(scale=3):
                        h5_result_out = gr.Image(
                            label="Frame  |  GT Heatmap  |  Predicted Costmap  |  Ranking Accuracy",
                            type="numpy",
                        )

                h5_run_btn.click(
                    fn=lambda h5p, bdir, ep, fr, instr: run_h5_inference(
                        h5p, bdir, ep, fr, instr, checkpoint_path, device
                    ),
                    inputs=[h5_path_input, h5_base_dir_input, episode_input, frame_input, h5_instruction_input],
                    outputs=[h5_result_out, h5_status],
                    api_name=False,
                )

                _H5  = "/data/ws/VLN-CE/controller/object_react/training/bigger_bot_0.3-sh_0.4/bigger_bot_0.3-sh_0.4_gt_topometric_e3d_masks_pls.h5"
                _DIR = "/data/ws/VLN-CE/controller/object_react/training/bigger_bot_0.3-sh_0.4"
                gr.Examples(
                    examples=[
                        [_H5, _DIR, "1S7LAXRdDqK_0000000_plant_42_", 0,
                         "Walk towards the door at the end of the corridor"],
                        [_H5, _DIR, "1S7LAXRdDqK_0000000_plant_42_", 5,
                         "Go to the chair near the window"],
                        [_H5, _DIR, "1S7LAXRdDqK_0003399_chair_44_", 0,
                         "Navigate to the chair"],
                    ],
                    inputs=[h5_path_input, h5_base_dir_input, episode_input, frame_input, h5_instruction_input],
                    label="Example H5 queries",
                )
            # ----------------------------------------------------------------
            # Tab 2 — standard image + masks .npy inference
            # ----------------------------------------------------------------
            with gr.Tab("VLN-CE dataset Demo"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Scene Image (RGB)",
                            height=320,
                        )
                        masks_input = gr.Textbox(
                            label="Masks file path  (.npy, shape [K, H, W])",
                            placeholder="/path/to/masks.npy",
                            lines=1,
                        )
                        instruction_input = gr.Textbox(
                            label="Navigation Instruction",
                            placeholder='e.g. "Go to the lamp on the nightstand"',
                            lines=2,
                        )
                        run_btn = gr.Button("Run Inference", variant="primary")
                        status_box = gr.Textbox(label="Status", interactive=False, lines=1)

                    with gr.Column(scale=2):
                        gt_diff_out = gr.Image(
                            label="GT vs Predicted Distances  (geodesic_distances.npy)",
                            type="numpy",
                        )
                        ranking_out = gr.Image(
                            label="GT vs Predicted Ranking Comparison",
                            type="numpy",
                        )
                        costmap_out = gr.Image(
                            label="Costmap Visualization",
                            type="numpy",
                        )
                        per_obj_out = gr.Image(
                            label="Per-Object Distance Chart",
                            type="numpy",
                        )
                        attn_out = gr.Image(
                            label="Cross-Attention Map",
                            type="numpy",
                        )

                run_btn.click(
                    fn=lambda img, masks, txt: run_inference(img, masks, txt, checkpoint_path, device),
                    inputs=[image_input, masks_input, instruction_input],
                    outputs=[costmap_out, per_obj_out, attn_out, gt_diff_out, ranking_out, status_box],
                    api_name=False,
                )

                gr.Examples(
                    examples=[
                        [None, "masks.npy", "Go to the lamp on the nightstand"],
                        [None, "masks.npy", "Navigate to the chair near the window"],
                        [None, "masks.npy", "Move towards the door at the end of the hallway"],
                    ],
                    inputs=[image_input, masks_input, instruction_input],
                    label="Example Instructions (supply your own image + masks path)",
                )

            

    demo.launch(share=share, show_api=False)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGeoNet Gradio App")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: cuda / cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port to serve on (default: 7860)",
    )
    args = parser.parse_args()

    # Make sure imports from the same package work when run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    build_app(
        checkpoint_path=args.checkpoint,
        device=args.device,
        share=args.share,
    )

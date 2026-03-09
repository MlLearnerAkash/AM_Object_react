"""
create_sample_input.py
======================
Generates and visualises a synthetic sample input for the Object-React VLN
model, illustrating:

    (a) Current RGB frame  (H × W × 3)
    (b) K segmentation masks  (H × W × K) — one binary mask per detected object
        together with a per-mask scalar navigation cost
    (c) Costmap overlay — RGB frame blended with colour-coded cost information
        (the same blend that TopoPaths.create_input() produces and feeds to
        GoalEncoder as ``goal_type = "image_mask_enc"``)

The figure is saved to ``sample_input.png`` and printed to stdout.

Usage
-----
    cd train/
    python create_sample_input.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os

# ── Configuration ─────────────────────────────────────────────────────────── #
H, W = 64, 85       # height × width (matches object_react config image_size)
K = 6               # number of segmentation masks (detected objects)
DIMS = 8            # positional-encoding channels used by TopoPaths / GoalEncoder
OUT_PATH = "sample_input.png"

np.random.seed(42)

# ── 1. Synthetic RGB image ─────────────────────────────────────────────────── #
rgb = np.zeros((H, W, 3), dtype=np.uint8)
rgb[H // 2:, :]           = [210, 180, 140]   # floor  — beige
rgb[:H // 4, :]            = [135, 206, 235]   # ceiling — sky-blue
rgb[H // 4 : H // 2, :]   = [180, 180, 180]   # wall   — grey

# Coloured rectangles represent indoor objects
_object_colors_u8 = [
    (220,  60,  60),   # 0 – sofa     red
    ( 60, 160,  60),   # 1 – plant    green
    ( 60,  60, 220),   # 2 – chair    blue
    (200, 200,  60),   # 3 – tv       yellow
    (200, 120,  60),   # 4 – table    orange
    (140,  60, 200),   # 5 – shelf    purple
]
_object_labels = ["sofa", "plant", "chair", "tv", "table", "shelf"]
_object_bboxes = [               # (y1, y2, x1, x2) in pixel coords
    (H // 3, 2 * H // 3,  2,  18),
    (H // 4, 3 * H // 4, 22,  34),
    (H // 3, 2 * H // 3, 38,  50),
    (H // 4,     H // 2, 54,  68),
    (H // 3, 2 * H // 3, 70,  82),
    (H // 5,     H // 2,  8,  20),
]

for i in range(K):
    y1, y2, x1, x2 = _object_bboxes[i]
    rgb[y1:y2, x1:x2] = _object_colors_u8[i]

# ── 2. K-channel binary segmentation masks ─────────────────────────────────── #
#   masks[h, w, k] == 1  iff pixel (h,w) belongs to object k
masks = np.zeros((H, W, K), dtype=np.uint8)
for i in range(K):
    y1, y2, x1, x2 = _object_bboxes[i]
    masks[y1:y2, x1:x2, i] = 1

# ── 3. Per-mask navigation costs ───────────────────────────────────────────── #
#   Semantics (matches TopoPaths):
#     low value  →  object lies along the preferred navigation path (close to goal)
#     high value →  object is far or obstructing (path-length estimate)
#     value >= 99 → unreachable / outlier  (rendered red in the costmap)
costs = np.array([5.0, 20.0, 12.0, 45.0, 8.0, 99.0])[:K]

# ── 4. Positional-encoded costmap  (mirrors TopoPaths.create_input) ──────────── #
# Each mask channel is weighted by its normalized cost and summed into a
# DIMS-channel feature map (GoalEncoder input).  Here we just show the
# 3-channel colour version that TopoPaths stores as ``plWtColorImg``.

def _cost_to_color(c: float, vmin: float = 0.0, vmax: float = 50.0,
                   cmap_name: str = "winter") -> np.ndarray:
    """Scalar cost → (3,) float RGB.  Outlier (>=99) rendered red."""
    if c >= 99.0:
        return np.array([1.0, 0.0, 0.0])
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return np.array(cmap(norm(c))[:3])


# Weighted blending of per-object colours across overlapping pixels
deno = masks.sum(-1).astype(float)   # (H, W)
deno[deno == 0] = 1.0
costmap_rgb = np.zeros((H, W, 3), dtype=float)
for i in range(K):
    color   = _cost_to_color(costs[i])          # (3,)
    weight  = masks[:, :, i] / deno              # (H, W)
    costmap_rgb += weight[:, :, None] * color    # broadcast
costmap_rgb = np.clip(costmap_rgb, 0.0, 1.0)

# ── 5. Visualise ──────────────────────────────────────────────────────────── #
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle(
    "Object-React VLN — Sample Model Input\n"
    f"Image size: {W}×{H}  |  K={K} segmentation masks  |  DIMS={DIMS} costmap channels",
    fontsize=12, fontweight="bold",
)

# (a) RGB frame
axes[0].imshow(rgb)
axes[0].set_title("(a) Current RGB Frame\n[obs_type = \"image\"]")
axes[0].axis("off")

# (b) Segmentation masks coloured by object label
mask_vis = np.zeros((H, W, 3), dtype=float)
for i in range(K):
    c = np.array(_object_colors_u8[i]) / 255.0
    mask_vis += masks[:, :, i:i+1] * c
mask_vis = np.clip(mask_vis, 0.0, 1.0)

legend_patches = [
    mpatches.Patch(
        color=np.array(_object_colors_u8[i]) / 255.0,
        label=f"{_object_labels[i]}  cost={costs[i]:.0f}",
    )
    for i in range(K)
]
axes[1].imshow(mask_vis)
axes[1].legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)
axes[1].set_title(f"(b) K={K} Segmentation Masks\n(each with navigation cost)")
axes[1].axis("off")

# (c) Costmap overlay (RGB + costmap blend)
blended = np.clip(0.45 * (rgb / 255.0) + 0.55 * costmap_rgb, 0.0, 1.0)
axes[2].imshow(blended)

sm = plt.cm.ScalarMappable(
    cmap="winter", norm=mcolors.Normalize(vmin=0, vmax=50)
)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
cbar.set_label("Navigation Cost\n(red = unreachable ≥99)", fontsize=8)
axes[2].set_title(
    "(c) Costmap Overlay\n[goal_type = \"image_mask_enc\" → GoalEncoder input]"
)
axes[2].axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"[create_sample_input] Saved visualisation → {os.path.abspath(OUT_PATH)}")
plt.close(fig)

# ── 6. Print expected tensor shapes ──────────────────────────────────────────── #
ctx = 5  # context_size from config
T   = 10  # len_traj_pred

print()
print("=" * 65)
print("  Tensor shapes as fed to the Object-React VLN model")
print("=" * 65)
print(f"  obs_image   (obs_type='image')         : [{3*(ctx+1):3d}, {H:2d}, {W:2d}]"
      f"  — {ctx+1} RGB frames stacked channel-wise")
print(f"  goal_image  (goal_type='image_mask_enc'): [{3+DIMS:3d}, {H//2:2d}, {W//2:2d}]"
      f"  — 3-ch RGB vis + {DIMS}-ch positional-encoded costmap")
print(f"  action_label (discrete_actions=True)    : [{T:3d}]"
      f"  — int64,  0=STOP  1=FORWARD  2=LEFT  3=RIGHT")
print(f"  action_logits (model output)            : [{T:3d},  4]"
      f"  — raw logits,  argmax → predicted action per step")
print(f"  dist_label                              : [  1]  — int64 (timesteps to goal)")
print()
print("  Weight transfer from pretrained object_react checkpoint:")
print("    GoalEncoder (processes costmap)   → FULL TRANSFER")
print("    linear_layers + dist_predictor    → FULL TRANSFER")
print("    discrete_action_predictor         → random init  (new head)")
print("    obs MobileNet (RGB encoder)       → random init  (new stream)")
print("=" * 65)

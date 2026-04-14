import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# CLIP image normalization constants (ViT-B/16 defaults)
_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _denorm_clip(tensor):
    """CHW float32 CLIP-normalised tensor → HWC uint8 numpy (224×224×3)."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)  # HWC
    img = img * _CLIP_STD + _CLIP_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _resize_masks_to(masks_np, H, W):
    """[K, Hm, Wm] bool → [K, H, W] bool (nearest-neighbour, via PIL)."""
    from PIL import Image as PILImage
    K = masks_np.shape[0]
    if K == 0 or (masks_np.shape[1] == H and masks_np.shape[2] == W):
        return masks_np
    return np.stack([
        np.array(
            PILImage.fromarray(masks_np[k].astype(np.uint8))
                    .resize((W, H), PILImage.NEAREST)
        ).astype(bool)
        for k in range(K)
    ])


def _cost_overlay(rgb_hwc, masks_np, costs, alpha=0.5):
    """
    Return [H, W, 3] float32 in [0, 1] — RGB with per-mask heat overlay.
    costs: [K] float32 in [0, 1].
    """
    H, W = rgb_hwc.shape[:2]
    K    = masks_np.shape[0]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "grn_red", ["#00cc44", "#ffff00", "#ff2200"]
    )
    img      = rgb_hwc.astype(np.float32) / 255.0
    canvas   = np.full((H, W), np.nan, dtype=np.float32)
    seg_mask = np.zeros((H, W), dtype=bool)

    for k in range(K):
        m = masks_np[k]
        if not m.any():
            continue
        c = float(costs[k])
        canvas[m]  = np.clip(c, 0.0, 1.0) if np.isfinite(c) else 0.5
        seg_mask  |= m

    if seg_mask.any():
        filled  = np.where(np.isnan(canvas), 0.5, canvas)
        heat    = cmap(filled)[:, :, :3]          # [H, W, 3]
        blended = (1 - alpha) * img + alpha * heat
        img     = np.where(seg_mask[:, :, None], blended, img)

    # white contours
    for k in range(K):
        m = masks_np[k]
        if not m.any():
            continue
        pad = np.pad(m.astype(np.uint8), 1, mode="constant")
        nb  = pad[:-2,1:-1] + pad[2:,1:-1] + pad[1:-1,:-2] + pad[1:-1,2:]
        cont = m & (nb < 4)
        img = np.where(cont[:, :, None], 1.0, img)

    return np.clip(img, 0.0, 1.0)


def _annotate_costs(ax, masks_np, costs):
    """Write cost value at each mask centroid."""
    K = masks_np.shape[0]
    for k in range(K):
        m = masks_np[k]
        if not m.any():
            continue
        ys, xs = np.where(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        c   = float(costs[k])
        lbl = f"{c:.2f}" if np.isfinite(c) else "∞"
        ax.text(cx, cy, lbl, fontsize=6, color="white",
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))


def make_joint_val_canvas(
    pixel_values,   # [3, 224, 224] CLIP-normalised tensor (CPU)
    masks,          # [K, Hm, Wm] bool tensor (CPU) — original resolution
    gt_costs,       # [K] float32 tensor (CPU), normalised [0,1]
    pred_costs,     # [K] float32 tensor (CPU), already sigmoid → [0,1]
    gt_action,      # [T, 4] float32 tensor (CPU) — local robot frame
    pred_action,    # [T, 4] float32 tensor (CPU) — local robot frame
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Render a single-row three-panel canvas:
      Panel 1 — GT cost heat-overlay on the (denormalised) RGB image
      Panel 2 — Predicted cost heat-overlay on the same RGB image
      Panel 3 — Top-down waypoint trajectory
                 (green solid = GT,  red dashed = Pred,  black ▲ = robot)

    Returns a HxW×3 uint8 numpy array suitable for wandb.Image.
    """
    rgb      = _denorm_clip(pixel_values)          # [224, 224, 3] uint8
    H, W     = rgb.shape[:2]
    masks_np = _resize_masks_to(masks.cpu().numpy().astype(bool), H, W)  # [K, H, W]
    gt_c     = gt_costs.cpu().numpy().astype(np.float32)
    pr_c     = pred_costs.cpu().numpy().astype(np.float32)
    gt_wp    = gt_action.cpu().numpy()             # [T, 4]: (x_fwd, y_left, cos, sin)
    pred_wp  = pred_action.cpu().numpy()

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "grn_red", ["#00cc44", "#ffff00", "#ff2200"]
    )

    gt_ov = _cost_overlay(rgb, masks_np, gt_c, alpha)
    pr_ov = _cost_overlay(rgb, masks_np, pr_c, alpha)

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.08)

    # ---- Panel 1: GT cost overlay ----------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(gt_ov)
    _annotate_costs(ax0, masks_np, gt_c)
    ax0.set_title("GT cost", fontsize=11, fontweight="bold")
    ax0.axis("off")

    # ---- Panel 2: Predicted cost overlay ---------------------------------
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(pr_ov)
    _annotate_costs(ax1, masks_np, pr_c)
    ax1.set_title("Pred cost", fontsize=11, fontweight="bold")
    ax1.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=[ax0, ax1], fraction=0.025, pad=0.02,
                 label="cost  (0 = low/green   1 = high/red)")

    # ---- Panel 3: Top-down trajectory ------------------------------------
    ax2 = fig.add_subplot(gs[2])
    # Robot frame: x = forward (plotted on Y-axis), y = left (X-axis)
    gt_x,   gt_y   = gt_wp[:, 0],   gt_wp[:, 1]
    pred_x, pred_y = pred_wp[:, 0], pred_wp[:, 1]

    ax2.plot([0, *gt_y],    [0, *gt_x],   "o-",  color="#00cc44",
             lw=1.8, ms=5, label="GT traj")
    ax2.plot([0, *pred_y],  [0, *pred_x], "s--", color="#ff2200",
             lw=1.8, ms=5, label="Pred traj")
    ax2.plot(0, 0, "k^", ms=10, zorder=5, label="Robot")

    all_x = np.concatenate([[0], gt_x, pred_x])
    all_y = np.concatenate([[0], gt_y, pred_y])
    pad   = max(0.5, float(np.abs(np.concatenate([all_x, all_y])).max()) * 0.15)
    ax2.set_xlim(all_y.min() - pad, all_y.max() + pad)
    ax2.set_ylim(all_x.min() - pad, all_x.max() + pad)
    ax2.set_xlabel("y  (left +)", fontsize=9)
    ax2.set_ylabel("x  (forward +)", fontsize=9)
    ax2.set_title("Trajectory  (local robot frame)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.canvas.draw()
    fw, fh = fig.canvas.get_width_height()
    canvas_img = (
        np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        .reshape(fh, fw, 3).copy()
    )
    plt.close(fig)
    return canvas_img


def _extract_sample_masks_and_rgb(batch, sample_idx):
    """Return (frame_rgb, masks_arr, node_ids) for one sample in a collated batch."""
    frame_rgb = batch.get("frame_rgbs", [None])[sample_idx]
    masks_arr = batch["masks_list"][sample_idx].cpu().numpy()   # [K, H, W]
    return frame_rgb, masks_arr, list(range(masks_arr.shape[0]))


def _render_cost_comparison_image(frame_rgb, masks_arr, gt_costs, pred_costs, alpha=0.5):
    """Side-by-side GT | PRED overlay using overlay_costs style:
    green (low cost) → yellow → red (high cost), white contours, centroid labels."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    H, W = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
    K = 0 if masks_arr is None else masks_arr.shape[0]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'green_red', ['#00cc44', '#ffff00', '#ff2200']   # green → yellow → red
    )

    def _make_overlay(costs):
        overlay      = frame_rgb.astype(np.float32) / 255.0
        cost_canvas  = np.full((H, W), np.nan, dtype=np.float32)
        segment_mask = np.zeros((H, W), dtype=bool)

        for k in range(K):
            m = masks_arr[k].astype(bool)
            if not m.any():
                continue
            c = float(costs[k])
            cost_canvas[m] = 0.5 if not np.isfinite(c) else c
            segment_mask  |= m

        if segment_mask.any():
            canvas_filled = np.where(np.isnan(cost_canvas), 0.5, cost_canvas)
            rgba_heat = np.zeros((H, W, 4), dtype=np.float32)
            # avoid boolean fancy-indexed in-place assignment (segfaults on old numpy)
            heat_vals = cmap(canvas_filled)                  # [H, W, 4] full array
            rgba_heat = np.where(segment_mask[:, :, None], heat_vals, rgba_heat)
            blended = (1 - alpha) * overlay + alpha * rgba_heat[:, :, :3]
            overlay = np.where(segment_mask[:, :, None], blended, overlay)

        # white contours
        contour_mask = np.zeros((H, W), dtype=bool)
        for k in range(K):
            m = masks_arr[k].astype(bool)
            if not m.any():
                continue
            pad   = np.pad(m.astype(np.uint8), 1, mode='constant')
            neigh = pad[:-2,1:-1] + pad[2:,1:-1] + pad[1:-1,:-2] + pad[1:-1,2:]
            contour_mask |= m & (neigh < 4)
        overlay = np.where(contour_mask[:, :, None], 1.0, overlay)

        return overlay

    gt_overlay = _make_overlay(gt_costs)
    pd_overlay = _make_overlay(pred_costs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, overlay, costs, title in [
        (axes[0], gt_overlay, gt_costs, "GT"),
        (axes[1], pd_overlay, pred_costs, "PRED"),
    ]:
        ax.imshow(np.clip(overlay, 0, 1))
        for k in range(K):
            m = masks_arr[k].astype(bool)
            if not m.any():
                continue
            ys, xs = np.where(m)
            cy, cx = int(ys.mean()), int(xs.mean())
            c = float(costs[k])
            label = f"{c:.2f}" if np.isfinite(c) else "∞"
            ax.text(cx, cy, label, fontsize=7, color='white', ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.4, lw=0))
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Normalised cost  (0 = low / green,  1 = high / red)', fontsize=8)
    fig.tight_layout()

    fig.canvas.draw()
    fig_w, fig_h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig_h, fig_w, 3).copy()
    plt.close(fig)
    return img

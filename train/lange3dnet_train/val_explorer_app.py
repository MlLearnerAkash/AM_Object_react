"""
LangGeoNet — Validation Dataset Explorer (Gradio App)

Interactively browse the H5EpisodePathLengthsDataset validation split,
change the navigation instruction, and see how the predicted cost heatmap
responds on the original frame.

Usage
-----
    python val_explorer_app.py \
        --h5_path   /path/to/episodes.h5 \
        --checkpoint checkpoints/best_model.pt

    # with public link
    python val_explorer_app.py \
        --h5_path   /path/to/episodes.h5 \
        --checkpoint checkpoints/best_model.pt \
        --share
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

# ── gradio_client 1.3.0 compatibility patch ───────────────────────────────────
try:
    import gradio_client.utils as _gc_utils

    _orig_j2p = _gc_utils._json_schema_to_python_type

    def _safe_j2p(schema, defs=None):
        if not isinstance(schema, dict):
            return "any"
        return _orig_j2p(schema, defs)

    _gc_utils._json_schema_to_python_type = _safe_j2p
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import H5EpisodePathLengthsDataset, create_h5_episode_pathlengths_dataloader
from model import LangGeoNetV2

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singletons (initialised once at app startup)
# ─────────────────────────────────────────────────────────────────────────────

_model: LangGeoNetV2 | None = None
_val_dataset: H5EpisodePathLengthsDataset | None = None
_clip_processor: CLIPProcessor | None = None
_device: torch.device | None = None

_episode_ids: list[str] = []
_ep_sample_indices: dict[str, list[int]] = {}
_ep_instructions: dict[str, str] = {}
_h5_path: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Startup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_instruction_from_h5(h5_path: str, ep_id: str) -> str:
    with h5py.File(h5_path, "r") as hf:
        if ep_id not in hf:
            return ""
        raw = hf[ep_id]["instruction"][()]
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def load_model_and_dataset(
    checkpoint_path: str,
    h5_path: str,
    device_str: str | None = None,
    val_split: float = 0.2,
    seed: int = 42,
    max_episodes: int | None = None,
) -> None:
    global _model, _val_dataset, _clip_processor, _device
    global _episode_ids, _ep_sample_indices, _ep_instructions, _h5_path

    _h5_path = h5_path

    _device = torch.device(
        device_str if device_str and device_str != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[App] device: {_device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    _model = LangGeoNetV2(
        d_model        = cfg["d_model"],
        n_heads        = cfg["n_heads"],
        n_layers       = cfg["n_layers"],
        clip_model_name= cfg["clip_model"],
    )
    _model.load_state_dict(ckpt["model_state_dict"])
    _model = _model.to(_device)
    _model.eval()

    _clip_processor = CLIPProcessor.from_pretrained(cfg["clip_model"])
    print(f"[App] model loaded — epoch {ckpt['epoch']}, "
          f"val MAE={ckpt.get('best_val_mae', '?')}")

    print("[App] building val dataset …")
    _, val_loader = create_h5_episode_pathlengths_dataloader(
        h5_path     = h5_path,
        batch_size  = 1,
        num_workers = 0,
        val_split   = val_split,
        seed        = seed,
    )
    _val_dataset = val_loader.dataset

    for idx in range(len(_val_dataset)):
        ep_id = _val_dataset.frames[idx]['ep_id']
        _ep_sample_indices.setdefault(ep_id, []).append(idx)
    _episode_ids[:] = sorted(_ep_sample_indices.keys())
    if max_episodes is not None:
        _episode_ids[:] = _episode_ids[:max_episodes]

    print(f"[App] caching instructions for {len(_episode_ids)} episodes …")
    for ep_id in _episode_ids:
        try:
            _ep_instructions[ep_id] = _read_instruction_from_h5(h5_path, ep_id)
        except Exception as exc:
            _ep_instructions[ep_id] = ""
            print(f"[App] Warning: instruction read failed for {ep_id}: {exc}")

    print(f"[App] ready — {len(_episode_ids)} val episodes, "
          f"{len(_val_dataset)} total frames")


# ─────────────────────────────────────────────────────────────────────────────
# GT cost computation  (mirrors _prepare_batch in train.py)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_gt_costs(node_registry: dict, node_ids: list) -> np.ndarray:
    raw = []
    for nid in node_ids:
        pr   = np.asarray(node_registry[nid].path_row, dtype=np.float64)
        cost = float(np.nanmean(pr)) if pr.size else np.nan
        raw.append(cost)

    finite       = [c for c in raw if np.isfinite(c)]
    c_min, c_max = (min(finite), max(finite)) if len(finite) > 1 else (0.0, 1.0)

    normed = []
    for c in raw:
        if not np.isfinite(c):
            normed.append(np.nan)
        elif c_max == c_min:
            normed.append(0.0)
        else:
            normed.append((c - c_min) / (c_max - c_min))
    return np.array(normed, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "gcr", ["#00cc44", "#ffff00", "#ff2200"]
)


def _heatmap_overlay(
    frame_rgb:    np.ndarray,
    masks_arr:    np.ndarray,
    costs:        np.ndarray,
    ranks:        np.ndarray,
    ax:           "plt.Axes",
    title:        str,
    alpha:        float = 0.55,
    show_gt_rank: np.ndarray | None = None,
) -> None:
    K, H, W     = masks_arr.shape
    overlay      = frame_rgb.astype(np.float32) / 255.0
    cost_canvas  = np.full((H, W), np.nan, dtype=np.float32)
    segment_mask = np.zeros((H, W), dtype=bool)
    contour_mask = np.zeros((H, W), dtype=bool)

    for k in range(K):
        m = masks_arr[k].astype(bool)
        if not m.any():
            continue
        c = float(costs[k])
        cost_canvas[m] = 0.5 if not np.isfinite(c) else c
        segment_mask  |= m
        pad   = np.pad(m.astype(np.uint8), 1, mode="constant")
        neigh = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        contour_mask |= m & (neigh < 4)

    if segment_mask.any():
        canvas_filled = np.where(np.isnan(cost_canvas), 0.5, cost_canvas)
        heat_vals     = _CMAP(canvas_filled)[:, :, :3]
        blended       = (1.0 - alpha) * overlay + alpha * heat_vals
        overlay       = np.where(segment_mask[:, :, None], blended, overlay)
    overlay = np.where(contour_mask[:, :, None], 1.0, overlay)

    ax.imshow(np.clip(overlay, 0, 1))
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    for k in range(K):
        m = masks_arr[k].astype(bool)
        if not m.any():
            continue
        ys, xs = np.where(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        c      = float(costs[k])
        r      = int(ranks[k])
        label  = f"#{r + 1}\n{c:.2f}" if r >= 0 else "NaN"
        if show_gt_rank is not None and show_gt_rank[k] >= 0:
            label += f"\n[GT#{int(show_gt_rank[k]) + 1}]"
        ax.text(
            cx, cy, label,
            fontsize=max(5, 7 - K // 10),
            color="white", ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.55, lw=0),
        )


def render_explorer_figure(
    frame_rgb:   np.ndarray,
    masks_arr:   np.ndarray,
    pred_costs:  np.ndarray,
    gt_costs:    np.ndarray,
    instruction: str,
) -> np.ndarray:
    K = masks_arr.shape[0]

    def _ranks(costs: np.ndarray) -> np.ndarray:
        valid = np.isfinite(costs)
        r = np.full(K, -1, dtype=int)
        if valid.any():
            r[valid] = np.argsort(np.argsort(costs[valid]))
        return r

    gt_ranks   = _ranks(gt_costs)
    pred_ranks = _ranks(pred_costs)

    valid_pairs = [
        (i, j)
        for i in range(K) for j in range(i + 1, K)
        if gt_ranks[i] >= 0 and gt_ranks[j] >= 0
    ]
    rank_acc_str = ""
    if valid_pairs:
        correct  = sum(
            1 for i, j in valid_pairs
            if (pred_costs[i] < pred_costs[j]) == (gt_costs[i] < gt_costs[j])
        )
        rank_acc     = correct / len(valid_pairs)
        mae          = float(np.nanmean(np.abs(pred_costs - gt_costs)))
        rank_acc_str = (
            f"Ranking Acc: {rank_acc:.3f}  ({correct}/{len(valid_pairs)} pairs)  "
            f"| MAE: {mae:.4f}"
        )

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(22, 14))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[6, 5], hspace=0.30, wspace=0.08)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_gt   = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[0, 2])
    ax_bar  = fig.add_subplot(gs[1, :])

    instr_display = instruction if len(instruction) <= 120 else instruction[:120] + "…"
    fig.suptitle(
        f'Instruction: "{instr_display}"\n{K} objects  |  {rank_acc_str}',
        fontsize=10, y=0.99,
    )

    ax_orig.imshow(frame_rgb)
    ax_orig.set_title("Original Frame", fontsize=10)
    ax_orig.axis("off")

    _heatmap_overlay(frame_rgb, masks_arr, gt_costs, gt_ranks,
                     ax=ax_gt, title="GT Cost  (lower = closer to goal)")
    _heatmap_overlay(frame_rgb, masks_arr, pred_costs, pred_ranks,
                     ax=ax_pred, title="Predicted Cost  (#rank  value  [GT#rank])",
                     show_gt_rank=gt_ranks)

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_gt, ax_pred], fraction=0.025, pad=0.01)
    cbar.set_label("Normalised cost  (green=low/close · red=high/far)", fontsize=8)

    # ── bar chart ─────────────────────────────────────────────────────────────
    def _bar_color(k: int) -> str:
        gr, pr = int(gt_ranks[k]), int(pred_ranks[k])
        if gr < 0 or pr < 0:
            return "#aaaaaa"
        diff = abs(gr - pr)
        if diff == 0:   return "#2ca02c"
        if diff == 1:   return "#ff7f0e"
        return "#d62728"

    sort_order = np.argsort(gt_costs if np.isfinite(gt_costs).any() else pred_costs)
    gt_vals    = gt_costs[sort_order]
    pred_vals  = pred_costs[sort_order]
    colors_s   = [_bar_color(k) for k in sort_order]
    xlabels    = [
        f"obj_{k}\nGT#{gt_ranks[k]+1 if gt_ranks[k]>=0 else '?'}"
        f"→P#{pred_ranks[k]+1 if pred_ranks[k]>=0 else '?'}"
        for k in sort_order
    ]

    pos   = np.arange(len(sort_order))
    width = 0.38
    ax_bar.bar(pos - width/2, gt_vals,   width=width, color=colors_s,
               alpha=0.80, label="GT cost",   edgecolor="black", linewidth=0.4)
    ax_bar.bar(pos + width/2, pred_vals, width=width, color=colors_s,
               alpha=0.55, label="Pred cost", edgecolor="black", linewidth=0.4,
               linestyle="--")
    ax_bar.set_xticks(pos)
    ax_bar.set_xticklabels(xlabels,
                           fontsize=max(5, 8 - K // 8),
                           rotation=45 if K > 8 else 0, ha="right")
    ax_bar.set_ylabel("Normalised cost")
    ax_bar.set_ylim(0, 1.15)
    ax_bar.set_title(
        "Per-Object Cost: GT vs Predicted  "
        "(sorted by GT rank, green=exact · orange=±1 · red=larger diff)",
        fontsize=9,
    )

    for idx, k in enumerate(sort_order):
        pr, gr = pred_ranks[k], gt_ranks[k]
        ypos   = max(float(gt_costs[k]), float(pred_costs[k])) + 0.02
        ax_bar.text(idx, ypos,
                    f"G{gr+1 if gr>=0 else '?'}→P{pr+1 if pr>=0 else '?'}",
                    ha="center", va="bottom",
                    fontsize=max(5, 7 - K // 10), fontweight="bold", color="black")

    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[
            Patch(facecolor="#2ca02c", alpha=0.8, label="Exact rank match"),
            Patch(facecolor="#ff7f0e", alpha=0.8, label="Rank off by 1"),
            Patch(facecolor="#d62728", alpha=0.8, label="Rank error > 1"),
            plt.Line2D([], [], color="black", linewidth=1, label="GT (solid)"),
            plt.Line2D([], [], color="black", linewidth=1,
                       linestyle="--", label="Pred (dashed border)"),
        ],
        fontsize=8, loc="upper left", ncol=2,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.canvas.draw()
    fw, fh = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fh, fw, 3).copy()
    plt.close(fig)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    episode_1based: int,
    frame_idx:      int,
    instruction:    str,
) -> tuple[np.ndarray | None, str]:
    if _model is None or _val_dataset is None:
        return None, "Model / dataset not initialised."

    ep_id          = _episode_ids[episode_1based - 1]
    sample_indices = _ep_sample_indices[ep_id]
    frame_idx      = max(0, min(int(frame_idx), len(sample_indices) - 1))
    dataset_idx    = sample_indices[frame_idx]
    sample         = _val_dataset[dataset_idx]

    instr     = (instruction or "").strip() or _ep_instructions.get(ep_id, "")
    frame_rgb = sample["frame_rgb"]           # [H, W, 3] uint8
    masks_arr = sample["masks"].numpy()       # [K, H, W] bool
    gt_costs  = sample["gt_costs"].numpy()    # [K] float32
    K         = masks_arr.shape[0]

    if K == 0:
        return None, f"Episode {episode_1based} Frame {frame_idx}: no objects found."

    # ── tokenise instruction ──────────────────────────────────────────────────
    if (instruction or "").strip():
        clip_text      = _clip_processor(
            text=instr, padding="max_length", truncation=True,
            max_length=77, return_tensors="pt",
        )
        input_ids      = clip_text["input_ids"].to(_device)
        attention_mask = clip_text["attention_mask"].to(_device)
    else:
        input_ids      = sample["input_ids"].unsqueeze(0).to(_device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(_device)

    pixel_values = sample["pixel_values"].unsqueeze(0).to(_device)
    masks_t      = torch.from_numpy(masks_arr).to(_device)

    with torch.no_grad():
        preds, _ = _model(pixel_values, [masks_t], input_ids, attention_mask)
    pred_costs = preds[0].float().cpu().numpy()   # [K]
    p_min, p_max = pred_costs.min(), pred_costs.max()
    pred_costs = (pred_costs - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(pred_costs)

    fig_img = render_explorer_figure(frame_rgb, masks_arr, pred_costs, gt_costs, instr)
    vp = [
        (i, j)
        for i in range(K) for j in range(i + 1, K)
        if np.isfinite(gt_costs[i]) and np.isfinite(gt_costs[j])
    ]
    if vp:
        ok     = sum(
            1 for i, j in vp
            if (pred_costs[i] < pred_costs[j]) == (gt_costs[i] < gt_costs[j])
        )
        ra     = ok / len(vp)
        mae    = float(np.nanmean(np.abs(pred_costs - gt_costs)))
        status = (
            f"Episode {episode_1based} ({ep_id}) | Frame {frame_idx} | "
            f"{K} objects | Ranking Acc: {ra:.3f} | MAE: {mae:.4f}"
        )
    else:
        status = (
            f"Episode {episode_1based} ({ep_id}) | Frame {frame_idx} | "
            f"{K} objects | (no GT pairs available)"
        )

    return fig_img, status


def get_frame_preview(episode_1based: int, frame_idx: int) -> np.ndarray | None:
    if _val_dataset is None:
        return None
    ep_id          = _episode_ids[episode_1based - 1]
    sample_indices = _ep_sample_indices[ep_id]
    if not sample_indices:
        return None
    frame_idx   = max(0, min(int(frame_idx), len(sample_indices) - 1))
    dataset_idx = sample_indices[frame_idx]
    return _val_dataset[dataset_idx]["frame_rgb"]


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

def build_app(share: bool = False, server_port: int = 7860) -> None:
    import gradio as gr

    ep_count = len(_episode_ids)
    if ep_count == 0:
        raise RuntimeError("No val episodes found — check h5_path and val_split.")

    first_ep_id    = _episode_ids[0]
    first_n_frames = len(_ep_sample_indices[first_ep_id])
    first_instr    = _ep_instructions.get(first_ep_id, "")

    def _on_episode_change(ep_1based: int):
        ep_id   = _episode_ids[int(ep_1based) - 1]
        n_f     = len(_ep_sample_indices[ep_id])
        default = _ep_instructions.get(ep_id, "")
        info    = f"ID: {ep_id}  ({n_f} frames)"
        return (
            gr.update(maximum=n_f - 1, value=0),
            default,
            info,
        )

    def _on_preview(ep_1based: int, frame_idx: int):
        try:
            return get_frame_preview(int(ep_1based), int(frame_idx))
        except Exception:
            return None

    def _on_run(ep_1based: int, frame_idx: int, instruction: str):
        try:
            fig, status = run_inference(int(ep_1based), int(frame_idx), instruction)
            return fig, fig, status
        except Exception as exc:
            err = f"Error: {exc}\n{traceback.format_exc()}"
            return None, None, err

    def _on_reset_instruction(ep_1based: int):
        ep_id = _episode_ids[int(ep_1based) - 1]
        return _ep_instructions.get(ep_id, "")

    with gr.Blocks(
        title="LangGeoNet — Val Dataset Explorer",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            ## LangGeoNet — Validation Dataset Explorer

            Browse the **validation split** of the H5 episode dataset.
            Select an episode and frame, optionally change the navigation instruction,
            then click **Run Inference** to see the predicted cost heatmap.

            | Colour | Meaning |
            |--------|---------|
            | Green  | Low cost — close to goal |
            | Yellow | Medium cost |
            | Red    | High cost — far from goal |

            Each segment shows: **#rank** · cost value · [GT#rank] (predicted panel).
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                ep_slider = gr.Slider(
                    minimum=1, maximum=ep_count, step=1, value=1,
                    label=f"Episode  (1 – {ep_count})",
                )
                ep_info_box = gr.Textbox(
                    label="Episode Info", interactive=False, lines=1,
                    value=f"ID: {first_ep_id}  ({first_n_frames} frames)",
                )
                frame_slider = gr.Slider(
                    minimum=0, maximum=first_n_frames - 1, step=1, value=0,
                    label="Frame within episode  (0-based)",
                )
                preview_btn = gr.Button("Preview Frame", size="sm")
                gr.Markdown("---")
                instruction_box = gr.Textbox(
                    label="Navigation Instruction  (editable)",
                    lines=4, value=first_instr,
                    placeholder="Enter instruction or use the dataset default …",
                )
                with gr.Row():
                    run_btn   = gr.Button("Run Inference", variant="primary")
                    reset_btn = gr.Button("Reset to Default", size="sm")
                status_box = gr.Textbox(
                    label="Status", interactive=False, lines=3,
                )

            with gr.Column(scale=3):
                preview_out = gr.Image(
                    label="Frame Preview", type="numpy", height=260,
                )
                result_out = gr.Image(
                    label="GT Heatmap  |  Predicted Heatmap  |  Ranking Chart",
                    type="numpy",
                )

        ep_slider.change(
            fn=_on_episode_change,
            inputs=[ep_slider],
            outputs=[frame_slider, instruction_box, ep_info_box],
        )
        preview_btn.click(
            fn=_on_preview, inputs=[ep_slider, frame_slider], outputs=[preview_out],
        )
        frame_slider.change(
            fn=_on_preview, inputs=[ep_slider, frame_slider], outputs=[preview_out],
        )
        run_btn.click(
            fn=_on_run,
            inputs=[ep_slider, frame_slider, instruction_box],
            outputs=[result_out, preview_out, status_box],
        )
        reset_btn.click(
            fn=_on_reset_instruction, inputs=[ep_slider], outputs=[instruction_box],
        )
        demo.load(
            fn=_on_episode_change,
            inputs=[ep_slider],
            outputs=[frame_slider, instruction_box, ep_info_box],
        )

    demo.launch(share=share, server_port=server_port, show_api=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LangGeoNet Validation Dataset Explorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5_path",    required=True,
                        help="Path to the H5 episodes file.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a model checkpoint (.pt) saved by train.py.")
    parser.add_argument("--device",     default=None,
                        help="cuda / cpu (default: auto-detect).")
    parser.add_argument("--val_split",  type=float, default=0.2,
                        help="Val fraction (must match training split).")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed for train/val split (must match training).")
    parser.add_argument("--port",       type=int,   default=7860,
                        help="Local port for the Gradio server.")
    parser.add_argument("--share",        action="store_true",
                        help="Create a public Gradio share link.")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit to first N episodes (default: all).")
    args = parser.parse_args()

    load_model_and_dataset(
        checkpoint_path = args.checkpoint,
        h5_path         = args.h5_path,
        device_str      = args.device,
        val_split       = args.val_split,
        seed            = args.seed,
        max_episodes    = args.max_episodes,
    )
    build_app(share=args.share, server_port=args.port)

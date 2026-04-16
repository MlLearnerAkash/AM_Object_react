"""
joint_val_app.py
================
Gradio app for browsing the joint validation set with LangGeoNetV2 + GNM.

Loads the checkpoint, iterates through the validation split, and shows:
  - GT cost map  (masks coloured by ground-truth path-length cost)
  - Predicted cost map  (masks coloured by model-predicted cost)
  - GT vs Predicted waypoint trajectory  (top-down local robot frame)

Usage (from the train/ directory):
    python joint_val_app.py
    python joint_val_app.py --checkpoint checkpoints/joint/best_joint.pth
    python joint_val_app.py --h5  /path/to/e3d_train_stratified.h5
    python joint_val_app.py --share
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# ---------------------------------------------------------------------------
# Path setup — mirrors joint_train.py so that all imports resolve correctly.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "lange3dnet_train"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = os.path.join(_SCRIPT_DIR, "checkpoints/joint/best_joint.pth")
DEFAULT_H5         = "/media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_train_stratified.h5"
DEFAULT_VAL_SPLIT  = 0.1
DEFAULT_SEED       = 42

# Model config — must match training config
MODEL_CFG = dict(
    # LangGeoNetV2
    d_model=256, n_heads=8, n_layers=2,
    clip_model="openai/clip-vit-base-patch16",
    dino_model="facebook/dinov2-small",
    freeze_clip=True, freeze_dino=True,
    # GNM
    context_size=5, len_traj_pred=10, learn_angle=True,
    obs_encoding_size=1024, goal_encoding_size=1024,
    goal_type="image_mask_enc", obs_type="disabled",
    dims=8, use_mask_grad=False, goal_uses_context=False,
    # TopoPaths / masks
    mask_h=120, mask_w=160,
)

# ---------------------------------------------------------------------------
# CLIP image de-normalisation
# ---------------------------------------------------------------------------
_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _denorm_clip(tensor) -> np.ndarray:
    """[3, H, W] CLIP-normalised CPU tensor → [H, W, 3] uint8."""
    img = tensor.cpu().float().numpy().transpose(1, 2, 0)
    img = img * _CLIP_STD + _CLIP_MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _resize_masks(masks_np: np.ndarray, H: int, W: int) -> np.ndarray:
    """[K, Hm, Wm] bool → [K, H, W] bool (nearest-neighbour via PIL)."""
    K = masks_np.shape[0]
    if K == 0 or (masks_np.shape[1] == H and masks_np.shape[2] == W):
        return masks_np
    return np.stack([
        np.array(
            Image.fromarray(masks_np[k].astype(np.uint8))
                 .resize((W, H), Image.NEAREST)
        ).astype(bool)
        for k in range(K)
    ])


# ---------------------------------------------------------------------------
# Lazy singleton: models + val dataset
# ---------------------------------------------------------------------------
_state: dict = {}


def _get_state(checkpoint_path: str, h5_path: str, device_str: str):
    """Load models and val dataset once; reuse afterwards."""
    key = (checkpoint_path, h5_path, device_str)
    if _state.get("key") == key:
        return _state

    print(f"[joint_val_app] loading checkpoint {checkpoint_path} …")
    device = torch.device(device_str)

    from vint_train.models.gnm.gnm import GNM
    from vint_train.models.object_react.dataloader import TopoPaths
    from lange3dnet_train.model import LangGeoNetV2
    from lange3dnet_train.joint_dataset import create_joint_dataloaders

    # ---- Models -----------------------------------------------------------
    lange3d = LangGeoNetV2(
        d_model=MODEL_CFG["d_model"],
        n_heads=MODEL_CFG["n_heads"],
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
    print(f"[joint_val_app] models loaded (epoch {ckpt.get('epoch', '?')})")

    topopaths = TopoPaths(
        dims=MODEL_CFG["dims"],
        w=MODEL_CFG["mask_w"],
        h=MODEL_CFG["mask_h"],
    )

    # ---- Validation dataset -----------------------------------------------
    print(f"[joint_val_app] indexing H5: {h5_path} …")
    _, val_loader = create_joint_dataloaders(
        h5_path=h5_path,
        batch_size=1,           # one sample at a time for the app
        num_workers=0,          # avoid worker issues in Gradio
        val_split=DEFAULT_VAL_SPLIT,
        seed=DEFAULT_SEED,
        len_traj_pred=MODEL_CFG["len_traj_pred"],
        gnm_mask_h=MODEL_CFG["mask_h"] // 2,
        gnm_mask_w=MODEL_CFG["mask_w"] // 2,
    )
    # Materialise the full val list so we can index it directly.
    val_samples = list(val_loader.dataset)
    print(f"[joint_val_app] {len(val_samples)} validation samples ready")

    _state.update(dict(
        key=key,
        device=device,
        lange3d=lange3d,
        gnm=gnm,
        topopaths=topopaths,
        val_samples=val_samples,
    ))
    return _state


# ---------------------------------------------------------------------------
# Visualisation helpers (no external dep on cost_overlay.py so the app is
# self-contained — mirrors the same design)
# ---------------------------------------------------------------------------

_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grn_red", ["#00cc44", "#ffff00", "#ff2200"]
)


def _cost_overlay(rgb_hwc: np.ndarray,
                  masks_np: np.ndarray,
                  costs: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """[H,W,3] uint8 + [K,H,W] bool + [K] costs → [H,W,3] float32 [0,1]."""
    H, W = rgb_hwc.shape[:2]
    img  = rgb_hwc.astype(np.float32) / 255.0
    canvas   = np.full((H, W), np.nan, dtype=np.float32)
    seg_mask = np.zeros((H, W), dtype=bool)

    for k in range(masks_np.shape[0]):
        m = masks_np[k]
        if not m.any():
            continue
        c = float(costs[k])
        canvas[m] = np.clip(c, 0.0, 1.0) if np.isfinite(c) else 0.5
        seg_mask |= m

    if seg_mask.any():
        filled  = np.where(np.isnan(canvas), 0.5, canvas)
        heat    = _CMAP(filled)[:, :, :3]
        blended = (1 - alpha) * img + alpha * heat
        img     = np.where(seg_mask[:, :, None], blended, img)

    # white contours
    for k in range(masks_np.shape[0]):
        m = masks_np[k]
        if not m.any():
            continue
        pad  = np.pad(m.astype(np.uint8), 1, mode="constant")
        nb   = pad[:-2,1:-1] + pad[2:,1:-1] + pad[1:-1,:-2] + pad[1:-1,2:]
        cont = m & (nb < 4)
        img  = np.where(cont[:, :, None], 1.0, img)

    return np.clip(img, 0.0, 1.0)


def _annotate_costs(ax, masks_np: np.ndarray, costs: np.ndarray):
    for k in range(masks_np.shape[0]):
        m = masks_np[k]
        if not m.any():
            continue
        ys, xs = np.where(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        lbl = f"{float(costs[k]):.2f}" if np.isfinite(costs[k]) else "∞"
        ax.text(cx, cy, lbl, fontsize=6, color="white",
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4, lw=0))


def _render_canvas(rgb: np.ndarray,
                   masks_np: np.ndarray,
                   gt_costs: np.ndarray,
                   pred_costs: np.ndarray,
                   gt_wp: np.ndarray,
                   pred_wp: np.ndarray,
                   nai_text: str,
                   ep_id: str,
                   frame_key: str) -> np.ndarray:
    """Return [H, W, 3] uint8 canvas with 3 panels + instruction row."""
    gt_ov   = _cost_overlay(rgb, masks_np, gt_costs)
    pred_ov = _cost_overlay(rgb, masks_np, pred_costs)

    fig = plt.figure(figsize=(18, 7))
    gs  = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.18,
                           height_ratios=[1, 0.12])

    # ---- Panel 1: GT cost -----------------------------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(gt_ov)
    _annotate_costs(ax0, masks_np, gt_costs)
    ax0.set_title(f"GT cost  [{ep_id} / {frame_key}]",
                  fontsize=10, fontweight="bold")
    ax0.axis("off")

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax0, fraction=0.025, pad=0.02,
                 label="cost  (0=low/green  1=high/red)")

    # ---- Panel 2: Predicted cost -----------------------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(pred_ov)
    _annotate_costs(ax1, masks_np, pred_costs)
    ax1.set_title("Predicted cost", fontsize=10, fontweight="bold")
    ax1.axis("off")

    # ---- Panel 3: Trajectory ---------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    # _to_local_coords_2d returns [right, backward] (Habitat Y-up convention:
    # col-0 = +X = right, col-1 = +Z = backward).
    # Convert to display frame: forward = -col-1, right = +col-0.
    gt_fwd,   gt_right   = -gt_wp[:, 1],    gt_wp[:, 0]
    pred_fwd, pred_right = -pred_wp[:, 1],  pred_wp[:, 0]

    ax2.plot([0, *gt_right],   [0, *gt_fwd],   "o-",  color="#00cc44",
             lw=1.8, ms=5, label="GT traj")
    ax2.plot([0, *pred_right], [0, *pred_fwd], "s--", color="#ff2200",
             lw=1.8, ms=5, label="Pred traj")
    ax2.plot(0, 0, "k^", ms=10, zorder=5, label="Robot")

    all_fwd   = np.concatenate([[0], gt_fwd,   pred_fwd])
    all_right = np.concatenate([[0], gt_right, pred_right])
    pad = max(0.5, float(np.abs(np.concatenate([all_fwd, all_right])).max()) * 0.15)
    ax2.set_xlim(all_right.min() - pad, all_right.max() + pad)
    ax2.set_ylim(all_fwd.min()   - pad, all_fwd.max()   + pad)
    ax2.set_xlabel("right  (+) / left  (−)", fontsize=9)
    ax2.set_ylabel("forward  (+)", fontsize=9)
    ax2.set_title("Trajectory  (local robot frame)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_aspect("equal")
    ax2.grid(True, ls=":", alpha=0.4)

    # ---- Instruction text row -------------------------------------------
    ax_txt = fig.add_subplot(gs[1, :])
    ax_txt.axis("off")
    import textwrap
    wrapped = textwrap.fill(nai_text, width=110)
    ax_txt.text(0.5, 0.5, wrapped,
                ha="center", va="center",
                fontsize=16, color="black",
                transform=ax_txt.transAxes)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_sample(
    sample_idx: int,
    custom_instruction: str,
    checkpoint_path: str,
    h5_path: str,
    device_str: str,
):
    """Run inference on one validation sample and return (canvas, status)."""
    try:
        st = _get_state(checkpoint_path, h5_path, device_str)
    except Exception as exc:
        return None, f"Load error: {exc}"

    val_samples = st["val_samples"]
    n = len(val_samples)
    if n == 0:
        return None, "No validation samples found."

    idx = int(sample_idx) % n
    sample = val_samples[idx]

    device     = st["device"]
    lange3d    = st["lange3d"]
    gnm        = st["gnm"]
    topopaths  = st["topopaths"]

    # Decide instruction
    nai_text = (custom_instruction.strip()
                if custom_instruction and custom_instruction.strip()
                else sample["nai_text"])

    # ---- Tokenize custom instruction if changed --------------------------
    if custom_instruction and custom_instruction.strip():
        from transformers import CLIPProcessor
        proc = CLIPProcessor.from_pretrained(MODEL_CFG["clip_model"])
        tok  = proc(text=nai_text, padding="max_length", truncation=True,
                    max_length=77, return_tensors="pt")
        nai_input_ids = tok["input_ids"].squeeze(0)
        nai_attn_mask = tok["attention_mask"].squeeze(0)
    else:
        nai_input_ids = sample["nai_input_ids"]
        nai_attn_mask = sample["nai_attn_mask"]

    # ---- Prepare tensors (add batch dim) --------------------------------
    pixel_values   = sample["pixel_values"].unsqueeze(0).to(device)       # [1,3,224,224]
    nai_input_ids  = nai_input_ids.unsqueeze(0).to(device)                # [1,77]
    nai_attn_mask  = nai_attn_mask.unsqueeze(0).to(device)                # [1,77]
    masks          = sample["masks"]                                       # [K,H,W] bool CPU
    gnm_masks      = sample["gnm_masks"].unsqueeze(0).to(device)          # [1,K,Hh,Wh]
    K_list         = [masks.shape[0]]
    gt_costs_t     = sample["gt_costs"]                                   # [K] CPU
    action_label   = sample["action_label"].cpu().numpy()                 # [T,4]
    ep_id          = sample.get("ep_id", "?")
    frame_key      = sample.get("frame_key", "?")

    # ---- Forward pass ----------------------------------------------------
    with torch.no_grad():
        lang_preds, _ = lange3d(
            pixel_values,
            [masks.to(device)],
            nai_input_ids,
            nai_attn_mask,
        )
        goal_enc = topopaths.build_differentiable_goal(
            lang_preds, gnm_masks, K_list, device
        )
        _, goal_img = goal_enc.split([3, goal_enc.shape[1] - 3], dim=1)
        obs_img = torch.zeros(1, 3, 120, 160, device=device)
        _, action_pred = gnm(obs_img, goal_img)

    # ---- Post-process predictions ----------------------------------------
    raw_pred = lang_preds[0].cpu().float().numpy()        # [K]
    # min-max normalise to [0,1] for display (same as val loop)
    pred_costs = (raw_pred - raw_pred.min()) / (raw_pred.max() - raw_pred.min() + 1e-8)

    gt_costs   = gt_costs_t.cpu().float().numpy()         # [K] already [0,1]
    pred_wp    = action_pred[0].cpu().float().numpy()     # [T,4]

    # ---- Denorm RGB + resize masks to image size ------------------------
    rgb      = _denorm_clip(sample["pixel_values"])       # [224,224,3] uint8
    H, W     = rgb.shape[:2]
    masks_np = _resize_masks(masks.cpu().numpy().astype(bool), H, W)  # [K,H,W]

    # ---- Build canvas ----------------------------------------------------
    canvas = _render_canvas(
        rgb, masks_np,
        gt_costs, pred_costs,
        action_label, pred_wp,
        nai_text,
        ep_id=str(ep_id), frame_key=str(frame_key),
    )

    status = (
        f"Sample {idx+1}/{n}  |  ep={ep_id}  frame={frame_key}  "
        f"K={len(gt_costs)} objects  |  instruction: {nai_text[:80]}"
    )
    return canvas, status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app(
    checkpoint_path: str,
    h5_path: str,
    device_str: str,
):
    import gradio as gr

    # Pre-load so first press is instant
    try:
        st = _get_state(checkpoint_path, h5_path, device_str)
        n_val = len(st["val_samples"])
    except Exception as exc:
        print(f"[warn] pre-load failed: {exc}")
        n_val = 9999

    with gr.Blocks(title="Joint Val Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Joint Val Explorer — LangGeoNetV2 + GNM
            Browse the **validation set** frame by frame.
            Each panel shows the GT cost map, predicted cost map, and GT vs predicted waypoints.
            The instruction defaults to the dataset's `next_action_instruction`; override it below.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                sample_slider = gr.Slider(
                    minimum=0, maximum=max(n_val - 1, 0),
                    step=1, value=0,
                    label=f"Sample index  (0 – {n_val-1})",
                )
                instruction_box = gr.Textbox(
                    label="Override instruction  (leave blank to use dataset default)",
                    placeholder="e.g. 'Go to the chair near the window'",
                    lines=2,
                )
                run_btn  = gr.Button("Run Inference", variant="primary")
                prev_btn = gr.Button("◀  Prev")
                next_btn = gr.Button("Next  ▶")
                status   = gr.Textbox(label="Status", interactive=False, lines=2)

                ckpt_box   = gr.Textbox(label="Checkpoint path",
                                        value=checkpoint_path, lines=1)
                h5_box     = gr.Textbox(label="H5 dataset path",
                                        value=h5_path, lines=1)
                device_box = gr.Textbox(label="Device",
                                        value=device_str, lines=1)
                reload_btn = gr.Button("Reload models / dataset")

            with gr.Column(scale=3):
                canvas_out = gr.Image(
                    label="GT cost  |  Predicted cost  |  Trajectory",
                    type="numpy",
                )

        # ---- Callbacks ---------------------------------------------------
        def _infer(idx, instr, ckpt, h5, dev):
            return run_sample(int(idx), instr, ckpt, h5, dev)

        def _prev(idx, instr, ckpt, h5, dev):
            new_idx = max(0, int(idx) - 1)
            img, st = run_sample(new_idx, instr, ckpt, h5, dev)
            return new_idx, img, st

        def _next(idx, instr, ckpt, h5, dev):
            new_idx = int(idx) + 1   # slider will clamp to max
            img, st = run_sample(new_idx, instr, ckpt, h5, dev)
            return new_idx, img, st

        def _reload(ckpt, h5, dev):
            _state.clear()
            try:
                st = _get_state(ckpt, h5, dev)
                return f"Reloaded. {len(st['val_samples'])} val samples."
            except Exception as exc:
                return f"Reload failed: {exc}"

        run_btn.click(
            fn=_infer,
            inputs=[sample_slider, instruction_box, ckpt_box, h5_box, device_box],
            outputs=[canvas_out, status],
            api_name=False,
        )
        prev_btn.click(
            fn=_prev,
            inputs=[sample_slider, instruction_box, ckpt_box, h5_box, device_box],
            outputs=[sample_slider, canvas_out, status],
            api_name=False,
        )
        next_btn.click(
            fn=_next,
            inputs=[sample_slider, instruction_box, ckpt_box, h5_box, device_box],
            outputs=[sample_slider, canvas_out, status],
            api_name=False,
        )
        reload_btn.click(
            fn=_reload,
            inputs=[ckpt_box, h5_box, device_box],
            outputs=[status],
            api_name=False,
        )

    demo.launch(share=True, show_api=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint validation Gradio app")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to best_joint.pth")
    parser.add_argument("--h5", default=DEFAULT_H5,
                        help="Path to the training H5 file (val split used)")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto-detect)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    build_app(
        checkpoint_path=args.checkpoint,
        h5_path=args.h5,
        device_str=args.device,
    )

#----------------------original code --------------------------------------------
"""
Training loop for LangGeoNet — H5EpisodePathLengthsDataset edition.
"""
from __future__ import annotations
import os
import time
import json
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

from model import LangGeoNetV2
from dataset import create_h5_episode_pathlengths_dataloader, create_lmdb_episode_pathlengths_dataloader
from losses import LangGeoNetLoss, CounterfactualContrastiveLoss, ObjectGroundingContrastiveLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------------------------------
# Gradient-norm helpers
# -------------------------------------------------------

_LANG_PREFIXES   = ("clip.text_model", "lang_proj")
_VISION_PREFIXES = ("clip.vision_model", "dino", "clip_vis_proj", "dino_vis_proj")


def _compute_branch_grad_norms(model: nn.Module) -> dict:
    """
    Return L2 gradient norms for the unfrozen language and vision branch
    parameters (those for which requires_grad=True and grad is not None).

    Returns a dict with keys:
        'grad_norm/lang'   – CLIP text + lang_proj
        'grad_norm/vision' – CLIP visual + DINOv2 + vis projections
        'grad_norm/head'   – all other trainable parameters
    """
    sq = {"lang": 0.0, "vision": 0.0, "head": 0.0}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        norm_sq = p.grad.detach().float().norm(2).item() ** 2
        if any(name.startswith(pfx) for pfx in _LANG_PREFIXES):
            sq["lang"] += norm_sq
        elif any(name.startswith(pfx) for pfx in _VISION_PREFIXES):
            sq["vision"] += norm_sq
        else:
            sq["head"] += norm_sq
    return {f"grad_norm/{k}": sq[k] ** 0.5 for k in sq}


# -------------------------------------------------------
# Batch adaptor  —  bridges dataloader → model interface
# -------------------------------------------------------

def _prepare_batch(batch: dict, device: torch.device):
    """
    Move a pre-processed H5EpisodePathLengthsDataset batch to the target device.
    Masks and GT costs are computed in __getitem__ (worker processes).
    """
    pixel_values = batch["pixel_values"].to(device)
    input_ids    = batch["nai_input_ids"].to(device) #batch["input_ids"].to(device)
    attn_mask    = batch["nai_attention_mask"].to(device)
    masks_list   = [m.to(device) for m in batch["masks_list"]]
    gts_list     = [g.to(device) for g in batch["gt_costs_list"]]
    return pixel_values, input_ids, attn_mask, masks_list, gts_list


# -------------------------------------------------------
# Metrics / checkpointing helpers  (unchanged)
# -------------------------------------------------------

def compute_metrics(all_preds, all_gts):
    flat_p = np.concatenate(all_preds)
    flat_g = np.concatenate(all_gts)

    mae  = float(np.mean(np.abs(flat_p - flat_g)))
    mse  = float(np.mean((flat_p - flat_g) ** 2))
    rmse = float(np.sqrt(mse))

    valid = flat_g > 0.01
    mape  = float(np.mean(np.abs(flat_p[valid] - flat_g[valid]) / flat_g[valid])) if valid.sum() else 0.0

    rank_accs, spearman_corrs = [], []
    for p, g in zip(all_preds, all_gts):
        if len(p) < 2:
            continue
        correct = total = 0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if abs(g[i] - g[j]) < 1e-6:
                    continue
                total += 1
                correct += int((p[i] < p[j]) == (g[i] < g[j]))
        if total:
            rank_accs.append(correct / total)
        if len(p) >= 3:
            c, _ = spearmanr(p, g)
            if not np.isnan(c):
                spearman_corrs.append(c)

    thresh = {
        f"acc@{d}": float(np.mean(np.abs(flat_p - flat_g) < d))
        for d in [0.05, 0.10, 0.20]
    }
    return {
        "mae": mae, "mse": mse, "rmse": rmse, "mape": mape,
        "ranking_accuracy": float(np.mean(rank_accs))   if rank_accs   else 0.0,
        "spearman":         float(np.mean(spearman_corrs)) if spearman_corrs else 0.0,
        **thresh,
    }


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


def _resume_from_checkpoint(resume, model, optimizer, scheduler, device):
    if not resume:
        return 0, float("inf")
    if not os.path.isfile(resume):
        raise FileNotFoundError(f"Checkpoint not found: {resume}")
    ckpt = torch.load(resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start = int(ckpt.get("epoch", 0))
    best  = float(ckpt.get("best_val_mae", float("inf")))
    logger.info(f"Resumed at epoch {start + 1}, best MAE={best:.4f}")
    return start, best


def make_exp_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    n = 1
    while True:
        candidate = os.path.join(base_dir, f"exp{n}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        n += 1


# -------------------------------------------------------
# Train / validate
# -------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    grad_accum=1, wandb_run=None, global_step=0,
                    cf_criterion=None, lambda_cf=0.5,
                    ogcl_criterion=None, lambda_ogcl=1.0):
    model.train()
    losses      = defaultdict(float)
    n           = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        pixel_values, input_ids, attn_mask, masks_list, gts_list = \
            _prepare_batch(batch, device)
        # skip frames where every registry was empty
        if all(m.shape[0] == 0 for m in masks_list):
            continue

        # extract counterfactual fields (only populated when dataset.counterfactual=True)
        has_cf  = batch.get("has_counterfactual", [False] * len(masks_list))
        cf_ids  = batch.get("cf_nai_input_ids")
        cf_attn = batch.get("cf_nai_attention_mask")
        if cf_ids is not None:
            cf_ids  = cf_ids.to(device)
            cf_attn = cf_attn.to(device)

        preds, geo_preds, _ = model(pixel_values, masks_list, input_ids, attn_mask,
                                     return_geo=True)
        loss, ld = criterion(preds, gts_list, geo_preds=geo_preds)

        # counterfactual contrastive loss: preds must diverge when noun is swapped
        ld["loss_cf"] = 0.0
        if cf_criterion is not None and any(has_cf) and cf_ids is not None:
            with torch.no_grad():
                cf_preds, _, _ = model(pixel_values, masks_list, cf_ids, cf_attn,
                                       return_geo=True)
            l_cf         = cf_criterion(preds, cf_preds, has_cf)
            loss         = loss + lambda_cf * l_cf
            ld["loss_cf"] = l_cf.item()

        # object grounding contrastive loss (directional: referenced class ranks lower-cost)
        ld['loss_ogcl'] = 0.0
        if ogcl_criterion is not None:
            class_match_list = [m.to(device) for m in batch.get('class_match_list', [])]
            if class_match_list:
                l_ogcl          = ogcl_criterion(preds, class_match_list)
                loss            = loss + lambda_ogcl * l_ogcl
                ld['loss_ogcl'] = l_ogcl.item()

        ld["loss_total"] = loss.item()
        (loss / grad_accum).backward()

        if (i + 1) % grad_accum == 0:
            grad_norms = _compute_branch_grad_norms(model)
            # Clip backbone (lang+vision) and head separately so the larger head
            # gradients don't force the backbone clip to fire every step.
            backbone_params = [p for n, p in model.named_parameters()
                               if p.requires_grad and p.grad is not None
                               and (any(n.startswith(pfx) for pfx in _LANG_PREFIXES)
                                    or any(n.startswith(pfx) for pfx in _VISION_PREFIXES))]
            head_params_grad = [p for n, p in model.named_parameters()
                                if p.requires_grad and p.grad is not None
                                and not (any(n.startswith(pfx) for pfx in _LANG_PREFIXES)
                                         or any(n.startswith(pfx) for pfx in _VISION_PREFIXES))]
            nn.utils.clip_grad_norm_(backbone_params,  5.0)
            nn.utils.clip_grad_norm_(head_params_grad, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            logger.debug(
                f"  grad_norms — lang={grad_norms['grad_norm/lang']:.4f}  "
                f"vision={grad_norms['grad_norm/vision']:.4f}  "
                f"head={grad_norms['grad_norm/head']:.4f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in ld.items()}
                    | {f"train/{k}": v for k, v in grad_norms.items()}
                    | {"batch_step": global_step}
                )
            global_step += 1
            ld.update(grad_norms)

        for k, v in ld.items():
            losses[k] += v
        n += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"  batch {i+1}/{len(loader)}  loss={losses['loss_total']/n:.4f}  "
                f"reg={losses.get('loss_regression', 0.0)/n:.4f}  "
                f"rank={losses.get('loss_ranking', 0.0)/n:.4f}  "
                f"div={losses.get('loss_diversity', 0.0)/n:.4f}  "
                f"ogcl={losses.get('loss_ogcl', 0.0)/n:.4f}  "
                f"grad_norm[lang]={losses.get('grad_norm/lang', 0.0)/n:.4f}  "
                f"grad_norm[vision]={losses.get('grad_norm/vision', 0.0)/n:.4f}  "
                f"grad_norm[head]={losses.get('grad_norm/head', 0.0)/n:.4f}"
            )
        # break

    return {k: v / max(n, 1) for k, v in losses.items()}, global_step


@torch.no_grad()
def validate(model, loader, criterion, device,
             wandb_run=None, exp_dir=None, epoch=None, max_viz=8):
    model.eval()
    losses               = defaultdict(float)
    n                    = 0
    all_preds, all_gts   = [], []
    viz_count=0
    viz_images= []
    first_batch_vizd = False
    for batch in loader:
        pixel_values, input_ids, attn_mask, masks_list, gts_list = \
            _prepare_batch(batch, device)

        if all(m.shape[0] == 0 for m in masks_list):
            del batch
            continue

        preds, geo_preds, _ = model(pixel_values, masks_list, input_ids, attn_mask,
                                     return_geo=True)
        _, ld    = criterion(preds, gts_list, geo_preds=geo_preds)

        for k, v in ld.items():
            losses[k] += v
        n += 1

        # accumulate metrics — apply sigmoid to convert raw logits to [0, 1]
        for p, g in zip(preds, gts_list):
            all_preds.append(torch.sigmoid(p).cpu().numpy())
            all_gts.append(g.cpu().numpy())

        # collect visualization samples (small number)
        if not first_batch_vizd:
            for i, (p, g) in enumerate(zip(preds, gts_list)):
                if viz_count >= max_viz:
                    break
                try:
                    frame_rgb, masks_arr, node_ids = _extract_sample_masks_and_rgb(batch, i)
                    pred_np = torch.sigmoid(p).cpu().numpy()  # logits -> [0,1] for viz
                    gt_np   = g.cpu().numpy()
                    K = min(len(pred_np), len(gt_np), masks_arr.shape[0])
                    if K == 0:
                        continue
                    img = _render_cost_comparison_image(frame_rgb, masks_arr[:K], gt_np[:K], pred_np[:K])
                    viz_images.append(img)
                    viz_count += 1
                except Exception as e:
                    print(f"Viz sample {i} failed: {e}")
            first_batch_vizd = True
        # break
    avg_losses = {k: v / max(n, 1) for k, v in losses.items()}
    metrics    = compute_metrics(all_preds, all_gts)
    metrics.update(avg_losses)

    if wandb_run is not None and viz_images:
        try:
            wb_imgs = [wandb.Image(img, caption=f"val_ep{epoch}_s{idx}") for idx, img in enumerate(viz_images)]
            wandb_run.log({"viz/pred_vs_gt": wb_imgs, "epoch": epoch})
        except Exception:
            pass

    return metrics


# -------------------------------------------------------
# Main entry-point
# -------------------------------------------------------

def train_h5(
    h5_path: str,
    lmdb_path: str | None = None,
    val_h5_path: str | None = None,
    output_dir: str = "./checkpoints",
    run_dir: str | None = None,
    use_wandb: bool = True,
    wandb_project: str = "langgeonet",
    wandb_run_name: str | None = None,
    clip_model: str = "openai/clip-vit-base-patch16",
    bert_model: str = "bert-base-uncased",
    d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
    num_classes: int = 1550,
    epochs: int = 50, batch_size: int = 8,
    lr_head: float = 1e-4, lr_backbone: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3, grad_accum: int = 1,
    lambda_rank: float = 0.5, lambda_si: float = 0.3, lambda_cf: float = 0.5,
    lambda_ogcl: float = 1.0,
    num_workers: int = 4, seed: int = 42, patience: int = 10,
    device=None,
    resume: str | None = None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    exp_dir  = make_exp_dir(run_dir) if run_dir else output_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    analysis_dir = os.path.join(exp_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    logger.info("Building model...")
    model   = LangGeoNetV2(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                           clip_model_name=clip_model).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {total_p:,} total  {train_p:,} trainable")

    logger.info("Loading data...")
    # train_loader, val_loader = create_lmdb_episode_pathlengths_dataloader(
    #     h5_path=h5_path, batch_size=batch_size,
    #     num_workers=num_workers,
    # )
    train_loader, val_loader = create_lmdb_episode_pathlengths_dataloader(
        lmdb_path=lmdb_path, batch_size=batch_size,
        num_workers=num_workers, counterfactual=True,
    )
    # val_loader = create_h5_episode_pathlengths_dataloader(
    #     h5_path=val_h5_path or h5_path,
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers,
    # )
    if not val_h5_path:
        logger.warning("No val H5 provided — reusing train set for validation.")

    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name or os.path.basename(exp_dir),
            config=dict(
                h5_path=h5_path, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                epochs=epochs, batch_size=batch_size, lr_head=lr_head,
                lr_backbone=lr_backbone, lambda_rank=lambda_rank, lambda_si=lambda_si,
            ),
            dir=exp_dir,
        )
        wandb_run.define_metric("batch_step")
        wandb_run.define_metric("train/*", step_metric="batch_step")
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("epoch/*", step_metric="epoch")
        wandb_run.define_metric("val/*",   step_metric="epoch")
        logger.info(f"W&B: {wandb_run.name} | {wandb_run.url}")
    elif use_wandb:
        logger.warning("wandb not installed — skipping.")

    backbone_names = {"clip", "bert"}
    head_params, bb_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if any(bn in name for bn in backbone_names) else head_params).append(p)

    optimizer = optim.AdamW([
        {"params": head_params, "lr": lr_head},
        {"params": bb_params,   "lr": lr_backbone},
    ], weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (
        (ep + 1) / warmup_epochs if ep < warmup_epochs
        else 0.5 * (1 + np.cos(np.pi * (ep - warmup_epochs) / max(1, epochs - warmup_epochs)))
    ))

    criterion    = LangGeoNetLoss(lambda_rank, lambda_si)
    cf_criterion = CounterfactualContrastiveLoss()
    ogcl_criterion = ObjectGroundingContrastiveLoss()
    start_epoch, best_mae = _resume_from_checkpoint(resume, model, optimizer, scheduler, device)
    wait        = 0
    history     = []
    global_step = start_epoch * len(train_loader)

    logger.info(f"Training: {epochs} epochs | bs={batch_size} | accum={grad_accum} | device={device}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        logger.info(f"\n{'='*60}\nEpoch {epoch+1}/{epochs}\n{'='*60}")

        train_loss, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_accum, wandb_run, global_step,
            cf_criterion=cf_criterion, lambda_cf=lambda_cf,
            ogcl_criterion=ogcl_criterion, lambda_ogcl=lambda_ogcl,
        )
        val_m = validate(model, val_loader, criterion, device,
                 wandb_run=wandb_run, exp_dir=analysis_dir, epoch=epoch + 1, max_viz=8)
        scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"  Train | total={train_loss['loss_total']:.4f}  "
            f"reg={train_loss['loss_regression']:.4f}  "
            f"rank={train_loss['loss_ranking']:.4f}  "
            f"si={train_loss['loss_scale_invariant']:.4f}"
        )
        logger.info(
            f"  Val   | MAE={val_m['mae']:.4f}  RMSE={val_m['rmse']:.4f}  "
            f"MAPE={val_m['mape']:.4f}  "
            f"RankAcc={val_m['ranking_accuracy']:.4f}  Spearman={val_m['spearman']:.4f}"
        )
        logger.info(
            f"  Val   | acc@0.05={val_m['acc@0.05']:.4f}  "
            f"acc@0.10={val_m['acc@0.1']:.4f}  "
            f"acc@0.20={val_m['acc@0.2']:.4f}"
        )
        logger.info(f"  LR={lr:.2e} | {dt:.1f}s")

        if wandb_run is not None:
            epoch_log = {f"epoch/train_{k}": v for k, v in train_loss.items()}
            epoch_log.update({f"epoch/val_{k}": v for k, v in val_m.items()})
            epoch_log["epoch/lr"]       = lr
            epoch_log["epoch/duration"] = dt
            epoch_log["epoch"]          = epoch + 1
            wandb_run.log(epoch_log)

        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_m, "lr": lr})

        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            wait     = 0
            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_mae":         best_mae,
                "val_metrics":          val_m,
                "config": dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                               num_classes=num_classes, clip_model=clip_model),
            }, os.path.join(ckpt_dir, "best_model.pt"))
            logger.info(f"  ★ Best model saved (MAE={best_mae:.4f})")
            if wandb_run is not None:
                wandb_run.summary["best_val_mae"]   = best_mae
                wandb_run.summary["best_val_epoch"] = epoch + 1
        else:
            wait += 1
            logger.info(f"  No improvement ({wait}/{patience})")

        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae":         best_mae,
        }, os.path.join(ckpt_dir, "latest_model.pt"))

        if wait >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    if wandb_run is not None:
        wandb_run.finish()

    logger.info(f"Done. Best val MAE: {best_mae:.4f} | {exp_dir}")
    return model, history


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--h5_path",       default=None)
    p.add_argument("--lmdb_path", default=None)
    p.add_argument("--val_h5_path",   default=None)
    p.add_argument("--run_dir",       default=None)
    p.add_argument("--output_dir",    default="./checkpoints")
    p.add_argument("--wandb_project", default="langgeonet")
    p.add_argument("--wandb_run_name",default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=24)
    p.add_argument("--lr_head",       type=float, default=3e-4)
    p.add_argument("--lr_backbone",   type=float, default=1e-5)
    p.add_argument("--d_model",       type=int,   default=256)
    p.add_argument("--n_layers",      type=int,   default=1)
    p.add_argument("--grad_accum",    type=int,   default=1)
    p.add_argument("--lambda_rank",   type=float, default=0.5)
    p.add_argument("--lambda_si",     type=float, default=0.0)
    p.add_argument("--lambda_cf",     type=float, default=0.0)
    p.add_argument("--lambda_ogcl",   type=float, default=1.0)
    p.add_argument("--patience",      type=int,   default=100)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--device",        default=None)
    p.add_argument("--resume",        default=None)
    a = p.parse_args()

    train_h5(
        h5_path=a.h5_path,
        lmdb_path= a.lmdb_path,
        val_h5_path=a.val_h5_path,
        run_dir=a.run_dir,
        output_dir=a.output_dir,
        use_wandb=not a.no_wandb,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr_head=a.lr_head,
        lr_backbone=a.lr_backbone,
        d_model=a.d_model,
        n_layers=a.n_layers,
        grad_accum=a.grad_accum,
        lambda_rank=a.lambda_rank,
        lambda_si=a.lambda_si,
        lambda_cf=a.lambda_cf,
        lambda_ogcl=a.lambda_ogcl,
        patience=a.patience,
        num_workers=a.num_workers,
        device=a.device,
        resume=a.resume,
    )
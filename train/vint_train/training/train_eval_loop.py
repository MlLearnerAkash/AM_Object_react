import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

try:
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# Backbone parameter name prefixes (same convention as lange3dnet_train/train.py)
_BACKBONE_PREFIXES = ("clip", "dino", "bert")

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    **kwargs,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, epochs):
        if train_model:
            print(f"Start ViNT Training Epoch {epoch}/{epochs}")
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                **kwargs,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(f"Start {dataset_type} ViNT Testing Epoch {epoch}/{epochs}")
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss, traj_metrics = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                **kwargs,
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler,
            "wandb_run_id": wandb.run.id if use_wandb else None,
            "wandb_run_dir": wandb.run.dir if use_wandb else None,
        }
        # log average eval loss
        if use_wandb:
            wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        if use_wandb:
            wandb.log(
                {
                    "avg_total_test_loss": np.mean(avg_total_test_loss),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                commit=False,
            )

        try:
            numbered_path = os.path.join(project_folder, f"{epoch}.pth")
            torch.save(checkpoint, latest_path)
            # torch.save(checkpoint, numbered_path)  # keep track of model at every epoch
            _lange3d = kwargs.get("lange3d_model", None)
            if _lange3d is not None:
                joint_latest_path = os.path.join(project_folder, "joint_latest.pth")
                # torch.save({
                #     "epoch": epoch,
                #     "gnm": model.state_dict(),
                #     "lange3d": _lange3d.state_dict(),
                #     "optimizer": optimizer.state_dict(),
                #     "avg_total_test_loss": np.mean(avg_total_test_loss),
                # }, joint_latest_path)
                checkpoint = {
                    "epoch": epoch,
                    "gnm": model.state_dict(),
                    "lange3d": _lange3d.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "avg_total_test_loss": np.mean(avg_total_test_loss),
                }
                if epoch % 50 == 0:
                    joint_epoch_path = os.path.join(project_folder, f"joint_{epoch}.pth")
                    torch.save(checkpoint, joint_epoch_path)
                
                torch.save(checkpoint, joint_latest_path)

        except Exception as e:
            print("Error saving model", e)

    # Flush the last set of eval logs
    if use_wandb:
        wandb.log({})
    print()


def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model, power=0.75)

    for epoch in range(current_epoch, epochs):
        if train_model:
            print(f"Start ViNT DP Training Epoch {epoch}/{epochs}")
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                print(f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{epochs}")
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        wandb.log(
            {
                "lr": optimizer.param_groups[0]["lr"],
            },
            commit=False,
        )

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log(
            {
                "lr": optimizer.param_groups[0]["lr"],
            },
            commit=False,
        )

    # Flush the last set of eval logs
    wandb.log({})
    print()


def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params


# ---------------------------------------------------------------------------
# Joint LangGeoNetV2 + GNM training loop
# ---------------------------------------------------------------------------

def train_eval_loop_joint(
    lange3d_model: nn.Module,
    gnm_model: nn.Module,
    lange3d_optimizer: torch.optim.Optimizer,
    gnm_optimizer: torch.optim.Optimizer,
    lange3d_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    gnm_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_loader: DataLoader,
    val_loader: DataLoader,
    topopaths,                              # TopoPaths instance
    lange3d_loss_fn: nn.Module,            # LangGeoNetLoss
    epochs: int,
    device: torch.device,
    lambda_action: float = 0.01,
    alpha_dist: float = 0.5,
    log_freq: int = 100,
    save_dir: str = "checkpoints",
    project_name: str = "joint_training",
    ogcl_criterion: nn.Module = None,      # ObjectGroundingContrastiveLoss
    lambda_ogcl: float = 1.0,
    lambda_aux: float = 0.001,             # cross-module skip-connection weight
):
    """
    End-to-end joint training of LangGeoNetV2 and GNM.

    LangGeoNetV2 predicts per-object costs from language + RGB;
    those costs are differentiably composed into a goal encoding that
    drives GNM action prediction (replaces pre-computed H5 costmaps).

    Loss:
        total = l_lang + lambda_ogcl * l_ogcl + lambda_action * l_gnm

    ogcl_criterion: ObjectGroundingContrastiveLoss — enforces that objects of
        the referenced category receive lower predicted costs than unmatched
        objects (directional margin ranking on raw logits).
    """
    from vint_train.training.train_utils import compute_success_rate

    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        lange3d_model.train()
        gnm_model.train()

        running_loss_total = 0.0
        running_loss_lang  = 0.0
        running_loss_gnm   = 0.0
        running_loss_ogcl  = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)     # [B, 3, 224, 224]
            nai_input_ids = batch["nai_input_ids"].to(device)   # [B, 77]
            nai_attn_mask = batch["nai_attn_mask"].to(device)   # [B, 77]
            masks_list    = [m.to(device) for m in batch["masks_list"]]
            gt_costs_list = [c.to(device) for c in batch["gt_costs_list"]]
            class_match_list = batch["class_match_list"]
            gnm_masks     = batch["gnm_masks"].to(device)       # [B, K_max, Hh, Wh]
            K_list        = batch["K_list"]
            action_label  = batch["action_label"].to(device)    # [B, T, 4]
            wp_mask       = batch["wp_mask"].to(device)           # [B, T] — 1 while moving
            dist_label    = batch["dist_label"].to(device)      # [B]
            action_mask   = batch["action_mask"].to(device)     # [B]

            B = pixel_values.shape[0]

            # Skip batches where every sample has no segmented objects — mirrors
            # train.py behaviour; prevents NaNs and wasted cost-head compute.
            if all(m.shape[0] == 0 for m in masks_list):
                continue

            lange3d_optimizer.zero_grad()
            gnm_optimizer.zero_grad()

            # ---- LangGeoNetV2 forward ------------------------------------
            lang_preds, _ = lange3d_model(
                pixel_values, masks_list, nai_input_ids, nai_attn_mask
            )

            # ---- Losses --------------------------------------------------
            l_lang, _ = lange3d_loss_fn(lang_preds, gt_costs_list)

            l_ogcl = torch.tensor(0.0, device=device)
            if ogcl_criterion is not None:
                class_match_list = [m.to(device) for m in batch.get('class_match_list', [])]
                if class_match_list:
                    l_ogcl = ogcl_criterion(lang_preds, class_match_list)

            l_lange3d = l_lang + lambda_ogcl * l_ogcl

            # Keeps the gradient path: l_gnm → goal_enc → lang_preds → lange3d_model
            goal_enc = topopaths.build_differentiable_goal(
                lang_preds, gnm_masks, K_list, device
            )
            _, goal_img = goal_enc.split([3, goal_enc.shape[1] - 3], dim=1)

            obs_img = torch.zeros(B, 3, 120, 160, device=device)
            dist_pred, action_pred = gnm_model(obs_img, goal_img)

            l_dist = F.mse_loss(dist_pred, dist_label.float().unsqueeze(1))

            action_diff = (action_pred - action_label) ** 2
            # Use wp_mask to exclude clamped (stop-in-place) waypoints from loss.
            eff_mask = action_mask.unsqueeze(1).expand(-1, action_pred.shape[1])  # [B, T] #wp_mask * action_mask.unsqueeze(1)  # [B, T]
            l_action = (action_diff.mean(-1) * eff_mask).sum()
            if eff_mask.sum() > 0:
                l_action = l_action / eff_mask.sum()

            l_gnm = alpha_dist * l_dist + (1.0 - alpha_dist) * l_action

            # ---- Single backward -----------------------------------------
            # l_lange3d (weight=1) → updates lange3d_model directly
            # lambda_action * l_gnm → updates gnm_model + lange3d_model (via lang_preds)
            total_loss = l_lange3d + lambda_action * l_gnm
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(lange3d_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(gnm_model.parameters(), 1.0)
            lange3d_optimizer.step()
            gnm_optimizer.step()

            running_loss_total += total_loss.item()
            running_loss_lang  += l_lang.item()
            running_loss_gnm   += l_gnm.item()
            running_loss_ogcl  += l_ogcl.item()
            n_batches += 1

            if (batch_idx + 1) % log_freq == 0:
                avg_total = running_loss_total / n_batches
                avg_lang  = running_loss_lang  / n_batches
                avg_gnm   = running_loss_gnm   / n_batches
                avg_ogcl  = running_loss_ogcl  / n_batches
                print(
                    f"  Epoch {epoch} | step {batch_idx+1}"
                    f" | total={avg_total:.4f}"
                    f"  lang={avg_lang:.4f}  ogcl={avg_ogcl:.4f}  gnm={avg_gnm:.4f}"
                )
                wandb.log({
                    "train/loss_total": avg_total,
                    "train/loss_lang":  avg_lang,
                    "train/loss_ogcl":  avg_ogcl,
                    "train/loss_gnm":   avg_gnm,
                    "epoch": epoch,
                })
                running_loss_total = running_loss_lang = running_loss_gnm = running_loss_ogcl = 0.0
                n_batches = 0
            # if batch_idx >2:
            #     break
        # LR schedulers
        if lange3d_scheduler is not None:
            lange3d_scheduler.step()
        if gnm_scheduler is not None:
            gnm_scheduler.step()

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        lange3d_model.eval()
        gnm_model.eval()

        val_loss_total = 0.0
        val_loss_lang  = 0.0
        val_loss_ogcl  = 0.0
        val_loss_gnm   = 0.0
        val_n = 0

        # Collect GT and predicted costs across val batches for Spearman ρ
        _gt_costs_all:   list = []
        _pred_costs_all: list = []

        viz_samples: list = []   # collect first few samples for canvas logging
        viz_done    = False

        with torch.no_grad():
            i=0
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                nai_input_ids = batch["nai_input_ids"].to(device)
                nai_attn_mask = batch["nai_attn_mask"].to(device)
                masks_list    = [m.to(device) for m in batch["masks_list"]]
                gt_costs_list = [c.to(device) for c in batch["gt_costs_list"]]
                gnm_masks     = batch["gnm_masks"].to(device)
                K_list        = batch["K_list"]
                action_label  = batch["action_label"].to(device)
                wp_mask       = batch["wp_mask"].to(device)           # [B, T]
                action_mask   = batch["action_mask"].to(device)
                dist_label    = batch["dist_label"].to(device)

                lang_preds, _ = lange3d_model(
                    pixel_values, masks_list, nai_input_ids, nai_attn_mask
                )
                l_lang, _ = lange3d_loss_fn(lang_preds, gt_costs_list)

                # OGCL on val set — same as train.py
                l_ogcl_v = torch.tensor(0.0, device=device)
                if ogcl_criterion is not None:
                    cm_dev = [m.to(device) for m in batch.get('class_match_list', [])]
                    if cm_dev:
                        l_ogcl_v = ogcl_criterion(lang_preds, cm_dev)

                goal_enc = topopaths.build_differentiable_goal(
                    lang_preds, gnm_masks, K_list, device
                )
                dims = goal_enc.shape[1] - 3
                _, goal_img = goal_enc.split([3, dims], dim=1)

                B = pixel_values.shape[0]
                obs_img = torch.zeros(B, 3, 120, 160, device=device)
                dist_pred, action_pred = gnm_model(obs_img, goal_img)

                dist_label_f = dist_label.float().unsqueeze(1)
                l_dist   = F.mse_loss(dist_pred, dist_label_f)
                a_diff   = (action_pred - action_label) ** 2
                eff_mask_v = wp_mask * action_mask.unsqueeze(1)  # [B, T]
                l_action_v = (a_diff.mean(-1) * eff_mask_v).sum()
                if eff_mask_v.sum() > 0:
                    l_action_v = l_action_v / eff_mask_v.sum()

                l_gnm_v = alpha_dist * l_dist + (1.0 - alpha_dist) * l_action_v
                total_v = l_lang + lambda_ogcl * l_ogcl_v + lambda_action * l_gnm_v

                val_loss_total += total_v.item()
                val_loss_lang  += l_lang.item()
                val_loss_ogcl  += l_ogcl_v.item()
                val_loss_gnm   += l_gnm_v.item()
                val_n += 1

                # Min-max normalise raw logits to [0, 1] for Spearman tracking
                for b in range(pixel_values.shape[0]):
                    _gt_costs_all.append(gt_costs_list[b].cpu().numpy())
                    raw = lang_preds[b].detach().cpu().float()
                    rng = raw.max() - raw.min()
                    _pred_costs_all.append(
                        ((raw - raw.min()) / (rng + 1e-8)).numpy()
                    )

                # Collect viz samples from the first validation batch only.
                if not viz_done:
                    pv_cpu = pixel_values.cpu()
                    ap_cpu = action_pred.cpu()
                    nai_texts = batch.get("nai_text", [""] * B)
                    for b in range(min(4, B)):
                        viz_samples.append({
                            "pixel_values": pv_cpu[b],
                            "masks":        batch["masks_list"][b],
                            "gt_costs":     gt_costs_list[b].cpu(),
                            "pred_costs":   (lambda p: (p - p.min()) / (p.max() - p.min() + 1e-8))(lang_preds[b].cpu()),
                            "gt_action":    batch["action_label"][b],    # CPU
                            "pred_action":  ap_cpu[b],
                            "nai_text":     nai_texts[b] if b < len(nai_texts) else "",
                        })
                    viz_done = True
                # i+=1
                # if i>2:
                #     break
        avg_val      = val_loss_total / max(val_n, 1)
        avg_val_lang = val_loss_lang  / max(val_n, 1)
        avg_val_ogcl = val_loss_ogcl  / max(val_n, 1)
        avg_val_gnm  = val_loss_gnm   / max(val_n, 1)

        # ---- Spearman rank correlation (cost prediction quality) ---------
        val_spearman = float("nan")
        if _SCIPY_OK and _gt_costs_all:
            try:
                gt_flat   = np.concatenate(_gt_costs_all)
                pred_flat = np.concatenate(_pred_costs_all)
                if len(gt_flat) > 2 and gt_flat.std() > 1e-8:
                    val_spearman = float(_spearmanr(gt_flat, pred_flat).correlation)
            except Exception:
                pass

        success_metrics = compute_success_rate(
            gnm_model, lange3d_model, val_loader, topopaths, device,
            max_batches=20,
        )
        success_rate = success_metrics["success_rate"]
        ndtw         = success_metrics["nDTW"]
        print(
            f"Epoch {epoch} | val_loss={avg_val:.4f}"
            f"  lang={avg_val_lang:.4f}  ogcl={avg_val_ogcl:.4f}"
            f"  gnm={avg_val_gnm:.4f}  success={success_rate:.3f}"
            f"  nDTW={ndtw:.3f}  spearman={val_spearman:.3f}"
        )

        # ---- Build visualisation canvases --------------------------------
        canvas_imgs = []
        try:
            from vis_utils.cost_overlay import make_joint_val_canvas
            for idx, s in enumerate(viz_samples):
                canvas = make_joint_val_canvas(
                    s["pixel_values"], s["masks"],
                    s["gt_costs"], s["pred_costs"],
                    s["gt_action"], s["pred_action"],
                    nai_text=s.get("nai_text", ""),
                )
                canvas_imgs.append(
                    wandb.Image(canvas, caption=f"ep{epoch}_s{idx}")
                )
        except Exception as exc:
            print(f"[viz] canvas generation failed: {exc}")

        log_dict = {
            "val/loss_total":       avg_val,
            "val/loss_lang":        avg_val_lang,
            "val/loss_ogcl":        avg_val_ogcl,
            "val/loss_gnm":         avg_val_gnm,
            "val/success_rate":     success_rate,
            "val/nDTW":             ndtw,
            "val/spearman_rho":     val_spearman,
            "epoch": epoch,
        }
        if canvas_imgs:
            log_dict["val/viz"] = canvas_imgs
        wandb.log(log_dict)

        # Save checkpoint — criterion is val lang loss (pure cost-prediction
        # quality), not total loss which includes GNM and can mask lang regression.
        if avg_val_lang < best_val_loss:
            best_val_loss = avg_val_lang
            torch.save({
                "epoch":            epoch,
                "lange3d":          lange3d_model.state_dict(),
                "gnm":              gnm_model.state_dict(),
                "lange3d_opt":      lange3d_optimizer.state_dict(),
                "gnm_opt":          gnm_optimizer.state_dict(),
                "val_loss_lang":    avg_val_lang,
                "val_spearman":     val_spearman,
            }, os.path.join(save_dir, "best_joint.pth"))
            print(f"  → best checkpoint saved (val_lang={avg_val_lang:.4f}, spearman={val_spearman:.3f})")

        torch.save({
            "epoch":        epoch,
            "lange3d":      lange3d_model.state_dict(),
            "gnm":          gnm_model.state_dict(),
            "val_loss":     avg_val,
        }, os.path.join(save_dir, f"latest.pth"))

    print("[train_eval_loop_joint] Done.")

"""
joint_train.py
==============
Entry point for jointly training LangGeoNetV2 (object-cost predictor) and
GNM (goal-conditioned navigation model).

Usage:
------
    cd /data/ws/VLN-CE/controller/object_react/train
    python joint_train.py --config config/joint_training.yaml

The script:
  1. Loads config from YAML.
  2. Builds LangGeoNetV2 (optionally from a pre-trained checkpoint).
  3. Builds GNM (optionally from a pre-trained checkpoint).
  4. Builds JointEpisodeDataset + DataLoaders.
  5. Builds TopoPaths (differentiable goal composer).
  6. Calls train_eval_loop_joint.
"""

from __future__ import annotations
import argparse
import os
import sys
import yaml
import wandb

import torch
import torch.optim as optim

# ---------------------------------------------------------------------------
# Path setup — ensure both train packages are importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "lange3dnet_train"))

from vint_train.models.gnm.gnm import GNM
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.training.train_eval_loop import train_eval_loop_joint, count_parameters

from lange3dnet_train.model import LangGeoNetV2
from lange3dnet_train.losses import LangGeoNetLoss, ObjectGroundingContrastiveLoss
from lange3dnet_train.joint_dataset import create_joint_dataloaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_gnm(cfg: dict, device: torch.device) -> GNM:
    gnm = GNM(
        context_size=cfg.get("context_size", 5),
        len_traj_pred=cfg.get("len_traj_pred", 10),
        learn_angle=cfg.get("learn_angle", True),
        obs_encoding_size=cfg.get("obs_encoding_size", 1024),
        goal_encoding_size=cfg.get("goal_encoding_size", 1024),
        goal_type=cfg.get("goal_type", "image_mask_enc"),
        obs_type=cfg.get("obs_type", "disabled"),
        dims=cfg.get("dims", 8),
        use_mask_grad=cfg.get("use_mask_grad", False),
        goal_uses_context=cfg.get("goal_uses_context", False),
    ).to(device)
    return gnm


def load_gnm_checkpoint(gnm: GNM, ckpt_path: str, device: torch.device) -> GNM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
        try:
            state = state.module.state_dict()
        except AttributeError:
            state = state.state_dict()
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    missing, unexpected = gnm.load_state_dict(state, strict=False)
    if missing:
        print(f"[GNM checkpoint] missing keys: {missing[:5]} …")
    if unexpected:
        print(f"[GNM checkpoint] unexpected keys: {unexpected[:5]} …")
    print(f"[GNM] loaded from {ckpt_path}")
    return gnm


def build_lange3d(cfg: dict, device: torch.device) -> LangGeoNetV2:
    model = LangGeoNetV2(
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 2),
        clip_model_name=cfg.get("clip_model", "openai/clip-vit-base-patch16"),
        dino_model_name=cfg.get("dino_model", "facebook/dinov2-small"),
        freeze_clip=cfg.get("freeze_clip", True),
        freeze_dino=cfg.get("freeze_dino", True),
    ).to(device)

    # Freeze cost/ranking heads so gradients only flow through the backbone.
    for mod_name in ("cost_head", "rank_refine", "rank_head"):
        mod = getattr(model, mod_name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = False
            print(f"[LangGeoNetV2] frozen: {mod_name}")
        else:
            print(f"[LangGeoNetV2] warning: module '{mod_name}' not found — skipping freeze")

    return model


def load_lange3d_checkpoint(
    model: LangGeoNetV2, ckpt_path: str, device: torch.device
) -> LangGeoNetV2:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[LangE3D checkpoint] missing keys: {missing[:5]} …")
    if unexpected:
        print(f"[LangE3D checkpoint] unexpected keys: {unexpected[:5]} …")
    print(f"[LangE3D] loaded from {ckpt_path}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Joint LangE3D + GNM training")
    parser.add_argument("--config", required=True, help="Path to joint_training.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config loaded from {args.config}")

    # ---- Device -----------------------------------------------------------
    gpu_ids = cfg.get("gpu_ids", [0])
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Seed -------------------------------------------------------------
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # ---- WandB ------------------------------------------------------------
    use_wandb = cfg.get("use_wandb", False)
    if use_wandb:
        wandb.init(
            project=cfg.get("wandb_project", "joint_lang_gnm"),
            name=cfg.get("run_name", "joint_run"),
            config=cfg,
        )

    # ---- Build models -----------------------------------------------------
    print("Building GNM …")
    gnm = build_gnm(cfg, device)
    if cfg.get("gnm_checkpoint"):
        gnm = load_gnm_checkpoint(gnm, cfg["gnm_checkpoint"], device)
    count_parameters(gnm)

    print("Building LangGeoNetV2 …")
    lange3d = build_lange3d(cfg, device)
    if cfg.get("lange3d_checkpoint"):
        lange3d = load_lange3d_checkpoint(lange3d, cfg["lange3d_checkpoint"], device)
    count_parameters(lange3d)

    # Multi-GPU (data-parallel), if requested.
    if len(gpu_ids) > 1 and torch.cuda.is_available():
        gnm     = torch.nn.DataParallel(gnm,     device_ids=gpu_ids)
        lange3d = torch.nn.DataParallel(lange3d, device_ids=gpu_ids)

    # ---- TopoPaths (differentiable goal builder) --------------------------
    topopaths = TopoPaths(
        dims=cfg.get("dims", 8),
        w=cfg.get("mask_w", 160),
        h=cfg.get("mask_h", 120),
    )
    print("TopoPaths ready.")

    # ---- LangGeoNet loss --------------------------------------------------
    lange3d_loss = LangGeoNetLoss(
        lambda_rank=cfg.get("lambda_rank", 0.5),
        lambda_si=cfg.get("lambda_si", 0.3),
    ).to(device)

    # ---- Object Grounding Contrastive Loss --------------------------------
    ogcl = ObjectGroundingContrastiveLoss(
        margin=cfg.get("ogcl_margin", 0.3),
    ).to(device)

    # ---- Optimizers -------------------------------------------------------
    _backbone_names = {"clip", "dino", "bert"}

    def _param_groups(model, lr_head, lr_bb):
        head_p, bb_p = [], []
        mdl = model.module if hasattr(model, "module") else model
        for name, p in mdl.named_parameters():
            if not p.requires_grad:
                continue
            if any(bn in name for bn in _backbone_names):
                bb_p.append(p)
            else:
                head_p.append(p)
        return [
            {"params": head_p, "lr": lr_head},
            {"params": bb_p,   "lr": lr_bb},
        ]

    lange3d_opt = optim.AdamW(
        _param_groups(
            lange3d,
            lr_head=cfg.get("lange3d_lr_head", 1e-4),
            lr_bb=cfg.get("lange3d_lr_backbone", 1e-5),
        ),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    gnm_opt = optim.Adam(
        filter(lambda p: p.requires_grad,
               (gnm.module if hasattr(gnm, "module") else gnm).parameters()),
        lr=cfg.get("gnm_lr", 7e-4),
    )

    # ---- LR schedulers (cosine) ------------------------------------------
    epochs = cfg.get("epochs", 30)
    warmup = cfg.get("warmup_epochs", 3)

    def _cosine_sched(opt, epochs, warmup):
        return optim.lr_scheduler.LambdaLR(opt, lambda ep: (
            (ep + 1) / max(warmup, 1) if ep < warmup
            else 0.5 * (1 + torch.cos(torch.tensor(
                3.14159 * (ep - warmup) / max(1, epochs - warmup)
            )).item())
        ))

    lange3d_sched = _cosine_sched(lange3d_opt, epochs, warmup)
    gnm_sched     = _cosine_sched(gnm_opt, epochs, warmup)

    # ---- Data loaders -----------------------------------------------------
    print("Building JointEpisodeDataset …")
    train_loader, val_loader = create_joint_dataloaders(
        h5_path=cfg["h5_path"],
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 8),
        val_split=cfg.get("val_split", 0.1),
        seed=seed,
        clip_model=cfg.get("clip_model", "openai/clip-vit-base-patch16"),
        len_traj_pred=cfg.get("len_traj_pred", 10),
        min_action_dist=cfg.get("action", {}).get("min_dist_cat", 2),
        max_action_dist=cfg.get("action", {}).get("max_dist_cat", 10),
    )
    print(
        f"  train: {len(train_loader.dataset)} frames"
        f" | val: {len(val_loader.dataset)} frames"
    )

    # ---- Run training loop ------------------------------------------------
    print("Starting joint training …")
    train_eval_loop_joint(
        lange3d_model=lange3d,
        gnm_model=gnm,
        lange3d_optimizer=lange3d_opt,
        gnm_optimizer=gnm_opt,
        lange3d_scheduler=lange3d_sched,
        gnm_scheduler=gnm_sched,
        train_loader=train_loader,
        val_loader=val_loader,
        topopaths=topopaths,
        lange3d_loss_fn=lange3d_loss,
        epochs=epochs,
        device=device,
        lambda_action=cfg.get("lambda_action", 0.01),
        alpha_dist=cfg.get("alpha", 0.5),
        log_freq=cfg.get("print_log_freq", 100),
        save_dir=cfg.get("output_dir", "./checkpoints/joint"),
        project_name=cfg.get("wandb_project", "joint_lang_gnm"),
        ogcl_criterion=ogcl,
        lambda_ogcl=cfg.get("lambda_ogcl", 1.0),
    )

    if use_wandb:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()

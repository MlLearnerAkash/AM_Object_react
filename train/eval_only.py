#!/usr/bin/env python3
"""
eval_only.py
============
Evaluation-only script for a pretrained Object-React model (GNM + optional
LangGeoNetV2).  Uses the same config, dataloaders, and evaluate() function as
train.py, but runs a single evaluation pass on the test set without any
training.

Two costmap sources are available (``--cost_source``):
  * ``predicted`` (default) — LangGeoNetV2 predicts per-object costs from NAI
    text + goal image; the predicted costmap replaces the GT one.
  * ``gt``                — use the ground-truth costmap embedded in the
    dataset (no LangGeoNetV2 needed).

Usage:
    cd /data/ws/VLN-CE/controller/object_react/train
    # Predicted costmaps (LangGeoNetV2 + GNM)
    python eval_only.py -c config/eval_only.yaml --cost_source predicted
    # GT costmaps from dataset (GNM only)
    python eval_only.py -c config/eval_only.yaml --cost_source gt
    # Using a specific config
    python eval_only.py -c config/e3d_obj.yaml --cost_source predicted

The GNM model weights are loaded from ``gnm_pretrained_checkpoint`` in the
config.  For ``--cost_source predicted`` the LangGeoNetV2 weights are loaded
from ``lange3d_checkpoint``.  Results are logged to console, and optionally
to wandb (controlled by config["use_wandb"]).
"""

import argparse
import os
import time
from os.path import normpath

import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

# ---------------------------------------------------------------------------
# Model imports  (same as train.py)
# ---------------------------------------------------------------------------
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.training.train_utils import evaluate, evaluate_nomad

from lange3dnet_train.model import LangGeoNetV2
from lange3dnet_train.losses import LangGeoNetLoss

import matplotlib

matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

plt.ioff()


# ---------------------------------------------------------------------------
# Collation helper  (identical to train.py)
# ---------------------------------------------------------------------------
def _collate_with_lange3d(batch):
    """Default-collate the first 7 ViNT outputs and pad the 8th lange3d-inputs
    dict (which has ragged per-sample masks)."""
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
    gnm_masks_padded = torch.zeros(
        len(batch), max(K_max, 1), Hm, Wm, dtype=torch.float32,
    )
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
# Dataloader construction  (test-only version of ready_dataloaders)
# ---------------------------------------------------------------------------
def build_test_dataloaders(config, **kwargs):
    """Build **only** the test dataloaders from the config.

    Returns a dict ``{dataset_name}_{split}: DataLoader``, exactly as the
    ``test_dataloaders`` dict produced by ``train.py``'s ``ready_dataloaders``.
    """
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    for dataset_name in config["datasets"]:
        dcfg = config["datasets"][dataset_name]
        if "negative_mining" not in dcfg:
            dcfg["negative_mining"] = True
        if "goals_per_obs" not in dcfg:
            dcfg["goals_per_obs"] = 1
        if "end_slack" not in dcfg:
            dcfg["end_slack"] = 0
        if "waypoint_spacing" not in dcfg:
            dcfg["waypoint_spacing"] = 1

        for split_key in ("test",):
            if split_key not in dcfg:
                continue
            dataset = ViNT_Dataset(
                data_folder=dcfg["data_folder"],
                data_split_folder=dcfg[split_key],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=dcfg["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=dcfg["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=dcfg["end_slack"],
                goals_per_obs=dcfg["goals_per_obs"],
                normalize=config["normalize"],
                **kwargs,
            )
            tag = f"{dataset_name}_{split_key}"
            test_dataloaders[tag] = dataset

    use_lange3d_collate = bool(kwargs.get("return_lange3d_inputs", False))
    collate_fn = _collate_with_lange3d if use_lange3d_collate else None

    eval_batch_size = config.get("eval_batch_size", config["batch_size"])
    eval_num_workers = config.get("eval_num_workers", config.get("num_workers", 0))

    for tag, ds in test_dataloaders.items():
        test_dataloaders[tag] = DataLoader(
            ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            drop_last=False,
            persistent_workers=True if eval_num_workers > 0 else False,
            collate_fn=collate_fn,
        )

    return test_dataloaders


# ---------------------------------------------------------------------------
# Model construction  (same as train.py)
# ---------------------------------------------------------------------------
def build_model(config, **kwargs):
    """Create the navigation model (GNM / ViNT / NoMaD) and optional
    noise_scheduler.

    Returns ``(model, noise_scheduler)``.
    """
    noise_scheduler = None
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
            **kwargs,
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vib":
            vision_encoder = ViB(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "vit":
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else:
            raise ValueError(
                f"Vision encoder {config['vision_encoder']} not supported"
            )

        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.models.conditional_unet_1d import ConditionalUnet1D

        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])

        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    else:
        raise ValueError(f"Model type {config['model_type']} not supported")
    return model, noise_scheduler


# ---------------------------------------------------------------------------
# Main evaluation entry-point
# ---------------------------------------------------------------------------
def main(config):
    # ---- kwargs forwarded to dataset / model / evaluate -------------------
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
        "return_lange3d_inputs": True,  # needed for GT costs (8th element) in both modes
        "clip_model_name": config.get(
            "lange3d_clip_model", "openai/clip-vit-base-patch16"
        ),
        "gnm_mask_h": config.get("gnm_mask_h", 60),
        "gnm_mask_w": config.get("gnm_mask_w", 80),
        "clip_grad_norm": config.get("clip_grad_norm", 1.0),
        "max_traj_len": config.get("max_traj_len", None),
        "filter_dead_samples": config.get("filter_dead_samples", False),
        "hm3d_to_habitat": config.get("hm3d_to_habitat", False),
        "viz_images_per_batch": config.get("viz_images_per_batch", None),
        "eval_negatives": config.get("eval_negatives", False),
    }

    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    # ---- Device setup ----------------------------------------------------
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in config["gpu_ids"]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True
    cudnn.benchmark = True

    # ---- Image transform -------------------------------------------------
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    # ---- Dataloaders (test only) -----------------------------------------
    print("Building test dataloaders...")
    test_dataloaders = build_test_dataloaders(config, **kwargs)
    for name, loader in test_dataloaders.items():
        print(f"  {name}: {len(loader.dataset)} samples  (batch_size={loader.batch_size})")

    # ---- Navigation model -------------------------------------------------
    print("Building model...")
    model, noise_scheduler = build_model(config, **kwargs)

    # Load GNM weights from gnm_pretrained_checkpoint (used in both modes)
    ckpt_path = config.get("gnm_pretrained_checkpoint")
    if ckpt_path is None:
        raise RuntimeError(
            "'gnm_pretrained_checkpoint' must be set in the config. "
            "This is the trained Object-React GNM weights file."
        )
    print(f"Loading GNM from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        state = ckpt["model"]
        if hasattr(state, "state_dict"):
            state = state.state_dict()
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print(f"  loaded {len(state) - len(unexpected)}/{len(state)} keys")

    # ---- Build TopoPaths (needed by _maybe_predict_goal_with_lange3d in    
    #       both gt and predicted modes) -----------------------------------
    topopaths = TopoPaths(
        dims=config.get("dims", 8),
        w=config.get("mask_w", 160),
        h=config.get("mask_h", 120),
    )
    kwargs["topopaths"] = topopaths

    # ---- LangGeoNetV2 (only when using predicted costmaps) ---------------
    lange3d_model = None
    if config["cost_source"] == "predicted":
        ckpt_path = config.get("lange3d_checkpoint")
        if ckpt_path is None:
            raise RuntimeError(
                "--cost_source predicted requires a LangGeoNetV2 checkpoint. "
                "Either use --cost_source gt (uses GT costmaps from the dataset) "
                "or set 'lange3d_checkpoint' in your config file."
            )
        print("Building LangGeoNetV2 (predicted-cost goal image)...")
        lange3d_model = LangGeoNetV2(
            d_model=config.get("lange3d_d_model", 256),
            n_heads=config.get("lange3d_n_heads", 8),
            n_layers=config.get("lange3d_n_layers", 2),
            clip_model_name=config.get(
                "lange3d_clip_model", "openai/clip-vit-base-patch16"
            ),
            dino_model_name=config.get(
                "lange3d_dino_model", "facebook/dinov2-small"
            ),
            freeze_clip=config.get("lange3d_freeze_clip", True),
            freeze_dino=config.get("lange3d_freeze_dino", True),
        )
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        miss, unexp = lange3d_model.load_state_dict(state, strict=False)
        print(
            f"  loaded {ckpt_path} | "
            f"missing={len(miss)} unexpected={len(unexp)}"
        )
        lange3d_model = lange3d_model.to(device)
        lange3d_model.eval()
        kwargs["lange3d_model"] = lange3d_model

    # ---- Multi-GPU & device move -----------------------------------------
    current_epoch = 0  # epoch counter for logging purposes only
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)
    model.eval()

    # ---- Project folder (for log output) ---------------------------------
    project_folder = config.get(
        "project_folder",
        os.path.join("logs", "eval", config.get("run_name", "eval_only")),
    )
    os.makedirs(project_folder, exist_ok=True)

    # ---- Disable wandb logging for eval-only (override via --wandb) -------
    use_wandb = config.get("use_wandb", False) and (not args.no_wandb)

    import wandb
    if use_wandb:
        wandb.login()
        wandb.init(
            project=config.get("project_name", "object_rel_nav"),
            settings=wandb.Settings(start_method="fork"),
            resume="allow",
            name=config.get("run_name", "eval_only") + "_eval",
            config=config,
        )

    # ---- Run evaluation on every test dataloader -------------------------
    print(f"\n{'='*60}")
    print(f"Starting evaluation (epoch ~{current_epoch})")
    print(f"{'='*60}\n")

    if config["model_type"] in ("vint", "gnm"):
        for dataset_type, loader in test_dataloaders.items():
            print(f"\n--- Evaluating {dataset_type} ---")
            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=config["normalize"],
                epoch=current_epoch,
                alpha=config.get("alpha", 0.5),
                learn_angle=config.get("learn_angle", True),
                num_images_log=config.get("num_images_log", 8),
                use_wandb=use_wandb,
                eval_fraction=config.get("eval_fraction", 1.0),
                use_tqdm=True,
                **kwargs,
            )
            print(
                f"  {dataset_type}  |  dist_loss={test_dist_loss:.6f}  "
                f"action_loss={test_action_loss:.6f}  "
                f"total_loss={total_eval_loss:.6f}"
            )
    elif config["model_type"] == "nomad":
        if noise_scheduler is None:
            raise RuntimeError(
                "NoMaD evaluation requires a noise_scheduler, but none was built."
            )
        for dataset_type, loader in test_dataloaders.items():
            print(f"\n--- Evaluating {dataset_type} (NoMaD) ---")
            evaluate_nomad(
                eval_type=dataset_type,
                ema_model=None,  # eval without EMA when explicitly loading ckpt
                dataloader=loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=config.get("goal_mask_prob", 0.0),
                project_folder=project_folder,
                epoch=current_epoch,
                print_log_freq=config.get("print_log_freq", 100),
                num_images_log=config.get("num_images_log", 8),
                wandb_log_freq=config.get("wandb_log_freq", 10),
                use_wandb=use_wandb,
                eval_fraction=config.get("eval_fraction", 1.0),
            )

    if use_wandb:
        wandb.finish()

    print("\nEvaluation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained Object-React model on test data"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/object_react.yaml",
        type=str,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=None,
        help="Override: fraction of test data to evaluate on",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override: device string, e.g. 'cuda:0' or 'cpu'",
    )
    parser.add_argument(
        "--cost_source",
        type=str,
        default="predicted",
        choices=["gt", "predicted"],
        help=(
            "Which costmap to use: 'predicted' (LangGeoNetV2 from checkpoint) "
            "or 'gt' (ground-truth costmap embedded in the dataset)"
        ),
    )

    args = parser.parse_args()

    # ---- Load config -----------------------------------------------------
    with open("config/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    # ---- Override eval_fraction ------------------------------------------
    if args.eval_fraction is not None:
        config["eval_fraction"] = args.eval_fraction

    # ---- Override device -------------------------------------------------
    if args.device is not None:
        config["gpu_ids"] = [int(args.device.split(":")[-1])]

    # ---- Cost-source mode ------------------------------------------------
    config["cost_source"] = args.cost_source
    if args.cost_source == "gt":
        # GT costmap: no LangGeoNetV2; the dataset's 8th element provides
        # real GT costs (from traj_data.pkl gt_costs + masks/*.npz).
        config["use_lange3d"] = False
        config["goal_type"] = "image_mask_enc"
        config["obs_type"] = config.get("obs_type", "disabled")
        print(f"[cost_source] Using GT costmaps from dataset")
    else:
        # Predicted costmap: load LangGeoNetV2.
        config["use_lange3d"] = True
        config["goal_type"] = "image_mask_enc"
        config["obs_type"] = config.get("obs_type", "disabled")
        print(f"[cost_source] Using LangGeoNetV2 predicted costmaps")

    # ---- Set up project folder (for logs) --------------------------------
    config["run_name"] = (
        config.get("run_name", "eval_only")
        + "_eval_"
        + time.strftime("%Y_%m_%d_%H_%M_%S")
    )
    config["project_folder"] = normpath(
        os.path.join("logs", config.get("project_name", "eval"), config["run_name"])
    )
    os.makedirs(config["project_folder"], exist_ok=True)

    # Force mode to eval (never train)
    config["mode"] = "eval"

    print("Config:")
    print(yaml.dump(config, default_flow_style=False))

    main(config)

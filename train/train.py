import os
import wandb
import argparse
import numpy as np
import yaml
import time
import pdb
import random
from os.path import basename, normpath

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.optimization import get_scheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

# from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)

from lange3dnet_train.model import LangGeoNetV2
from lange3dnet_train.losses import LangGeoNetLoss

import visualize

# import check_internet
import matplotlib.pyplot as plt

# Turn off interactive mode
plt.ioff()


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
    K_max  = max(K_list) if K_list else 0
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


def ready_dataloaders(config, data_config, dataset_name, data_split_type, **kwargs):

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                dataset = ViNT_Dataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config["image_size"],
                    waypoint_spacing=data_config["waypoint_spacing"],
                    min_dist_cat=config["distance"]["min_dist_cat"],
                    max_dist_cat=config["distance"]["max_dist_cat"],
                    min_action_distance=config["action"]["min_dist_cat"],
                    max_action_distance=config["action"]["max_dist_cat"],
                    negative_mining=data_config["negative_mining"],
                    len_traj_pred=config["len_traj_pred"],
                    learn_angle=config["learn_angle"],
                    context_size=config["context_size"],
                    context_type=config["context_type"],
                    end_slack=data_config["end_slack"],
                    goals_per_obs=data_config["goals_per_obs"],
                    normalize=config["normalize"],
                    # goal_type=config["goal_type"],
                    **kwargs,
                )
                if data_split_type == "train":
                    train_dataset.append(dataset)
                else:
                    dataset_type = f"{dataset_name}_{data_split_type}"
                    if dataset_type not in test_dataloaders:
                        test_dataloaders[dataset_type] = {}
                    test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    use_lange3d_collate = bool(kwargs.get("return_lange3d_inputs", False))
    collate_fn = _collate_with_lange3d if use_lange3d_collate else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True if config["num_workers"] > 0 else False,
        collate_fn=collate_fn,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=config["eval_num_workers"],
            drop_last=False,
            persistent_workers=True if config["eval_num_workers"] > 0 else False,
            collate_fn=collate_fn,
        )
    return train_dataset, train_loader, test_dataloaders


def ready_model(config, **kwargs):

    # Create the model
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
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

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
        raise ValueError(f"Model {config['model']} not supported")
    return model, noise_scheduler


def ready_trainer(config, model):
    if config["clipping"]:
        print("Clipping gradients to", config["max_norm"])
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.0,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    return optimizer, scheduler


def resume_model(config, model, load_project_folder=None):
    if load_project_folder is None:
        load_project_folder = os.path.join("logs", config["load_run"])
    print("Loading model from ", load_project_folder)
    latest_path = os.path.join(load_project_folder, "latest.pth")
    latest_checkpoint = torch.load(
        latest_path, map_location=f"cuda:0" if torch.cuda.is_available() else "cpu"
    )  # TODO: fails when visible = [1]
    load_model(model, config["model_type"], latest_checkpoint)
    current_epoch = 0
    if "epoch" in latest_checkpoint:
        current_epoch = latest_checkpoint["epoch"] + 1
    return latest_checkpoint, current_epoch


def resume_trainer(latest_checkpoint, optimizer, scheduler):
    if "optimizer" in latest_checkpoint:
        optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
    if scheduler is not None and "scheduler" in latest_checkpoint:
        scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
    return optimizer, scheduler


def main(config):

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
        # returns the goal-frame inputs (8th tuple element) and the train
        "return_lange3d_inputs": bool(config.get("use_lange3d", False)),
        "clip_model_name": config.get(
            "lange3d_clip_model", "openai/clip-vit-base-patch16"
        ),
        "gnm_mask_h": config.get("gnm_mask_h", 60),
        "gnm_mask_w": config.get("gnm_mask_w", 80),
        "clip_grad_norm": config.get("clip_grad_norm", 1.0),
        "max_traj_len":   config.get("max_traj_len", None),
        "filter_dead_samples": config.get("filter_dead_samples", False),
    }

    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
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
        # seed = config["seed"]
        # os.environ["PL_GLOBAL_SEED"] = str(seed)
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # os.environ["PL_SEED_WORKERS"] = f"{int(config['num_workers'])}"

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform)

    print("Readying dataloaders...")
    train_dataset, train_loader, test_dataloaders = ready_dataloaders(
        config, config["datasets"], "vint", "train", **kwargs
    )

    print("Readying model...")
    model, noise_scheduler = ready_model(config, **kwargs)

    print("Readying trainer...")
    optimizer, scheduler = ready_trainer(config, model)
    current_epoch = 0

    # ---- LangGeoNetV2 (predicted costs replace GT costmap as goal image) ----
    lange3d_model = None
    topopaths     = None
    if config.get("use_lange3d", False):
        print("Readying LangGeoNetV2 (predicted-cost goal image)...")
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
        if config.get("lange3d_checkpoint"):
            ck = torch.load(
                config["lange3d_checkpoint"],
                map_location=f"cuda:{first_gpu_id}"
                if torch.cuda.is_available() else "cpu",
                weights_only=False,
            )
            state = ck.get("model_state_dict", ck)
            miss, unexp = lange3d_model.load_state_dict(state, strict=False)
            print(f"  loaded {config['lange3d_checkpoint']} | "
                  f"missing={len(miss)} unexpected={len(unexp)}")
        #Applying different learning rate
        head_params, backbone_params = [], []
        for n, p in lange3d_model.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = n.startswith("clip.") or n.startswith("dino.")
            (backbone_params if is_backbone else head_params).append(p)
        lr_head = float(config.get(
            "lange3d_lr_head", config.get("lange3d_lr_head", config["lange3d_lr_head"])
        ))
        lr_bb   = float(config.get("lange3d_lr_backbone", lr_head * 0.1))
        if head_params:
            optimizer.add_param_group({"params": head_params, "lr": lr_head, "is_lange3d": True})
        if backbone_params:
            optimizer.add_param_group({"params": backbone_params, "lr": lr_bb, "is_lange3d": True})
        print(
            f"  lange3d optimizer groups: head={len(head_params)} "
            f"params @ lr={lr_head:.1e}, backbone={len(backbone_params)} "
            f"params @ lr={lr_bb:.1e}"
        )
        
        # Build a TopoPaths for the train loop (uses build_differentiable_goal).
        topopaths = TopoPaths(
            dims=config.get("dims", 8),
            w=config.get("mask_w", 160),
            h=config.get("mask_h", 120),
        )
        # Build the supervised cost-predictor loss and expose it to the train loop.
        lange3d_loss = LangGeoNetLoss(
            lambda_rank=float(config.get("lange3d_lambda_rank", 0.3)),
            lambda_si=float(config.get("lange3d_lambda_si", 0.0)),
        )
        lange3d_loss = lange3d_loss.to(
            f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        kwargs["lange3d_model"]  = lange3d_model
        kwargs["topopaths"]      = topopaths
        kwargs["lange3d_loss_fn"] = lange3d_loss
        kwargs["lambda_lange3d"] = float(config.get("lambda_lange3d", 0.1))

    if "load_run" in config:
        print("Resuming model...")
        latest_checkpoint, current_epoch = resume_model(config, model)

    # Multi-GPU
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)
    if lange3d_model is not None:
        lange3d_model = lange3d_model.to(device)
        kwargs["lange3d_model"] = lange3d_model
        if "lange3d_loss_fn" in kwargs:
            kwargs["lange3d_loss_fn"] = kwargs["lange3d_loss_fn"].to(device)

    if (
        "load_run" in config and config["mode"] == "train"
    ):  # load optimizer and scheduler after data parallel
        optimizer, scheduler = resume_trainer(latest_checkpoint, optimizer, scheduler)

    if config["mode"] == "vis":
        visualize.visualize(
            config,
            model,
            test_dataloaders[list(test_dataloaders.keys())[0]],
            current_epoch,
            device,
            transform,
            **kwargs,
        )
        return

    if config["model_type"] == "vint" or config["model_type"] == "gnm":
        train_eval_loop(
            train_model=config["mode"] == "train",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            **kwargs,
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"] == "train",
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
        )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    parser.add_argument(
        "--exp_name", "-n", default="", type=str, help="Experiment name"
    )

    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    if "load_run" in config and config["mode"] == "train":
        config["run_name"] = basename(normpath(config["load_run"]))
        config["project_folder"] = normpath(os.path.join("logs", config["load_run"]))
    else:
        config["run_name"] += (
            "_" + time.strftime("%Y_%m_%d_%H_%M_%S") + "_" + args.exp_name
        )
        config["project_folder"] = normpath(
            os.path.join("logs", config["project_name"], config["run_name"])
        )
        os.makedirs(
            config[
                "project_folder"
            ],  # should error if dir already exists to avoid overwriting and old project
        )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
            resume="allow",
            id=config.get("wandb_id", None),
            # entity="visualNav", # TODO: change this to your wandb entity
        )
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)

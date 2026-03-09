#!/usr/bin/env python3
"""
inference.py  —  Object-React VLN discrete-action inference

Loads a trained model (object_react_vln config) and predicts the next
discrete action(s) given the current RGB observation and a goal costmap.

Actions:  0=STOP   1=FORWARD   2=LEFT   3=RIGHT

─── Quickstart (from the train/ directory) ───────────────────────────────

[A] Pre-built H5 costmap (training / custom dataset)
    python inference.py \
        --config     config/object_react_vln.yaml \
        --checkpoint logs/object_react_vln/<run>/latest.pth \
        --image      /path/to/current.png \
        --costmap_h5 ../custom_dataset/costmaps.h5 \
        --traj_name  ep_21909 \
        --frame_idx  0

[B] Live k-channel masks + costs (no H5 file needed)
    # masks.npy  — numpy file with shape (K, H, W), float32/bool
    # K cost values, one per mask channel
    python inference.py \
        --config     config/object_react_vln.yaml \
        --checkpoint logs/object_react_vln/<run>/latest.pth \
        --image      /path/to/current.png \
        --masks_npy  /path/to/masks.npy \
        --costs      14.4 29.7 5.1 0.8

[C] Full context window (oldest → newest, last = current frame)
    python inference.py ... --image f0.png f1.png f2.png f3.png f4.png f5.png

─── Python API ────────────────────────────────────────────────────────────

    from inference import run_inference, predict_from_image_masks_costs
    from inference import build_model, load_checkpoint, load_config

    # Option A — H5 file
    result = run_inference(
        config_path="config/object_react_vln.yaml",
        checkpoint_path="logs/.../latest.pth",
        image_paths=["current.png"],
        costmap_h5="../custom_dataset/costmaps.h5",
        traj_name="ep_21909", frame_idx=42,
    )

    # Option B — live masks array
    result = run_inference(
        config_path="config/object_react_vln.yaml",
        checkpoint_path="logs/.../latest.pth",
        image_paths=["current.png"],
        masks_khw=masks_array,   # np.ndarray (K, H, W)
        costs=costs_array,       # np.ndarray (K,)
    )

    # Option C — reuse model across frames (most efficient)
    config  = load_config("config/object_react_vln.yaml")
    model   = load_checkpoint(build_model(config), "logs/.../latest.pth", "gnm")
    result  = predict_from_image_masks_costs(
        model=model, config=config,
        image_paths=["current.png"],
        masks_khw=masks_array,   # np.ndarray (K, H, W)
        costs=costs_array,       # np.ndarray (K,)
    )
    print(result["next_action"])   # e.g. "FORWARD"
───────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import argparse

import numpy as np
import yaml
import h5py

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

import sys, types
# Allow checkpoints saved with NumPy 2.x to load under NumPy 1.x
if not hasattr(sys.modules.get("numpy", None), "_core"):
    import numpy
    numpy._core = numpy.core  # type: ignore[attr-defined]
    
# ── ensure the train/ package is importable regardless of cwd ────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from vint_train.models.gnm.gnm import GNM
from vint_train.training.train_eval_loop import load_model
from vint_train.models.object_react.dataloader import TopoPaths

# ── constants ────────────────────────────────────────────────────────────────
ACTION_NAMES = {0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT"}
IMAGE_ASPECT_RATIO = 4.0 / 3.0          # same as data_utils.IMAGE_ASPECT_RATIO
IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# ── image helpers ─────────────────────────────────────────────────────────────

def load_rgb_frame(path: str, image_size) -> torch.Tensor:
    """
    Load one RGB frame, apply 4:3 aspect-ratio centre-crop then resize.
    Matches vint_train.data.data_utils.resize_and_aspect_crop().

    Args:
        path:        path to PNG/JPG image
        image_size:  (W, H) tuple — e.g. (85, 64) from config["image_size"]
    Returns:
        Tensor [3, H, W], float32 in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * IMAGE_ASPECT_RATIO)))
    else:
        img = TF.center_crop(img, (int(w / IMAGE_ASPECT_RATIO), w))
    img = img.resize(image_size)            # PIL.resize takes (W, H)
    return TF.to_tensor(img)               # [3, H, W] float32 in [0,1]


def build_obs_tensor(
    image_paths,
    image_size,
    context_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the observation tensor for GNM with obs_type='image'.

    Produces a (1, 3*(context_size+1), H, W) tensor where:
      - context_size past frames + 1 current frame are stacked
      - each frame is individually normalised with ImageNet stats
      - if fewer than (context_size+1) paths are given the earliest
        frame is repeated to fill the window

    Args:
        image_paths:  list of str, oldest → newest (last = current frame)
        image_size:   (W, H)
        context_size: from config (e.g. 5)
    """
    needed = context_size + 1
    frames = [load_rgb_frame(p, image_size) for p in image_paths]

    # pad on the left by repeating the oldest frame
    while len(frames) < needed:
        frames.insert(0, frames[0].clone())
    frames = frames[-needed:]              # keep the most recent `needed` frames

    # apply ImageNet normalisation to each frame independently
    frames = [IMAGENET_NORM(f) for f in frames]

    obs = torch.cat(frames, dim=0).unsqueeze(0).to(device)   # (1, 3*needed, H, W)
    return obs


# ── goal / costmap helpers ─────────────────────────────────────────────────────

def _build_topopaths(config: dict) -> TopoPaths:
    """Instantiate TopoPaths with the parameters from the training config."""
    return TopoPaths(
        dims=config.get("dims", 8),
        precomputed_filename=None,          # we read the H5 manually
        pl_perturb_ratio=0.0,               # no perturbation at inference
        pl_perturb_type=config.get("pl_perturb_type", "max_val"),
        mask_crop_ratio=config.get("mask_crop_ratio", 1.0),
        use_mask_grad=config.get("use_mask_grad", False),
    )


def load_goal_from_h5(
    h5_path: str,
    traj_name: str,
    frame_idx: int,
    topopaths: TopoPaths,
) -> torch.Tensor:
    """
    Load the goal costmap encoding from a pre-built H5 file.

    Handles two storage formats automatically:
      • Training format  — RLE-encoded masks + a 'size' dataset per entry
      • Custom format    — flat pixel-index arrays, no 'size' dataset

    Args:
        h5_path:    path to the .h5 file (e.g. ../custom_dataset/costmaps.h5)
        traj_name:  trajectory name (e.g. "ep_21909")
        frame_idx:  goal frame index (e.g. 0 → key "ep_21909_0")
        topopaths:  TopoPaths instance (for `create_input`)

    Returns:
        Tensor [dims, H//2, W//2] float32  — the positional-encoding costmap
    """
    key = f"{traj_name}_{frame_idx}"

    with h5py.File(h5_path, "r") as f:
        if key not in f:
            print(f"[inference] Key '{key}' not found in {h5_path}; using default (zero) encoding.")
            enc, _ = topopaths.create_input(None, None)
            return torch.as_tensor(enc, dtype=torch.float32)

        grp = f[key]
        masks_grp = grp["img_masks"]
        img_pls = grp["img_pls"][()]          # (K,) float64
        n_masks = len(masks_grp)
        H, W = topopaths.h, topopaths.w

        if "size" in grp:
            # ── RLE format (training dataset) ──────────────────────────────
            img_size = grp["size"][()]          # (2,) → [H, W]
            masks_rle = [
                {"size": img_size, "counts": masks_grp[str(i)][()]}
                for i in range(n_masks)
            ]
            enc, _ = topopaths.create_input(img_pls, masks_rle, convertMask=True)
        else:
            # ── Flat pixel-index format (custom dataset) ───────────────────
            # Each masks_grp[str(i)] contains the flat 1-D indices of pixels
            # that belong to object i in the original H×W image.
            masks = []
            for i in range(n_masks):
                flat_indices = masks_grp[str(i)][()]           # (num_pixels,) int64
                m = np.zeros(H * W, dtype=float)
                m[flat_indices] = 1.0
                masks.append(m.reshape(H, W))
            masks = np.stack(masks, axis=0)                    # (K, H, W)
            enc, _ = topopaths.create_input(img_pls, masks, convertMask=False)

    return torch.as_tensor(enc, dtype=torch.float32)           # (dims, H//2, W//2)


def build_goal_from_arrays(
    pixel_indices_list,
    costs,
    topopaths: TopoPaths,
) -> torch.Tensor:
    """
    Build a goal encoding from flat pixel-index lists and cost values.

    Args:
        pixel_indices_list: list of np.ndarray, one per object;
                            each array contains flat pixel indices
                            (row * W + col) in the ORIGINAL H×W image
        costs:              list/array of float, one per object
                            (path-length cost; higher = more salient/target-like)
        topopaths:          TopoPaths instance

    Returns:
        Tensor [dims, H//2, W//2] float32
    """
    H, W = topopaths.h, topopaths.w
    pls = np.array(costs, dtype=np.float64)
    masks = []
    for idx_arr in pixel_indices_list:
        m = np.zeros(H * W, dtype=float)
        m[np.asarray(idx_arr, dtype=int)] = 1.0
        masks.append(m.reshape(H, W))
    masks = np.stack(masks, axis=0)        # (K, H, W)
    enc, _ = topopaths.create_input(pls, masks, convertMask=False)
    return torch.as_tensor(enc, dtype=torch.float32)           # (dims, H//2, W//2)


def build_goal_from_masks_array(
    masks_khw: np.ndarray,
    costs,
    topopaths: TopoPaths,
) -> torch.Tensor:
    """
    Build the goal positional-encoding tensor from dense K-channel binary masks.

    This is the most natural interface when you have a live segmentation model
    that produces K binary masks (one per object class) together with a scalar
    cost/salience score for each channel.

    The function replicates exactly what the training dataloader does when it
    calls TopoPaths.create_input() — so the model sees the same representation
    at inference as it did during training.

    Args:
        masks_khw: np.ndarray, shape (K, H, W), dtype float32 or bool.
                   K = number of object channels (can be any positive integer).
                   H × W = the ORIGINAL image resolution before any downsampling
                   (must match the resolution TopoPaths was configured with;
                   by default that is 120 × 160 — the raw camera resolution).
                   Each slice masks_khw[k] is a binary occupancy map for object k:
                   1.0 where the object is present, 0.0 elsewhere.
        costs:     array-like, shape (K,) — one scalar cost per channel.
                   Semantics follow the training convention:
                     • Higher value  →  object is closer to the navigation target
                     • Outlier value →  object should be ignored (set to 99+)
                   The values are normalised internally by TopoPaths, so the
                   absolute scale only matters for relative ordering.
        topopaths: TopoPaths instance (obtained via _build_topopaths(config) or
                   constructed manually).

    Returns:
        Tensor [dims, H//2, W//2] float32 — positional-encoding costmap ready
        to be batch-unsqueezed and fed directly into the model's GoalEncoder.

    Example::

        import numpy as np
        from inference import _build_topopaths, build_goal_from_masks_array, load_config

        config    = load_config("config/object_react_vln.yaml")
        topopaths = _build_topopaths(config)

        # Suppose your segmentation model gives 3 objects in a 120×160 scene:
        masks = np.zeros((3, 120, 160), dtype=np.float32)
        masks[0, 40:80, 20:60]   = 1.0   # object 0 occupies a region
        masks[1, 60:90, 80:130]  = 1.0   # object 1
        masks[2, 10:30, 100:150] = 1.0   # object 2
        costs = np.array([29.7, 5.1, 0.8])  # object 0 is closest to goal

        goal_enc = build_goal_from_masks_array(masks, costs, topopaths)
        # goal_enc: Tensor [8, 60, 80]
    """
    pls   = np.asarray(costs, dtype=np.float64)       # (K,)
    masks = np.asarray(masks_khw, dtype=float)        # (K, H, W)
    enc, _ = topopaths.create_input(pls, masks, convertMask=False)
    return torch.as_tensor(enc, dtype=torch.float32)  # (dims, H//2, W//2)


# ── config / model helpers ────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load training config, merging defaults.yaml first."""
    defaults_path = os.path.join(SCRIPT_DIR, "config", "defaults.yaml")
    with open(defaults_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config_path, "r") as f:
        config.update(yaml.safe_load(f))
    return config


def build_model(config: dict) -> GNM:
    """Construct the GNM model from config parameters."""
    kwargs = {
        "goal_type":        config.get("goal_type", "image"),
        "obs_type":         config.get("obs_type", "image"),
        "dims":             config.get("dims", 8),
        "predict_dists":    config.get("predict_dists", True),
        "discrete_actions": config.get("discrete_actions", False),
        "use_mask_grad":    config.get("use_mask_grad", False),
        "goal_uses_context":config.get("goal_uses_context", False),
    }
    return GNM(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config.get("learn_angle", False),
        obs_encoding_size=config["obs_encoding_size"],
        goal_encoding_size=config["goal_encoding_size"],
        **kwargs,
    )


def load_checkpoint(
    model: GNM,
    checkpoint_path: str,
    model_type: str,
    device: torch.device = None,
) -> GNM:
    """
    Load weights from a .pth checkpoint into `model`.

    Args:
        model:           GNM instance (already on the target device)
        checkpoint_path: path to latest.pth
        model_type:      "gnm" / "vint" / "nomad"
        device:          torch.device; inferred from model parameters if None
    """
    if device is None:
        device = next(model.parameters()).device
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    load_model(model, model_type, ckpt)
    print(f"[inference] Loaded checkpoint: {checkpoint_path}")
    return model


# ── core prediction ───────────────────────────────────────────────────────────


def predict_from_image_masks_costs(
    model: GNM,
    config: dict,
    image_paths,
    masks_khw: np.ndarray,
    costs,
    device: torch.device = None,
) -> dict:
    """
    High-level single-call API: RGB image(s)  +  k-channel masks  +  k costs
    → discrete action prediction.

    This is the recommended entry-point when you already have a loaded model
    and want to run inference repeatedly on new frames (e.g. inside a ROS node
    or a control loop) without paying the model-loading overhead each time.

    Args:
        model:       GNM instance already loaded and on the correct device.
                     Obtain it with::

                         config = load_config("config/object_react_vln.yaml")
                         model  = build_model(config)
                         model  = load_checkpoint(model, "logs/.../latest.pth", "gnm")
                         model  = model.to(device).eval()

        config:      Config dict (from load_config).

        image_paths: list[str] — one or more image file paths, ordered oldest→newest.
                     The last path is the current frame.  If fewer than
                     (context_size + 1) paths are given, the earliest frame is
                     repeated to fill the window.

        masks_khw:   np.ndarray, shape (K, H, W), dtype float32 or bool.
                     K-channel binary segmentation masks for the GOAL frame.
                     H × W = original camera resolution (default 120 × 160).
                     Each channel masks_khw[k] marks pixels belonging to object k.

        costs:       array-like, shape (K,) — cost/salience per mask channel.
                     Higher → object is more relevant to the navigation goal.

        device:      torch.device to place tensors on.  If None, inferred from
                     model parameters.

    Returns:
        dict with keys:
            next_action      (str)         — most likely first action
            action_sequence  (list[str])   — one name per prediction step
            action_indices   (list[int])   — raw 0–3 index per step
            probabilities    (list[list])  — softmax (4,) per step
            distance_pred    (float|None)  — normalised distance estimate

    Example::

        import numpy as np
        from inference import load_config, build_model, load_checkpoint
        from inference import predict_from_image_masks_costs

        config  = load_config("config/object_react_vln.yaml")
        model   = build_model(config)
        model   = load_checkpoint(model, "logs/object_react_vln/my_run/latest.pth",
                                  model_type="gnm")
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model   = model.to(device).eval()

        # K objects visible from the robot's current position
        K = 4
        masks = np.zeros((K, 120, 160), dtype=np.float32)
        masks[0, 30:70, 10:50]   = 1.0   # chair
        masks[1, 50:90, 80:130]  = 1.0   # sofa
        masks[2, 10:40, 110:150] = 1.0   # plant
        masks[3, 70:110, 40:80]  = 1.0   # tv_monitor
        costs = np.array([29.7, 5.1, 0.8, 199.5])  # tv_monitor is the target

        result = predict_from_image_masks_costs(
            model=model, config=config,
            image_paths=["/data/ep_21909/images/00042.png"],
            masks_khw=masks,
            costs=costs,
            device=device,
        )
        print(result["next_action"])          # e.g. "FORWARD"
        print(result["action_sequence"])      # ['FORWARD', 'LEFT', ...]
    """
    if device is None:
        device = next(model.parameters()).device

    image_size   = tuple(config["image_size"])
    context_size = config["context_size"]
    obs_tensor   = build_obs_tensor(image_paths, image_size, context_size, device)

    topopaths  = _build_topopaths(config)
    goal_enc   = build_goal_from_masks_array(masks_khw, costs, topopaths)
    goal_tensor = goal_enc.unsqueeze(0).to(device)

    return predict(model, obs_tensor, goal_tensor)


@torch.no_grad()
def predict(
    model: GNM,
    obs_image: torch.Tensor,
    goal_image: torch.Tensor,
) -> dict:
    """
    Run a single forward pass and return human-readable results.

    Args:
        obs_image:  (1, 3*(context_size+1), H, W) on the model's device
        goal_image: (1, dims, H//2, W//2) on the model's device

    Returns dict:
        next_action      : str  — most likely first action name
        action_sequence  : list[str] — predicted action name per step
        action_indices   : list[int] — raw index (0–3) per step
        probabilities    : list[list[float]] — softmax prob (4,) per step
        distance_pred    : float | None — predicted goal distance (norm.)
    """
    model.eval()
    dist_pred, action_logits = model(obs_image, goal_image)
    # action_logits: (1, len_traj_pred, 4)
    probs   = torch.softmax(action_logits, dim=-1).squeeze(0)   # (T, 4)
    indices = probs.argmax(dim=-1).tolist()                      # (T,)
    names   = [ACTION_NAMES[i] for i in indices]

    return {
        "next_action":     names[0],
        "action_sequence": names,
        "action_indices":  indices,
        "probabilities":   probs.cpu().tolist(),
        "distance_pred":   float(dist_pred.squeeze()) if dist_pred is not None else None,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Object-React VLN inference — predicts discrete navigation actions."
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to training config YAML (e.g. config/object_react_vln.yaml)",
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint .pth (e.g. logs/object_react_vln/<run>/latest.pth)",
    )
    p.add_argument(
        "--image", nargs="+", required=True, metavar="IMG",
        help=(
            "One or more RGB image paths, ordered oldest → newest "
            "(the last is the current frame). If fewer than context_size+1 "
            "images are provided the earliest is repeated to fill the window."
        ),
    )

    goal_grp = p.add_mutually_exclusive_group(required=True)
    goal_grp.add_argument(
        "--costmap_h5", metavar="H5",
        help="Path to pre-built costmap H5 file (requires --traj_name).",
    )
    goal_grp.add_argument(
        "--masks_npy", metavar="NPY",
        help=(
            "Path to a .npy file containing k-channel binary segmentation masks, "
            "shape (K, H, W), dtype float32 or bool. "
            "Must be paired with --costs."
        ),
    )
    goal_grp.add_argument(
        "--no_goal", action="store_true",
        help="Zero out the goal encoding (useful for testing obs-only behaviour).",
    )

    # H5 options
    p.add_argument(
        "--traj_name", default=None,
        help="Trajectory name key in the H5 file (required with --costmap_h5).",
    )
    p.add_argument(
        "--frame_idx", type=int, default=0,
        help="Goal frame index inside the H5 entry (default: 0).",
    )

    # Masks-array options
    p.add_argument(
        "--costs", nargs="+", type=float, metavar="COST", default=None,
        help=(
            "K space-separated float cost values, one per mask channel "
            "(required with --masks_npy). "
            "Higher value = object is more salient / closer to the navigation target."
        ),
    )

    p.add_argument(
        "--device", default="auto",
        help="'cpu', 'cuda', or 'auto' (default: auto-detect).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[inference] Device: {device}")

    # ── config ────────────────────────────────────────────────────────────────
    config = load_config(args.config)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(config).to(device)
    model = load_checkpoint(model, args.checkpoint, config["model_type"], device)
    model.eval()

    # ── observation tensor ────────────────────────────────────────────────────
    image_size   = tuple(config["image_size"])   # (W, H)
    context_size = config["context_size"]
    obs_tensor   = build_obs_tensor(args.image, image_size, context_size, device)
    print(f"[inference] obs tensor shape: {list(obs_tensor.shape)}")

    # ── goal tensor ───────────────────────────────────────────────────────────
    dims = config.get("dims", 8)

    if args.no_goal:
        goal_tensor = torch.zeros(1, dims, 60, 80, device=device)
        print("[inference] Goal: (zeroed out)")

    elif args.masks_npy is not None:
        # ── k-channel masks + costs ──────────────────────────────────────────
        if args.costs is None:
            raise ValueError("--costs is required when using --masks_npy.")
        masks_khw = np.load(args.masks_npy).astype(np.float32)  # (K, H, W)
        costs     = np.array(args.costs, dtype=np.float64)
        if masks_khw.ndim != 3:
            raise ValueError(f"masks_npy must be a 3-D array (K, H, W); "
                             f"got shape {masks_khw.shape}")
        if len(costs) != masks_khw.shape[0]:
            raise ValueError(f"Number of --costs ({len(costs)}) must equal the number "
                             f"of mask channels ({masks_khw.shape[0]}).")
        print(f"[inference] Masks shape: {list(masks_khw.shape)}  "
              f"Costs: {costs.tolist()}")
        topopaths  = _build_topopaths(config)
        goal_enc   = build_goal_from_masks_array(masks_khw, costs, topopaths)
        goal_tensor = goal_enc.unsqueeze(0).to(device)
        print(f"[inference] Goal tensor shape: {list(goal_tensor.shape)}")

    else:
        # ── H5 costmap file ───────────────────────────────────────────────────
        if args.costmap_h5 is None or args.traj_name is None:
            raise ValueError("Both --costmap_h5 and --traj_name are required "
                             "when not using --masks_npy or --no_goal.")
        topopaths  = _build_topopaths(config)
        goal_enc   = load_goal_from_h5(
            args.costmap_h5, args.traj_name, args.frame_idx, topopaths
        )                                           # (dims, H//2, W//2)
        goal_tensor = goal_enc.unsqueeze(0).to(device)   # (1, dims, H//2, W//2)
        print(f"[inference] Goal tensor shape: {list(goal_tensor.shape)}")

    # ── predict ───────────────────────────────────────────────────────────────
    result = predict(model, obs_tensor, goal_tensor)

    print("\n" + "=" * 60)
    print(f"  Next action     : {result['next_action']}")
    print(f"  Full sequence   : {' → '.join(result['action_sequence'])}")
    print("  Per-step probabilities (STOP / FWD / LEFT / RIGHT):")
    for t, probs in enumerate(result["probabilities"]):
        bar = "  ".join(f"{p:.3f}" for p in probs)
        chosen = ACTION_NAMES[result["action_indices"][t]]
        print(f"    step {t:2d}: [{bar}]  → {chosen}")
    if result["distance_pred"] is not None:
        print(f"  Distance pred   : {result['distance_pred']:.4f}")
    print("=" * 60 + "\n")

    # also print machine-readable JSON
    print(json.dumps(result, indent=2))
    return result


# ── allow import-level use as a library ──────────────────────────────────────

def run_inference(
    config_path: str,
    checkpoint_path: str,
    image_paths,
    costmap_h5: str = None,
    traj_name: str = None,
    frame_idx: int = 0,
    masks_khw: np.ndarray = None,
    costs=None,
    pixel_indices_list=None,
    device_str: str = "auto",
) -> dict:
    """
    Programmatic API — loads model + config and runs a single inference call.

    Goal input — provide EXACTLY ONE of:
      1. costmap_h5 + traj_name [+ frame_idx]   — read from pre-built H5 file
      2. masks_khw + costs                       — dense (K,H,W) masks + K costs
      3. pixel_indices_list + costs              — flat pixel-index lists + K costs
      (none of the above)                        — zero goal encoding

    .. note::
        If you are calling this in a loop, prefer
        :func:`predict_from_image_masks_costs` instead — it reuses the already-
        loaded model, avoiding the checkpoint-loading overhead on every call.

    Args:
        config_path:        path to YAML config (e.g. "config/object_react_vln.yaml")
        checkpoint_path:    path to .pth checkpoint
        image_paths:        list[str] — RGB frame paths, oldest → newest
        costmap_h5:         path to H5 costmap file (option 1)
        traj_name:          trajectory key in the H5 file (option 1)
        frame_idx:          frame index inside the H5 entry, default 0 (option 1)
        masks_khw:          np.ndarray (K, H, W) binary masks (option 2)
        costs:              array-like (K,) cost per mask/object (options 2 & 3)
        pixel_indices_list: list of flat pixel-index arrays (option 3)
        device_str:         "cpu" / "cuda" / "auto"

    Returns:
        dict — same keys as :func:`predict`:
        ``next_action``, ``action_sequence``, ``action_indices``,
        ``probabilities``, ``distance_pred``

    Examples::

        from inference import run_inference
        import numpy as np

        # Option 1 — H5 file
        result = run_inference(
            config_path     = "config/object_react_vln.yaml",
            checkpoint_path = "logs/object_react_vln/my_run/latest.pth",
            image_paths     = ["frame_0.png"],
            costmap_h5      = "../custom_dataset/costmaps.h5",
            traj_name       = "ep_21909",
            frame_idx       = 42,
        )
        print(result["next_action"])   # e.g. "FORWARD"

        # Option 2 — live k-channel masks
        K = 3
        masks = np.zeros((K, 120, 160), dtype=np.float32)
        masks[0, 30:70, 10:50] = 1.0   # object 0
        masks[1, 50:90, 80:130] = 1.0  # object 1
        masks[2, 10:40, 110:150] = 1.0 # object 2
        result = run_inference(
            config_path     = "config/object_react_vln.yaml",
            checkpoint_path = "logs/object_react_vln/my_run/latest.pth",
            image_paths     = ["frame_0.png"],
            masks_khw       = masks,
            costs           = [29.7, 5.1, 0.8],
        )
        print(result["next_action"])   # e.g. "LEFT"
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    config = load_config(config_path)
    model  = build_model(config).to(device)
    model  = load_checkpoint(model, checkpoint_path, config["model_type"], device)
    model.eval()

    image_size   = tuple(config["image_size"])
    context_size = config["context_size"]
    obs_tensor   = build_obs_tensor(image_paths, image_size, context_size, device)

    dims = config.get("dims", 8)
    topopaths = _build_topopaths(config)

    if costmap_h5 is not None and traj_name is not None:
        goal_enc = load_goal_from_h5(costmap_h5, traj_name, frame_idx, topopaths)
        goal_tensor = goal_enc.unsqueeze(0).to(device)
    elif masks_khw is not None and costs is not None:
        goal_enc = build_goal_from_masks_array(masks_khw, costs, topopaths)
        goal_tensor = goal_enc.unsqueeze(0).to(device)
    elif pixel_indices_list is not None and costs is not None:
        goal_enc = build_goal_from_arrays(pixel_indices_list, costs, topopaths)
        goal_tensor = goal_enc.unsqueeze(0).to(device)
    else:
        goal_tensor = torch.zeros(1, dims, 60, 80, device=device)

    return predict(model, obs_tensor, goal_tensor)


if __name__ == "__main__":
    main()

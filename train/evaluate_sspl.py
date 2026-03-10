#!/usr/bin/env python3
"""
evaluate_sspl.py  —  Closed-loop Success Rate and SSPL evaluation for
                     Object-React continuous (v, w) models.

For each test trajectory the robot is simulated step-by-step:
  1. At each virtual time-step the nearest ground-truth frame is found by 2-D
     position proximity.
  2. The costmap at that frame is fed to the model as the "goal" input (as
     during training: goal_type="image_mask_enc", obs_type="disabled").
  3. The model outputs (v, w); the virtual robot state is advanced:
         pos  +=  v * [cos(yaw), sin(yaw)]  (world frame)
         yaw  +=  w
  4. Simulation ends when the robot is within ``success_threshold`` metres of
     the episode goal, or after ``max_steps`` steps.

Metrics
-------
  SR    – fraction of episodes that succeed (reach within threshold of goal).
  SSPL  – mean(S_i * L_i / max(P_i, L_i))  where
            S_i = success indicator (0/1),
            L_i = Euclidean start→goal straight-line distance (shortest path),
            P_i = total simulated path length.

Usage (from the train/ directory)
----------------------------------
    python evaluate_sspl.py \\
        --config     config/object_react_vw.yaml \\
        --checkpoint logs/object_react_vw/<run_name>/latest.pth \\
        [--split     test]        \\
        [--threshold 1.0]         \\
        [--max_steps 500]         \\
        [--step_scale 1.0]        \\
        [--verbose]
"""

import os
import sys
import argparse
import pickle
import io
import lmdb
import math

import numpy as np
import yaml
import torch
import torchvision.transforms as T

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from vint_train.models.gnm.gnm import GNM
from vint_train.training.train_eval_loop import load_model
from vint_train.models.object_react.dataloader import TopoPaths
from vint_train.data.data_utils import img_path_to_data


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg: dict) -> GNM:
    kwargs = {
        "predict_dists":        cfg.get("predict_dists", True),
        "precomputed_filename": cfg.get("precomputed_filename", None),
        "pl_perturb_ratio":     cfg.get("pl_perturb_ratio", 0.0),
        "pl_perturb_type":      cfg.get("pl_perturb_type", "max_val"),
        "mask_crop_ratio":      cfg.get("mask_crop_ratio", 1.0),
        "use_mask_grad":        cfg.get("use_mask_grad", False),
        "goal_type":            cfg.get("goal_type", "image"),
        "obs_type":             cfg.get("obs_type", "image"),
        "dims":                 cfg.get("dims", None),
        "goal_uses_context":    cfg.get("goal_uses_context", False),
        "discrete_actions":     False,
        "output_vw":            True,
    }
    model = GNM(
        context_size=cfg["context_size"],
        len_traj_pred=cfg["len_traj_pred"],
        learn_angle=False,
        obs_encoding_size=cfg["obs_encoding_size"],
        goal_encoding_size=cfg["goal_encoding_size"],
        **kwargs,
    )
    return model


def load_checkpoint(model: GNM, checkpoint_path: str) -> GNM:
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    load_model(model, "gnm", ckpt)
    return model


def _angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


# ── per-trajectory simulation ─────────────────────────────────────────────────

class TrajectorySimulator:
    """
    Simulate a robot along a recorded trajectory using model-predicted (v, w).

    The episode goal is the last frame's 2-D position.
    """

    def __init__(
        self,
        cfg: dict,
        model: GNM,
        topopaths: TopoPaths,
        transform,
        device: torch.device,
        success_threshold: float = 1.0,
        max_steps: int = 500,
        step_scale: float = 1.0,
        verbose: bool = False,
    ):
        self.cfg = cfg
        self.model = model
        self.topopaths = topopaths
        self.transform = transform
        self.device = device
        self.success_threshold = success_threshold
        self.max_steps = max_steps
        self.step_scale = step_scale   # multiplier to un-normalise v/w
        self.verbose = verbose

        # Retrieve normalisation scale from data_config.yaml
        import yaml as _yaml
        data_config_path = os.path.join(
            os.path.dirname(__file__), "vint_train", "data", "data_config.yaml"
        )
        with open(data_config_path) as f:
            all_dc = _yaml.safe_load(f)

        # Pick the first dataset in the config for metric spacing
        ds_name = list(cfg["datasets"].keys())[0]
        dc = all_dc.get(ds_name, {})
        waypoint_spacing = cfg["datasets"][ds_name].get("waypoint_spacing", 1)
        self.metric_scale = dc.get("metric_waypoint_spacing", 0.20) * waypoint_spacing

    def _get_goal_tensor(self, traj_name: str, frame_idx: int) -> torch.Tensor:
        """Return the (image_mask_enc) goal tensor for a given frame."""
        # get_topo_path returns (img_enc, plWtColorImg); we only need img_enc
        img_enc, _vis = self.topopaths.get_topo_path(traj_name, frame_idx)
        # img_enc shape: (dims, H//2, W//2)
        goal_tensor = torch.as_tensor(img_enc, dtype=torch.float32).unsqueeze(0)
        return goal_tensor.to(self.device)

    def _nearest_frame(self, pos: np.ndarray, gt_positions: np.ndarray) -> int:
        dists = np.linalg.norm(gt_positions - pos, axis=1)
        return int(np.argmin(dists))

    def run(
        self,
        traj_name: str,
        gt_positions: np.ndarray,
        gt_yaw: np.ndarray,
        teacher_forcing: bool = False,
    ) -> dict:
        """
        Run one episode.

        Args:
            teacher_forcing: if True, always use GT frame t's costmap at step t
                             (open-loop integration from start). This gives a
                             cleaner measure of whether predicted (v,w) actions
                             reproduce the GT trajectory well.

        Returns dict with keys: success, path_length, shortest_path, spl.
        """
        N = len(gt_positions)
        goal_pos = gt_positions[-1].copy()
        start_pos = gt_positions[0].copy()

        shortest_path = float(np.linalg.norm(goal_pos - start_pos))

        # Virtual robot state
        pos = start_pos.copy().astype(float)
        yaw = float(gt_yaw[0])
        path_length = 0.0
        success = False

        max_t = N if teacher_forcing else self.max_steps

        for step in range(max_t):
            # Check goal reached
            if np.linalg.norm(pos - goal_pos) < self.success_threshold:
                success = True
                break

            if teacher_forcing:
                # Use the GT frame at this step as costmap input
                frame_idx = min(step, N - 1)
            else:
                # Find nearest GT frame for costmap lookup
                frame_idx = self._nearest_frame(pos, gt_positions)
                frame_idx = min(frame_idx, N - 1)

            # Get goal (costmap) tensor
            goal_tensor = self._get_goal_tensor(traj_name, frame_idx)

            # Observation is zeros for obs_type="disabled" (ignored by model)
            obs_tensor = torch.zeros(1, 3, 64, 85, device=self.device)

            with torch.no_grad():
                _, action_pred = self.model(obs_tensor, goal_tensor)

            # action_pred: (1, T, 2)  [normalised_v, normalised_w]
            v_norm = float(action_pred[0, 0, 0].cpu())
            w_norm = float(action_pred[0, 0, 1].cpu())

            # Denormalise
            v = v_norm * self.metric_scale * self.step_scale   # metres
            w = w_norm * math.pi * self.step_scale             # radians

            # Advance robot
            dx = v * math.cos(yaw)
            dy = v * math.sin(yaw)
            pos = pos + np.array([dx, dy])
            yaw = _angle_wrap(yaw + w)
            path_length += abs(v)

            if self.verbose and step % 20 == 0:
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                mode_str = "[TF]" if teacher_forcing else "[CL]"
                print(
                    f"  {mode_str} step {step:4d}  pos=({pos[0]:.2f},{pos[1]:.2f})"
                    f"  yaw={math.degrees(yaw):.1f}°"
                    f"  v={v:.3f}  w={math.degrees(w):.1f}°"
                    f"  d_goal={dist_to_goal:.2f}m  frame={frame_idx}"
                )

        d_final = float(np.linalg.norm(pos - goal_pos))
        progress = max(0.0, 1.0 - d_final / (shortest_path + 1e-6))

        # Hard SPL (existing)
        spl = 0.0
        if success:
            spl = shortest_path / max(shortest_path, path_length + 1e-6)

        # Soft-SSPL using progress
        soft_spl = progress * shortest_path / max(shortest_path, path_length + 1e-6)

        return {
            "success": success,
            "path_length": path_length,
            "shortest_path": shortest_path,
            "d_final": d_final,
            "progress": progress,
            "spl": spl,
            "soft_spl": soft_spl,
        }


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_sspl(
    cfg: dict,
    model: GNM,
    split: str = "test",
    success_threshold: float = 1.0,
    max_steps: int = 500,
    step_scale: float = 1.0,
    verbose: bool = False,
    teacher_forcing: bool = False,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Run the full SSPL / SR evaluation over all trajectories in ``split``.

    Returns:
        dict with keys: success_rate, sspl, num_episodes, per_episode_results.
    """
    transform = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build TopoPaths for costmap lookup
    topopaths = TopoPaths(
        dims=cfg.get("dims", 8),
        precomputed_filename=cfg["precomputed_filename"],
        pl_perturb_ratio=0.0,   # no perturbation during eval
        pl_perturb_type="max_val",
        mask_crop_ratio=1.0,
    )

    simulator = TrajectorySimulator(
        cfg=cfg,
        model=model,
        topopaths=topopaths,
        transform=transform,
        device=device,
        success_threshold=success_threshold,
        max_steps=max_steps,
        step_scale=step_scale,
        verbose=verbose,
    )

    results = []
    total_episodes = 0
    total_success = 0
    total_spl = 0.0

    for ds_name, ds_cfg in cfg["datasets"].items():
        split_dir = ds_cfg.get(split)
        if split_dir is None:
            print(f"[WARN] No '{split}' split for dataset '{ds_name}'. Skipping.")
            continue

        data_folder = ds_cfg.get("data_folder", split_dir)

        traj_names_file = os.path.join(split_dir, "traj_names.txt")
        if not os.path.exists(traj_names_file):
            print(f"[WARN] {traj_names_file} not found. Skipping.")
            continue

        with open(traj_names_file) as f:
            traj_names = [l.strip() for l in f if l.strip()]

        print(f"\n[{ds_name}/{split}] {len(traj_names)} trajectories")

        for traj_name in traj_names:
            traj_pkl = os.path.join(data_folder, traj_name, "traj_data.pkl")
            if not os.path.exists(traj_pkl):
                print(f"  [skip] {traj_name}: no traj_data.pkl")
                continue

            with open(traj_pkl, "rb") as f:
                traj_data = pickle.load(f)

            gt_positions = traj_data["position"].astype(float)
            gt_yaw = traj_data["yaw"].astype(float)

            if len(gt_positions) < 2:
                continue

            if verbose:
                print(f"\n  Episode: {traj_name}  (T={len(gt_positions)})")

            res = simulator.run(traj_name, gt_positions, gt_yaw,
                               teacher_forcing=teacher_forcing)

            total_episodes += 1
            total_success += int(res["success"])
            total_spl += res["spl"]
            results.append({"traj": traj_name, **res})

            print(
                f"  {traj_name:40s}  success={res['success']}  "
                f"L={res['shortest_path']:.2f}m  P={res['path_length']:.2f}m  "
                f"SPL={res['spl']:.3f} "
                f"SSPL= {res['soft_spl']:.3f}"
            )

    sr   = total_success / max(total_episodes, 1)
    sspl = total_spl    / max(total_episodes, 1)

    print("\n" + "=" * 60)
    print(f"Results on '{split}' split")
    print(f"  Num episodes : {total_episodes}")
    print(f"  Success Rate : {sr:.4f}  ({total_success}/{total_episodes})")
    print(f"  SSPL         : {sspl:.4f}")
    print("=" * 60)

    return {
        "success_rate": sr,
        "sspl": sspl,
        "num_episodes": total_episodes,
        "per_episode_results": results,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an Object-React (v,w) model with SR and SSPL."
    )
    parser.add_argument("--config",     "-c", required=True,
                        help="Path to config YAML (e.g. config/object_react_vw.yaml)")
    parser.add_argument("--checkpoint", "-k", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--split",      default="test",
                        help="Dataset split to evaluate ('train' or 'test')")
    parser.add_argument("--threshold",  type=float, default=3.0,
                        help="Success distance threshold in metres (default: 1.0)")
    parser.add_argument("--teacher_forcing", action="store_true",
                        help="Use GT frame costmaps at each step (open-loop integration)")
    parser.add_argument("--max_steps",  type=int,   default=500,
                        help="Maximum simulation steps per episode (default: 500)")
    parser.add_argument("--step_scale", type=float, default=1.0,
                        help="Multiplier applied to v and w after denormalisation")
    parser.add_argument("--gpu",        type=int,   default=0,
                        help="GPU id to use (-1 for CPU)") 
    parser.add_argument("--verbose",    action="store_true",
                        help="Print per-step debug output")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override costmap path to be relative to this script's directory if needed
    if not os.path.isabs(cfg.get("precomputed_filename", "")):
        cfg["precomputed_filename"] = os.path.join(
            SCRIPT_DIR, cfg["precomputed_filename"]
        )
    # Also fix dataset paths
    for ds_name in cfg.get("datasets", {}):
        for key in ("data_folder", "train", "test"):
            val = cfg["datasets"][ds_name].get(key)
            if val and not os.path.isabs(val):
                cfg["datasets"][ds_name][key] = os.path.join(SCRIPT_DIR, val)

    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    model = build_model(cfg)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    results = evaluate_sspl(
        cfg=cfg,
        model=model,
        split=args.split,
        success_threshold=args.threshold,
        max_steps=args.max_steps,
        step_scale=args.step_scale,
        verbose=args.verbose,
        teacher_forcing=args.teacher_forcing,
        device=device,
    )

    # Optionally save results as JSON
    import json
    out_path = os.path.join(
        os.path.dirname(args.checkpoint), f"sspl_results_{args.split}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "per_episode_results"},
            f, indent=2,
        )
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()

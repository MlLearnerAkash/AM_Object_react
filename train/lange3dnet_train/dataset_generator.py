"""
process_frame.py
----------------
Two modes of operation:

1. Offline post-processing  – process_frame()
   Reads a saved norm_frame image + semantic .npy file and writes
   masks.npy / class_ids.npy / geodesic_distances.npy into an output dir.

2. Online collection  – episode_generator() + __main__
   Drives the Habitat simulator along reference paths, collects frames,
   and writes the full dataset in the expected structure:

   data_root/
       train.txt                    episode IDs, one per line
       episode_<id>/
           instruction.txt
           frame_000/
               rgb.png
               masks.npy            [K, H, W] uint8
               class_ids.npy        [K]       int64
               geodesic_distances.npy [K]     float32  normalised to [0,1]
           frame_001/
               ...

Outputs
-------
masks.npy              [K, H, W]  uint8  binary masks (1 = object pixel)
class_ids.npy          [K]        int64  semantic class IDs
geodesic_distances.npy [K]        float32 normalised geodesic distances [0,1]
                                   (0 = closest to goal, 1 = farthest)
"""

import os
import shutil
import time
import sys

import numpy as np
import tqdm
from PIL import Image

# ---------------------------------------------------------------------------
# Offline helper: build per-frame labels from saved files
# ---------------------------------------------------------------------------

def process_frame(
    norm_frame_path: str,
    semantic_path: str,
    *,
    exclude_background: bool = True,
    output_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build masks, class IDs and normalised geodesic distances for one frame.

    Parameters
    ----------
    norm_frame_path : str
        Path to the norm frame image (grayscale, uint8).
    semantic_path : str
        Path to the semantic segmentation .npy file (H, W) int32.
    exclude_background : bool
        If True (default) class IDs ≤ 0 are treated as background and excluded.
    output_dir : str
        Directory in which to write masks.npy, class_ids.npy and
        geodesic_distances.npy.

    Returns
    -------
    masks : np.ndarray  [K, H, W]  uint8
    class_ids : np.ndarray  [K]  int64
    geodesic_distances : np.ndarray  [K]  float32  (mean pixel value in norm frame)
    """
    norm = np.array(Image.open(norm_frame_path), dtype=np.float32) / 255.0  # [0,1]
    semantic = np.load(semantic_path)                                         # (H, W)
    H, W = semantic.shape
    assert norm.shape == (H, W), (
        f"Shape mismatch: norm_frame {norm.shape} vs semantic {(H, W)}"
    )

    unique_ids = np.unique(semantic)
    if exclude_background:
        unique_ids = unique_ids[unique_ids > 0]

    K = len(unique_ids)
    masks = np.stack(
        [(semantic == cid).astype(np.uint8) for cid in unique_ids], axis=0
    )                                        # [K, H, W]
    class_ids = unique_ids.astype(np.int64)  # [K]

    geodesic_distances = np.array(
        [norm[masks[k].astype(bool)].mean() if masks[k].any() else 0.0
         for k in range(K)],
        dtype=np.float32,
    )

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "masks.npy"), masks)
    np.save(os.path.join(output_dir, "class_ids.npy"), class_ids)
    np.save(os.path.join(output_dir, "geodesic_distances.npy"), geodesic_distances)

    return masks, class_ids, geodesic_distances


# ---------------------------------------------------------------------------
# Online helper: save a single frame captured from the simulator
# ---------------------------------------------------------------------------

def _save_frame_data(
    rgb: np.ndarray,
    semantic: np.ndarray,
    distances: dict,
    output_dir: str,
) -> None:
    """
    Persist one frame's data in the canonical directory layout.

    Parameters
    ----------
    rgb : (H, W, 3) uint8
    semantic : (H, W) int32   – object instance IDs (-1 / 0 = background)
    distances : dict  {object_id: float | None}  raw geodesic distances in metres
    output_dir : str  target directory (created if absent)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── RGB ──────────────────────────────────────────────────────────────────
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(output_dir, "rgb.png"))

    # ── Masks & class IDs ────────────────────────────────────────────────────
    unique_ids = np.array(
        [int(uid) for uid in np.unique(semantic) if uid > 0], dtype=np.int64
    )
    if unique_ids.size == 0:
        H, W = semantic.shape
        np.save(os.path.join(output_dir, "masks.npy"),
                np.zeros((0, H, W), dtype=np.uint8))
        np.save(os.path.join(output_dir, "class_ids.npy"),
                np.zeros(0, dtype=np.int64))
        np.save(os.path.join(output_dir, "geodesic_distances.npy"),
                np.zeros(0, dtype=np.float32))
        return

    masks = np.stack(
        [(semantic == uid).astype(np.uint8) for uid in unique_ids], axis=0
    )  # [K, H, W]

    # ── Geodesic distances: normalise raw metric values to [0, 1] ──────────
    # Missing / inf distances are clamped to 55 m (same cap as semanitc_handler)
    raw = np.array(
        [float(distances.get(int(uid)) or 55.0) for uid in unique_ids],
        dtype=np.float32,
    )
    d_min, d_max = float(raw.min()), float(raw.max())
    denom = d_max - d_min if d_max > d_min else 1.0
    geodesic_distances = ((raw - d_min) / denom).astype(np.float32)

    np.save(os.path.join(output_dir, "masks.npy"), masks)
    np.save(os.path.join(output_dir, "class_ids.npy"), unique_ids)
    np.save(os.path.join(output_dir, "geodesic_distances.npy"), geodesic_distances)


# ---------------------------------------------------------------------------
# Episode generator
# ---------------------------------------------------------------------------

def episode_generator(env, num_episodes: int = 10):
    """
    Drive *env* along each episode's reference path and yield per-episode data.

    Yields
    ------
    dict with keys:
        episode_id   : str
        instruction  : str
        frames       : list of (H, W, 3) uint8  RGB images
        distances    : list of dict {object_id: float}  raw geodesic distances
        semantics    : list of (H, W) int32  semantic frames
    """
    from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
    from reference_path_follower_utils.semanitc_handler import get_object_geodesic_distances

    follower = ShortestPathFollowerCompat(
        env._env.sim, goal_radius=0.5, return_one_hot=False
    )
    follower.mode = "geodesic_path"

    with tqdm.tqdm(total=num_episodes, desc="Collecting episodes") as pbar:
        for _ in range(num_episodes):
            obs = env.reset()
            episode_id    = env.current_episode.episode_id
            reference_path = env.current_episode.reference_path
            instruction   = env.current_episode.instruction.instruction_text

            frames    = []
            all_dists = []
            semantics = []

            for point in reference_path:
                while not env._env.episode_over:
                    best_action = follower.get_next_action(point)
                    if best_action is None:
                        break

                    obs, _, done, _ = env.step(best_action)

                    rgb      = obs["rgb"]
                    semantic = obs["semantic"]
                    dists    = get_object_geodesic_distances(env, semantic)

                    frames.append(rgb)
                    all_dists.append(dists)
                    semantics.append(semantic)

                    if done:
                        break

                if env._env.episode_over:
                    break

            yield {
                "episode_id": episode_id,
                "instruction": instruction,
                "frames":      frames,
                "distances":   all_dists,
                "semantics":   semantics,
            }
            pbar.update()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _run_offline(episode_dir: str) -> None:
    """Post-process a single already-collected episode directory."""
    sem_dir = os.path.join(episode_dir, "semantics")
    indices = sorted(
        int(f.replace("semantic_", "").replace(".npy", ""))
        for f in os.listdir(sem_dir)
        if f.endswith(".npy")
    )

    t0 = time.time()
    for idx in indices:
        norm_path = os.path.join(episode_dir, "norm_frames", f"norm_frame_{idx}.png")
        sem_path  = os.path.join(sem_dir, f"semantic_{idx}.npy")
        rgb_src   = os.path.join(episode_dir, "frames", f"frame_{idx}.png")
        out_dir   = os.path.join(episode_dir, f"frame_{idx:03d}")

        masks, class_ids, geo_dists = process_frame(
            norm_path, sem_path, output_dir=out_dir
        )
        shutil.copy(rgb_src, os.path.join(out_dir, "rgb.png"))
        print(f"  frame {idx:03d}  masks={masks.shape}  saved → {out_dir}/")

    print(f"\nAll {len(indices)} frames done in {time.time() - t0:.2f}s")


def _run_online(
    config_path: str,
    data_root: str,
    num_episodes: int,
    split: str = "train",
) -> None:
    """
    Collect episodes from the simulator and write the full dataset.

    Parameters
    ----------
    config_path  : path to Habitat YAML config
    data_root    : root output directory
    num_episodes : number of episodes to collect
    split        : "train" or "val" – determines the split file written
    """
    sys.path.append("../..")
    from vlnce_baselines.common.environments import VLNCEDaggerEnv
    from vlnce_baselines.config.default import get_config

    config = get_config(config_paths=config_path)
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS = []
    config.freeze()

    os.makedirs(data_root, exist_ok=True)
    split_file = os.path.join(data_root, f"{split}.txt")

    env = VLNCEDaggerEnv(config=config)
    try:
        with open(split_file, "w") as fh_split:
            for ep_data in episode_generator(env, num_episodes=num_episodes):
                episode_id = ep_data["episode_id"]
                ep_dir = os.path.join(data_root, f"episode_{episode_id}")
                os.makedirs(ep_dir, exist_ok=True)

                # instruction.txt
                with open(os.path.join(ep_dir, "instruction.txt"), "w") as fh:
                    fh.write(ep_data["instruction"])

                # per-frame files
                for frame_idx, (rgb, dists, semantic) in enumerate(
                    zip(ep_data["frames"], ep_data["distances"], ep_data["semantics"])
                ):
                    frame_dir = os.path.join(ep_dir, f"frame_{frame_idx:03d}")
                    _save_frame_data(rgb, semantic, dists, frame_dir)

                fh_split.write(f"{episode_id}\n")
                print(
                    f"  episode {episode_id}  {len(ep_data['frames'])} frames"
                    f"  → {ep_dir}/"
                )
    finally:
        env.close()

    print(f"\nDataset written to {data_root}  ({split} split: {split_file})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build LangGeoNet dataset frames")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── offline mode ──────────────────────────────────────────────────────────
    p_off = subparsers.add_parser(
        "offline",
        help="Post-process a single already-collected episode directory",
    )
    p_off.add_argument("episode_dir", help="Path to episode directory")

    # ── online mode ───────────────────────────────────────────────────────────
    p_on = subparsers.add_parser(
        "online",
        help="Collect episodes from the Habitat simulator and save dataset",
    )
    p_on.add_argument(
        "--config",
        default="/data/ws/VLN-CE/vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml",
        help="Path to Habitat YAML config",
    )
    p_on.add_argument(
        "--data_root",
        default="/data/ws/data_langgeonet",
        help="Root output directory for the dataset",
    )
    p_on.add_argument(
        "--num_episodes", type=int, default=1,
        help="Number of episodes to collect",
    )
    p_on.add_argument(
        "--split", default="train", choices=["train", "val"],
        help="Dataset split (writes train.txt or val.txt)",
    )

    args = parser.parse_args()

    if args.mode == "offline":
        _run_offline(args.episode_dir)
    else:
        _run_online(
            config_path=args.config,
            data_root=args.data_root,
            num_episodes=args.num_episodes,
            split=args.split,
        )

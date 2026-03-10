"""
Create a small sample VLN-CE dataset for quick training tests.

Each episode has:
  - images/  with numbered PNG frames
  - traj_data.pkl  with keys ``position`` (N,2), ``yaw`` (N,), ``action`` (N,)

The ``action`` array holds VLN-CE ground-truth discrete actions:
    0 = STOP, 1 = FORWARD, 2 = LEFT, 3 = RIGHT

Usage:
    python create_sample_dataset_vln.py [--force]
"""

import argparse
import os
import pickle
import shutil

import numpy as np
from PIL import Image

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "custom_dataset",
)
DATASET_ROOT = os.path.normpath(DATASET_ROOT)

# VLN-CE action codes
VLN_STOP = 0
VLN_FORWARD = 1
VLN_LEFT = 2
VLN_RIGHT = 3

NUM_TRAIN_EPISODES = 4
NUM_TEST_EPISODES = 2
TRAJ_LEN = 40
IMAGE_SIZE = (160, 120)  # W, H


def _make_episode(ep_dir: str, traj_len: int, rng: np.random.Generator) -> None:
    """Create one episode directory with images and traj_data.pkl."""
    img_dir = os.path.join(ep_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Random walk: mostly forward with occasional turns and stops
    actions = np.zeros(traj_len, dtype=np.int64)
    position = np.zeros((traj_len, 2), dtype=np.float64)
    yaw = np.zeros(traj_len, dtype=np.float64)

    x, y, theta = 0.0, 0.0, 0.0
    step_size = 0.2  # metres per forward step
    turn_angle = 0.3  # radians per turn

    for t in range(traj_len):
        # Choose a random action (weighted towards forward)
        p = rng.random()
        if t == traj_len - 1:
            act = VLN_STOP
        elif p < 0.65:
            act = VLN_FORWARD
        elif p < 0.80:
            act = VLN_LEFT
        elif p < 0.95:
            act = VLN_RIGHT
        else:
            act = VLN_STOP

        actions[t] = act

        # Apply action to update pose
        if act == VLN_FORWARD:
            x += step_size * np.cos(theta)
            y += step_size * np.sin(theta)
        elif act == VLN_LEFT:
            theta += turn_angle
        elif act == VLN_RIGHT:
            theta -= turn_angle
        # STOP: no change

        position[t] = [x, y]
        yaw[t] = theta

        # Create a simple coloured image so the dataloader can load it
        colour = rng.integers(0, 255, size=3, dtype=np.uint8)
        img = Image.fromarray(
            np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), colour, dtype=np.uint8)
        )
        img.save(os.path.join(img_dir, f"{t:05d}.png"))

    traj_data = {
        "position": position,
        "yaw": yaw,
        "action": actions,
    }
    with open(os.path.join(ep_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)


def main(force: bool = False) -> None:
    # ---- train split -------------------------------------------------------
    train_dir = os.path.join(DATASET_ROOT, "train")
    test_dir = os.path.join(DATASET_ROOT, "test")
    rng = np.random.default_rng(42)

    for split_dir, num_eps, prefix in [
        (train_dir, NUM_TRAIN_EPISODES, "ep_"),
        (test_dir, NUM_TEST_EPISODES, "ep_"),
    ]:
        os.makedirs(split_dir, exist_ok=True)
        ep_names = []
        for i in range(num_eps):
            ep_name = f"{prefix}{i:02d}"
            ep_dir = os.path.join(split_dir, ep_name)
            if force and os.path.exists(ep_dir):
                shutil.rmtree(ep_dir)
            if not os.path.exists(os.path.join(ep_dir, "traj_data.pkl")):
                _make_episode(ep_dir, TRAJ_LEN, rng)
                print(f"  Created {ep_dir}")
            else:
                print(f"  Skipping {ep_dir} (already exists)")
            ep_names.append(ep_name)

        with open(os.path.join(split_dir, "traj_names.txt"), "w") as f:
            f.write("\n".join(ep_names))

    print("Sample VLN-CE dataset created at", DATASET_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    args = parser.parse_args()
    main(force=args.force)

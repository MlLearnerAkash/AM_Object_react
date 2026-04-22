"""
convert_h5_to_vint.py
=====================
Convert an Object-React H5 episode dataset (e.g. ``e3d_train.h5``) into the
ViNT/GNM training layout described in ``README_GNM.md``::

    <output_dir>/
      <traj_name>/                # one per H5 episode
        0.jpg, 1.jpg, ..., N-1.jpg          # forward-facing RGB per step
        masks/0.npz, ...                    # uint8 per-frame instance masks
        traj_data.pkl                       # dict with 'position', 'yaw', ...

The pickle additionally carries the data needed for joint training with
``LangGeoNetV2``:

    {
        "position":     np.ndarray [T, 2]  world XZ (Habitat convention)
        "yaw":          np.ndarray [T]     world yaw (radians)
        "instruction":  str                episode instruction
        "gt_costs":     list[T]            per-frame [K_t] float32, raw
                                           Dijkstra path-length to goal step
        "cat_names":    list[T]            per-frame list[K_t] of MP3D class
                                           names (best-effort)
    }

Usage
-----
    cd /data/ws/VLN-CE/controller/object_react/train
    python convert_h5_to_vint.py \\
        --h5 /media/opervu-user/Data2/ws/data_langgeonet_e3d_action/e3d_train.h5 \\
        --out /media/opervu-user/Data2/ws/data_langgeonet_e3d_action/vint_format \\
        --jpeg_quality 92

After conversion run ``data_split.py`` (already in this repo) to produce the
``train/`` and ``test/`` ``traj_names.txt`` splits expected by the dataloader.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import types

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mock _magnum so pickled NetworkX graphs (containing Habitat C++ types) load
# without the native extension.  Mirrors the helper in joint_dataset.py.
# ---------------------------------------------------------------------------
def _install_mock_magnum() -> None:
    if "_magnum" in sys.modules:
        return

    class _MagnumBase:
        def __init__(self, *a, **kw):
            pass
        def __setstate__(self, state):
            self._state = state

    class _Vector3(_MagnumBase):
        def as_array(self) -> np.ndarray:
            return np.frombuffer(self._state, dtype=np.float32).astype(np.float64)

    mod = types.ModuleType("_magnum")
    mod.Vector3 = _Vector3
    for _name in ("Vector4", "Matrix4", "Matrix3", "Quaternion",
                  "Rad", "Deg", "Range3D", "Range2D"):
        setattr(mod, _name, type(_name, (_MagnumBase,), {}))
    sys.modules["_magnum"] = mod


_install_mock_magnum()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _yaw_from_rotmat(R: np.ndarray) -> float:
    """Habitat: forward = -R[:,2]; yaw = atan2(forward_x, forward_z)."""
    return float(np.arctan2(-R[0, 2], -R[2, 2]))


def _load_graph(raw_bytes: bytes):
    return pickle.loads(raw_bytes)


def _extract_per_frame(G, h5_frame_keys):
    """
    Return dict frame_key -> (raw_costs[K], cat_names[K], pos[3], rot[3,3]).
    raw_costs are minimum Dijkstra path lengths from each source node in the
    frame to any node in the goal frame (largest map[0] step in the graph).
    """
    nodes = list(G.nodes())
    if not nodes:
        return {}

    apl = G.graph.get("all_paths_lengths")
    if apl is None:
        return {}

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    steps = [G.nodes[n]["map"][0] for n in nodes]
    goal_step = max(steps)
    goal_idxs = [node_to_idx[n] for n in nodes
                 if G.nodes[n]["map"][0] == goal_step]

    step_to_nodes: dict = {}
    for n in nodes:
        step_to_nodes.setdefault(G.nodes[n]["map"][0], []).append(n)
    for s in step_to_nodes:
        step_to_nodes[s].sort(key=lambda n: G.nodes[n]["map"][1])

    h5_keys = set(h5_frame_keys)
    out: dict = {}
    for step, src_nodes in step_to_nodes.items():
        fk = f"{step:03d}"
        if fk not in h5_keys:
            continue
        src_idxs = [node_to_idx[n] for n in src_nodes]
        if not src_idxs or not goal_idxs:
            continue
        rows = apl[np.ix_(src_idxs, goal_idxs)]            # [K, G_goal]
        raw_costs = np.nanmin(rows, axis=1).astype(np.float32)
        cat_names = [
            str(G.nodes[n].get("instance_dict", {}).get("category_name", ""))
            for n in src_nodes
        ]
        pos = np.asarray(G.nodes[src_nodes[0]]["agent_position"],
                         dtype=np.float64)
        rot = np.asarray(G.nodes[src_nodes[0]]["agent_rotation"],
                         dtype=np.float64)
        out[fk] = (raw_costs, cat_names, pos, rot)
    return out


# ---------------------------------------------------------------------------
# Per-episode conversion
# ---------------------------------------------------------------------------
def convert_episode(
    ep_grp,
    ep_id: str,
    out_root: str,
    jpeg_quality: int,
    save_masks: bool,
) -> bool:
    """Convert one H5 episode into a ViNT trajectory folder. Returns success."""
    if "frames" not in ep_grp or "graph" not in ep_grp:
        return False

    raw_instr = ep_grp["instruction"][()]
    instruction = (raw_instr.decode("utf-8")
                   if isinstance(raw_instr, bytes) else str(raw_instr))

    try:
        G = _load_graph(ep_grp["graph"][()].tobytes())
    except Exception as e:                                  # noqa: BLE001
        print(f"  [skip {ep_id}] graph unpickle failed: {e}")
        return False

    h5_keys = sorted(ep_grp["frames"].keys())
    per_frame = _extract_per_frame(G, h5_keys)
    del G
    if not per_frame:
        return False

    # Keep only frames with masks AND graph entry.
    valid_keys = [
        fk for fk in h5_keys
        if fk in per_frame and "masks" in ep_grp["frames"][fk]
    ]
    if len(valid_keys) < 2:
        return False

    traj_dir = os.path.join(out_root, ep_id)
    os.makedirs(traj_dir, exist_ok=True)
    masks_dir = os.path.join(traj_dir, "masks")
    if save_masks:
        os.makedirs(masks_dir, exist_ok=True)

    positions, yaws = [], []
    gt_costs, cat_names_all = [], []

    for i, fk in enumerate(valid_keys):
        raw_costs, cat_names, pos, rot = per_frame[fk]
        # ---- World XZ position + yaw ------------------------------------
        positions.append(np.array([pos[0], pos[2]], dtype=np.float64))
        yaws.append(_yaw_from_rotmat(rot))

        # ---- RGB → JPEG -------------------------------------------------
        rgb = ep_grp["frames"][fk]["rgb"][()].astype(np.uint8)
        Image.fromarray(rgb).save(
            os.path.join(traj_dir, f"{i}.jpg"),
            quality=jpeg_quality,
        )

        # ---- Masks → packed npz ----------------------------------------
        if save_masks:
            masks_arr = ep_grp["frames"][fk]["masks"][()].astype(np.uint8)
            np.savez_compressed(
                os.path.join(masks_dir, f"{i}.npz"),
                masks=masks_arr,
            )

        gt_costs.append(raw_costs.astype(np.float32))
        cat_names_all.append(cat_names)

    traj_data = {
        "position":    np.stack(positions, axis=0).astype(np.float64),  # [T, 2]
        "yaw":         np.asarray(yaws, dtype=np.float64),              # [T]
        "instruction": instruction,
        "gt_costs":    gt_costs,
        "cat_names":   cat_names_all,
    }
    with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", required=True, help="Path to the input H5 file")
    ap.add_argument("--out", required=True,
                    help="Output dataset directory (ViNT layout)")
    ap.add_argument("--jpeg_quality", type=int, default=92)
    ap.add_argument("--no_masks", action="store_true",
                    help="Skip writing per-frame masks.npz (saves disk)")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Convert at most N episodes (debug)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n_ok = n_skip = 0
    with h5py.File(args.h5, "r") as hf:
        ep_ids = sorted(hf.keys())
        if args.limit > 0:
            ep_ids = ep_ids[:args.limit]
        for ep_id in tqdm(ep_ids, desc="Episodes"):
            try:
                ok = convert_episode(
                    hf[ep_id], ep_id, args.out,
                    args.jpeg_quality, save_masks=not args.no_masks,
                )
            except Exception as e:                          # noqa: BLE001
                print(f"  [error {ep_id}] {e}")
                ok = False
            n_ok += int(ok)
            n_skip += int(not ok)

    print(f"\nDone. Converted {n_ok} episodes, skipped {n_skip}. Output: {args.out}")


if __name__ == "__main__":
    main()

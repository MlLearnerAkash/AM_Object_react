"""
JointEpisodeDataset
====================
Unified dataset for joint LangGeoNetV2 + GNM training.

GT costs are derived from the **NetworkX graph** stored per episode in the H5
file, using the pre-computed ``all_paths_lengths`` (all-pairs Dijkstra)
matrix.  For every source frame the minimum path length to any node in the
**goal frame** (last step) is computed, then min-max normalised within that
frame.  Object category names are also read from graph node attributes so
no separate class-id lookup is needed.

Because the graph was pickled with Habitat/Magnum C++ objects (Vector3,
Matrix4, …), and those native extensions are not available at training time,
a lightweight mock ``_magnum`` module is injected into ``sys.modules`` at
import time so that unpickling succeeds without the real C++ bindings.

Per-frame it produces:
  • LangE3D inputs  — pixel_values, masks, NAI tokens
  • LangE3D targets — gt_costs (per-frame min-max-normalised path lengths)
  • OGCL inputs     — class_match (which objects match the NAI category)
  • GNM inputs      — gnm_masks (resized to gnm_mask_h × gnm_mask_w)

H5 layout expected
------------------
  <ep_id>/
    instruction                  (bytes string)
    graph                        (uint8 blob — pickled NetworkX graph)
    frames/
      <NNN>/
        rgb                      [H, W, 3] uint8
        masks                    [K, H, W] uint8  (binary, already decoded)
        next_action_instruction  (bytes string, optional)
"""

from __future__ import annotations
import sys
import types
import pickle
import random

import numpy as np
import h5py
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

from dataset import (          # noqa: E402
    _extract_mp3d_class,
)

# ---------------------------------------------------------------------------
# Mock _magnum so that graphs pickled with Habitat C++ objects can be loaded
# without the real native extension.  Installed once at module import time.
# ---------------------------------------------------------------------------

def _install_mock_magnum() -> None:
    """Inject a stub ``_magnum`` module into sys.modules if absent."""
    if "_magnum" in sys.modules:
        return

    class _MagnumBase:
        """Stub for any Magnum C++ value type (Vector3, Matrix4, …)."""
        def __init__(self, *args, **kwargs):
            pass
        def __setstate__(self, state):
            # Magnum types serialise as raw bytes of their underlying data.
            self._state = state

    class _Vector3(_MagnumBase):
        """Decodable stub: exposes the 3 float32 components as a numpy array."""
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

def _parse_nai_class(text: str) -> str | None:
    """Return the first MP3D class name found in *text*, or None."""
    cls, _ = _extract_mp3d_class(text)
    return cls


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Per-frame min-max normalise a [K] float array to [0, 1].

    Objects with non-finite values are set to 1.0 (treat as maximum cost).
    If all finite values collapse to a single value, returns all zeros for
    finite entries (nothing to discriminate).
    """
    out = np.ones(len(arr), dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.sum() < 1:
        return out
    v = arr[finite].astype(np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo > 1e-8:
        out[finite] = ((arr[finite].astype(np.float64) - lo) / (hi - lo)).astype(np.float32)
    else:
        out[finite] = 0.0      # all equal → all zero cost
    return out


def _yaw_from_rotmat(R: np.ndarray) -> float:
    """
    Extract yaw (rotation around the Y/up axis) from a 3×3 agent-to-world
    rotation matrix using the Habitat/MP3D Y-up convention.

    The forward direction in world space is  -R[:,2]  (Habitat's forward is
    -Z in agent frame).  Yaw is the signed angle of that vector in the XZ
    plane: atan2(forward_x, forward_z).
    """
    fwd_x = -R[0, 2]
    fwd_z = -R[2, 2]
    return float(np.arctan2(fwd_x, fwd_z))


def _to_local_coords_2d(
    world_pos: np.ndarray,
    origin_pos: np.ndarray,
    origin_rot: np.ndarray,
) -> np.ndarray:
    """
    Project *world_pos* into the 2-D local frame defined by *origin_pos*
    and *origin_rot* (3×3 agent-to-world rotation matrix).

    The world displacement is rotated by R.T (world→agent), then the X and Z
    components (right and forward in the horizontal plane) are returned as a
    [2] float32 array (x_local, z_local).
    """
    d = (world_pos - origin_pos).astype(np.float64)
    d_local = origin_rot.T @ d         # [3] in agent frame
    return np.array([d_local[0], d_local[2]], dtype=np.float32)


def _load_graph(raw_bytes: bytes):
    """
    Unpickle a Habitat/MP3D NetworkX graph that may reference ``_magnum``
    native types.  The mock module injected at import time absorbs those
    references so the remainder of the graph (numpy arrays, plain dicts,
    etc.) deserialises cleanly.
    """
    return pickle.loads(raw_bytes)


def _extract_graph_frame_data(G, h5_frame_keys: list[str]) -> dict:
    """
    Given a loaded NetworkX graph *G* and the set of H5 frame keys present
    for an episode, return a dict mapping

        frame_key → (raw_costs, cat_names, agent_pos, agent_rot, obb_centers)

    raw_costs  : [K] float64      — min Dijkstra path length to goal frame.
    cat_names  : [K] str          — MP3D category name per node, in map[1] order.
    agent_pos  : [3] float64      — agent position in world frame for this step.
    agent_rot  : [3,3] float64    — agent-to-world rotation matrix for this step.
    obb_centers: list[K] float64  — world-space OBB centre for each object.

    Only frame keys that appear in *h5_frame_keys* are included.
    """
    nodes       = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    apl         = G.graph.get("all_paths_lengths")

    if apl is None or len(nodes) == 0:
        return {}

    # Goal frame = the largest frame-step index in the graph.
    all_steps  = [G.nodes[n]["map"][0] for n in nodes]
    goal_step  = max(all_steps)
    goal_nodes = [n for n in nodes if G.nodes[n]["map"][0] == goal_step]
    goal_idxs  = [node_to_idx[n] for n in goal_nodes]

    # Group source nodes by frame step; sort by map[1] (object index within frame)
    # to match the H5 mask ordering.
    step_to_nodes: dict[int, list] = {}
    for n in nodes:
        step = G.nodes[n]["map"][0]
        step_to_nodes.setdefault(step, []).append(n)
    for step in step_to_nodes:
        step_to_nodes[step].sort(key=lambda n: G.nodes[n]["map"][1])

    h5_key_set = set(h5_frame_keys)
    result: dict[str, tuple] = {}

    for step, src_nodes in step_to_nodes.items():
        frame_key = f"{step:03d}"
        if frame_key not in h5_key_set:
            continue

        src_idxs = [node_to_idx[n] for n in src_nodes]
        K        = len(src_idxs)

        if K == 0 or len(goal_idxs) == 0:
            raw_costs = np.full(K, np.nan, dtype=np.float64)
        else:
            path_rows = apl[np.ix_(src_idxs, goal_idxs)]  # [K, G_goal]
            raw_costs = np.nanmin(path_rows, axis=1)        # [K] float64

        cat_names = [
            G.nodes[n].get("instance_dict", {}).get("category_name", "")
            for n in src_nodes
        ]

        # All nodes in a frame share the same agent pose.
        agent_pos = np.asarray(G.nodes[src_nodes[0]]["agent_position"],
                               dtype=np.float64)
        agent_rot = np.asarray(G.nodes[src_nodes[0]]["agent_rotation"],
                               dtype=np.float64)

        # obb_center world positions — decoded from the _Vector3 stub.
        obb_centers = []
        for n in src_nodes:
            obb = G.nodes[n].get("instance_dict", {}).get("obb_center")
            if obb is not None and hasattr(obb, "as_array"):
                obb_centers.append(obb.as_array())   # [3] float64
            else:
                obb_centers.append(agent_pos.copy()) # fallback: agent pos

        result[frame_key] = (raw_costs, cat_names, agent_pos, agent_rot, obb_centers)

    return result


def _resize_masks_to(masks_uint8: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Resize [K, Hm, Wm] uint8 masks to [K, H, W] float32 using nearest-neighbour.
    Returns float32 in {0.0, 1.0}.
    """
    K, Hm, Wm = masks_uint8.shape
    if K == 0:
        return np.zeros((0, H, W), dtype=np.float32)
    if Hm == H and Wm == W:
        return masks_uint8.astype(np.float32)
    t = torch.from_numpy(masks_uint8).unsqueeze(0).float()  # [1, K, Hm, Wm]
    t = F.interpolate(t, size=(H, W), mode="nearest")       # [1, K, H, W]
    return t.squeeze(0).numpy()                             # [K, H, W] float32


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JointEpisodeDataset(Dataset):
    """
    Parameters
    ----------
    h5_path          : Path to the stratified H5 file.
    episode_ids      : Episode keys to use (None = all).
    clip_model_name  : CLIP processor HuggingFace tag.
    len_traj_pred    : T — waypoint steps (action_label shape; always zeros).
    min_action_dist  : Unused; kept for API compatibility.
    max_action_dist  : Unused; kept for API compatibility.
    gnm_mask_h       : Target height for GNM masks (default 60 = 120//2).
    gnm_mask_w       : Target width  for GNM masks (default 80 = 160//2).
    """

    def __init__(
        self,
        h5_path: str,
        episode_ids: list = None,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        len_traj_pred: int = 10,
        min_action_dist: int = 2,
        max_action_dist: int = 10,
        gnm_mask_h: int = 60,
        gnm_mask_w: int = 80,
    ):
        super().__init__()
        self.h5_path       = h5_path
        self.len_traj_pred = len_traj_pred
        self.gnm_mask_h    = gnm_mask_h
        self.gnm_mask_w    = gnm_mask_w
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self._h5: h5py.File = None   # opened lazily per DataLoader worker

        self.frames: list[dict] = []
        self._build_index(h5_path, episode_ids)

    # ------------------------------------------------------------------
    # Index construction (once at start-up, main process)
    # ------------------------------------------------------------------

    def _build_index(self, h5_path: str, episode_ids):
        """
        Walk the H5, load each episode's graph, pre-compute per-frame
        min-path-to-goal costs and category names.  No pixel data is read.
        """
        n_frames    = 0
        n_skipped   = 0
        n_ep_skip   = 0

        with h5py.File(h5_path, "r") as hf:
            all_keys  = sorted(hf.keys())
            requested = episode_ids if episode_ids is not None else all_keys
            valid_eps = [ep for ep in requested
                         if ep in hf and "frames" in hf[ep]]

            print(f"[JointEpisodeDataset] indexing {len(valid_eps)} episodes …")

            for ep_id in valid_eps:
                ep_grp = hf[ep_id]

                # ---- Episode-level instruction (fallback) ----------------
                raw = ep_grp["instruction"][()]
                instruction = (raw.decode("utf-8")
                               if isinstance(raw, bytes) else str(raw))

                # ---- Load graph and extract per-frame cost data ----------
                if "graph" not in ep_grp:
                    n_ep_skip += 1
                    continue
                try:
                    raw_graph = ep_grp["graph"][()].tobytes()
                    G = _load_graph(raw_graph)
                except Exception as exc:
                    n_ep_skip += 1
                    print(f"  [warn] {ep_id}: graph load failed — {exc}")
                    continue

                h5_frame_keys = sorted(ep_grp["frames"].keys())
                frame_data = _extract_graph_frame_data(G, h5_frame_keys)

                if not frame_data:
                    n_ep_skip += 1
                    continue

                # Free the graph from memory before the next episode.
                del G

                # ---- Build per-frame action labels ----------------------
                # Order frames by step index so we can look up future poses.
                valid_keys = [
                    fk for fk in h5_frame_keys
                    if fk in frame_data
                    and "masks" in ep_grp["frames"][fk]
                    and len(frame_data[fk][0]) > 0
                ]
                T = self.len_traj_pred

                for seq_idx, frame_key in enumerate(valid_keys):
                    raw_costs, cat_names, curr_pos, curr_rot, obb_centers = frame_data[frame_key]
                    K = len(raw_costs)

                    # ---- Action label: interpolate to min-cost object --------
                    finite_mask = np.isfinite(raw_costs)
                    if finite_mask.any():
                        best_k  = int(np.argmin(
                            np.where(finite_mask, raw_costs, np.inf)
                        ))
                        target_world = obb_centers[best_k]   # [3] world XYZ

                        # --- Fix 1: cap target distance -------------------
                        MAX_TARGET_DIST = 25.0          # metres
                        dir_vec  = target_world - curr_pos
                        dist_3d  = float(np.linalg.norm(dir_vec))
                        if dist_3d > MAX_TARGET_DIST and dist_3d > 1e-4:
                            target_world = curr_pos + dir_vec * (MAX_TARGET_DIST / dist_3d)

                        # --- Fix 2: correct yaw labels --------------------
                        target_local = _to_local_coords_2d(target_world, curr_pos, curr_rot)
                        dist_2d = float(np.linalg.norm(target_local))
                        if dist_2d > 1e-4:
                            cos_yaw = float(target_local[0] / dist_2d)
                            sin_yaw = float(target_local[1] / dist_2d)
                        else:
                            cos_yaw, sin_yaw = 1.0, 0.0

                        STEP_SIZE    = 0.5  # metres per step
                        action_label = np.zeros((T, 4), dtype=np.float32)
                        # per-waypoint mask: 1 if still moving, 0 once clamped
                        wp_mask = np.zeros(T, dtype=np.float32)
                        for t in range(T):
                            d = (t + 1) * STEP_SIZE
                            if d <= dist_2d:
                                ratio = d / dist_2d if dist_2d > 1e-4 else 1.0
                                wp_mask[t] = 1.0
                            else:
                                ratio = 1.0   # clamped at target
                            wp_world = curr_pos + ratio * (target_world - curr_pos)
                            xy = _to_local_coords_2d(wp_world, curr_pos, curr_rot)
                            action_label[t] = [xy[0], xy[1], cos_yaw, sin_yaw]
                        # action_mask: scalar — episode is valid; wp_mask stored separately
                        action_mask = 1.0
                        action_mask = 1.0
                    else:
                        action_label = np.zeros((T, 4), dtype=np.float32)
                        action_mask  = 0.0

                    # dist_label: remaining H5 frames to goal.
                    dist_label = len(valid_keys) - 1 - seq_idx

                    self.frames.append({
                        "ep_id":        ep_id,
                        "frame_key":    frame_key,
                        "instruction":  instruction,
                        "K":            K,
                        "raw_costs":    raw_costs,     # [K] float64
                        "cat_names":    cat_names,     # [K] str
                        "action_label": action_label,  # [T, 4] float32
                        "wp_mask":      wp_mask if finite_mask.any() else np.zeros(T, dtype=np.float32),  # [T]
                        "action_mask":  float(action_mask),
                        "dist_label":   dist_label,    # int
                    })
                    n_frames += 1

        print(
            f"[JointEpisodeDataset] ready — {n_frames} frames "
            f"({n_ep_skip} episodes skipped, {n_skipped} frames skipped)"
        )

    # ------------------------------------------------------------------
    # Worker helpers
    # ------------------------------------------------------------------

    def _open_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.frames)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        meta      = self.frames[idx]
        ep_id     = meta["ep_id"]
        frame_key = meta["frame_key"]

        h5   = self._open_h5()
        fg   = h5[ep_id]["frames"][frame_key]

        # ---- RGB → CLIP pixel_values ------------------------------------
        rgb = fg["rgb"][()]                                    # [H, W, 3] uint8
        H, W = rgb.shape[:2]
        pil_img = Image.fromarray(rgb.astype(np.uint8))
        pixel_values = self.clip_processor(
            images=pil_img, return_tensors="pt"
        )["pixel_values"].squeeze(0)                          # [3, 224, 224]

        # ---- Masks [K, H, W] bool --------------------------------------
        masks_arr = fg["masks"][()].astype(bool)              # [K, H, W]
        K = masks_arr.shape[0]

        # ---- GT costs from all_paths_lengths (pre-computed at index time) --
        # raw_costs are min Dijkstra distances to goal frame, float64 [K].
        gt_costs = _minmax_normalize(meta["raw_costs"])       # [K] float32 in [0,1]

        # ---- NAI text → tokens -----------------------------------------
        nai_text = ""
        if "next_action_instruction" in fg:
            raw = fg["next_action_instruction"][()]
            nai_text = (raw.decode("utf-8")
                        if isinstance(raw, bytes) else str(raw))
        if not nai_text.strip():
            nai_text = meta["instruction"]

        nai_tok = self.clip_processor(
            text=nai_text,
            padding="max_length", truncation=True,
            max_length=77, return_tensors="pt",
        )
        nai_input_ids = nai_tok["input_ids"].squeeze(0)       # [77]
        nai_attn_mask = nai_tok["attention_mask"].squeeze(0)  # [77]

        # ---- Class match for OGCL --------------------------------------
        # Use category names extracted from graph node attributes at index time.
        nai_class   = _parse_nai_class(nai_text)
        cat_names   = meta["cat_names"]                        # [K] str
        if nai_class:
            class_match = np.array(
                [c == nai_class for c in cat_names], dtype=bool
            )
        else:
            class_match = np.zeros(K, dtype=bool)

        # ---- GNM masks (resized to [K, gnm_mask_h, gnm_mask_w]) --------
        gnm_masks = _resize_masks_to(
            fg["masks"][()],       # [K, H, W] uint8
            self.gnm_mask_h,
            self.gnm_mask_w,
        )                          # [K, gnm_mask_h, gnm_mask_w] float32

        # ---- Action label (pre-computed at index time from graph poses) ----
        action_label = meta["action_label"]          # [T, 4] float32
        wp_mask      = meta["wp_mask"]               # [T] float32 — 1 while moving
        action_mask  = meta["action_mask"]           # float
        dist_label   = meta["dist_label"]            # int
        has_target   = False
        target_bearing = 0.0

        return {
            # LangE3D
            "pixel_values":   pixel_values,                           # [3, 224, 224]
            "masks":          torch.from_numpy(masks_arr).bool(),     # [K, H, W]
            "nai_input_ids":  nai_input_ids,                         # [77]
            "nai_attn_mask":  nai_attn_mask,                         # [77]
            "gt_costs":       torch.from_numpy(gt_costs),            # [K]
            "class_match":    torch.from_numpy(class_match),         # [K] bool
            # GNM
            "gnm_masks":      torch.from_numpy(gnm_masks),           # [K, Hh, Wh]
            "action_label":   torch.from_numpy(action_label),        # [T, 4]
            "wp_mask":        torch.from_numpy(wp_mask),             # [T]
            "dist_label":     torch.tensor(dist_label, dtype=torch.int64),
            "action_mask":    torch.tensor(action_mask, dtype=torch.float32),
            # Success metric
            "target_bearing": torch.tensor(target_bearing, dtype=torch.float32),
            "has_target":     torch.tensor(has_target, dtype=torch.bool),
            # Instruction text for visualization
            "nai_text":       nai_text,
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def joint_collate_fn(batch: list[dict]) -> dict:
    """
    Stack fixed-size tensors; keep variable-K tensors as lists.
    gnm_masks is padded to [B, K_max, Hh, Wh] and K_list returned.
    """
    K_list = [b["masks"].shape[0] for b in batch]
    K_max  = max(K_list) if K_list else 1

    ref    = batch[0]["gnm_masks"]
    Hh, Wh = ref.shape[1], ref.shape[2]
    gnm_masks_padded = torch.zeros(len(batch), K_max, Hh, Wh, dtype=torch.float32)
    for b_idx, b in enumerate(batch):
        k = K_list[b_idx]
        if k > 0:
            gnm_masks_padded[b_idx, :k] = b["gnm_masks"]

    return {
        # LangE3D — variable K → lists
        "pixel_values":     torch.stack([b["pixel_values"]  for b in batch]),
        "nai_input_ids":    torch.stack([b["nai_input_ids"] for b in batch]),
        "nai_attn_mask":    torch.stack([b["nai_attn_mask"] for b in batch]),
        "masks_list":       [b["masks"]       for b in batch],
        "gt_costs_list":    [b["gt_costs"]    for b in batch],
        "class_match_list": [b["class_match"] for b in batch],
        # GNM — padded/stacked
        "gnm_masks":        gnm_masks_padded,
        "K_list":           K_list,
        "action_label":     torch.stack([b["action_label"] for b in batch]),
        "wp_mask":          torch.stack([b["wp_mask"]      for b in batch]),
        "dist_label":       torch.stack([b["dist_label"]   for b in batch]),
        "action_mask":      torch.stack([b["action_mask"]  for b in batch]),
        # Success metric
        "target_bearing":   torch.stack([b["target_bearing"] for b in batch]),
        "has_target":       torch.stack([b["has_target"]     for b in batch]),
        # Instruction text for visualization
        "nai_text":         [b["nai_text"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_joint_dataloaders(
    h5_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    val_split: float = 0.1,
    seed: int = 42,
    clip_model: str = "openai/clip-vit-base-patch16",
    len_traj_pred: int = 10,
    min_action_dist: int = 2,
    max_action_dist: int = 10,
    gnm_mask_h: int = 60,
    gnm_mask_w: int = 80,
) -> tuple:
    """Return (train_loader, val_loader) with an episode-level split."""
    with h5py.File(h5_path, "r") as hf:
        all_keys = sorted(hf.keys())

    rng = random.Random(seed)
    keys = list(all_keys)
    rng.shuffle(keys)
    split_idx = int(len(keys) * (1 - val_split))
    train_ids = keys[:split_idx]
    val_ids   = keys[split_idx:]

    ds_kw = dict(
        clip_model_name=clip_model,
        len_traj_pred=len_traj_pred,
        min_action_dist=min_action_dist,
        max_action_dist=max_action_dist,
        gnm_mask_h=gnm_mask_h,
        gnm_mask_w=gnm_mask_w,
    )
    train_ds = JointEpisodeDataset(h5_path, train_ids, **ds_kw)
    val_ds   = JointEpisodeDataset(h5_path, val_ids,   **ds_kw)

    loader_kw = dict(
        collate_fn=joint_collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, **loader_kw,
    )
    return train_loader, val_loader

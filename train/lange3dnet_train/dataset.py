import os
import glob
import json
import numpy as np
import h5py
from PIL import Image
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from utils.h5_writer import load_episode_from_hdf5
from pycocotools import mask as mask_utils


# ---------------------------------------------------------------------------
# MP3D category vocabulary and counterfactual instruction utilities (A3 / A2)
# ---------------------------------------------------------------------------

MP3D_CATS: dict = {
     0: "void",        1: "wall",          2: "floor",         3: "chair",
     4: "door",        5: "table",         6: "picture",       7: "cabinet",
     8: "cushion",     9: "window",       10: "sofa",         11: "bed",
    12: "curtain",    13: "chest_of_drawers", 14: "plant",   15: "sink",
    16: "stairs",     17: "ceiling",     18: "toilet",       19: "stool",
    20: "towel",      21: "mirror",      22: "tv_monitor",   23: "shower",
    24: "column",     25: "bathtub",     26: "counter",      27: "fireplace",
    28: "lighting",   29: "beam",        30: "railing",      31: "shelving",
    32: "blinds",     33: "gym_equipment", 34: "seating",   35: "board_panel",
    36: "furniture",  37: "appliances",  38: "clothes",      39: "objects",
}

# Subset of categories that are reasonable navigation targets.
_NAVIGABLE_CLASSES: frozenset = frozenset({
    "chair", "door", "table", "picture", "cabinet", "cushion", "window",
    "sofa", "bed", "curtain", "chest_of_drawers", "plant", "sink", "stairs",
    "toilet", "stool", "towel", "mirror", "tv_monitor", "shower", "bathtub",
    "counter", "fireplace", "shelving", "blinds", "gym_equipment", "seating",
    "furniture", "appliances", "clothes",
})
_NAVIGABLE_LIST: list = sorted(_NAVIGABLE_CLASSES)

# Longest names first so "chest_of_drawers" is matched before "chest".
_OBJ_NAMES_SORTED = sorted(MP3D_CATS.values(), key=len, reverse=True)
_OBJ_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(o) for o in _OBJ_NAMES_SORTED) + r')\b',
    re.IGNORECASE,
)


def _extract_mp3d_class(text: str):
    """Return (class_name, re.Match) for the first MP3D noun in *text*, else (None, None)."""
    m = _OBJ_PATTERN.search(text)
    if m:
        return m.group(1).lower(), m
    return None, None


def _gen_counterfactual_text(text: str, orig_class: str, match) -> tuple:
    """
    Replace *orig_class* in *text* with a randomly sampled navigable distractor.
    Preserves first-letter case of the original word.
    Returns (cf_text, distractor_class_name) or (None, None) if no distractor exists.
    """
    candidates = [c for c in _NAVIGABLE_LIST if c != orig_class]
    if not candidates:
        return None, None
    distractor = random.choice(candidates)
    orig_word  = match.group(1)
    display    = (distractor[0].upper() + distractor[1:]) if orig_word[0].isupper() else distractor
    return text[: match.start()] + display + text[match.end() :], distractor


def _node_category(node_data: dict) -> str:
    """Extract the MP3D category name from a graph node attribute dict.

    Tries common attribute names used across E3D / MP3D graph representations.
    Returns an empty string when no category can be resolved — the caller
    treats the empty string as "unknown" and skips grounding for this node.
    """
    for key in ('category_name', 'category', 'mp3d_category', 'class_name', 'label'):
        v = node_data.get(key)
        if v is not None:
            return str(v).lower()
    for key in ('category_id', 'mp3d_category_id', 'class_id', 'label_id'):
        v = node_data.get(key)
        if isinstance(v, (int, np.integer)):
            return MP3D_CATS.get(int(v), '')
    return ''


class LangGeoNetDataset(Dataset):
    """
    Dataset that yields individual (frame, instruction, GT geodesic) tuples.
    The same instruction is shared by all frames within an episode.
    """

    def __init__(
        self,
        data_root,
        clip_model_name="openai/clip-vit-base-patch16",
        max_instruction_length=77,
        max_objects=50,
        split="train",
    ):
        super().__init__()
        self.data_root = data_root
        self.max_objects = max_objects
        self.max_instruction_length = max_instruction_length

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        self.samples = []
        self._build_index(split)

    def _build_index(self, split):
        """Index all (episode, frame) pairs."""
        split_file = os.path.join(self.data_root, f"{split}.txt")

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                episode_ids = [line.strip() for line in f if line.strip()]
            episode_dirs = [os.path.join(self.data_root, "episode_"+eid) for eid in episode_ids]
        else:
            episode_dirs = sorted(glob.glob(os.path.join(self.data_root, "episode_*")))

        for ep_dir in episode_dirs:
            if not os.path.isdir(ep_dir):
                continue

            instr_path = os.path.join(ep_dir, "instruction.txt")
            if not os.path.exists(instr_path):
                continue

            frame_dirs = sorted(glob.glob(os.path.join(ep_dir, "frame_*")))
            for frame_dir in frame_dirs:
                required = [
                    os.path.join(frame_dir, "rgb.png"),
                    os.path.join(frame_dir, "masks.npy"),
                    os.path.join(frame_dir, "class_ids.npy"),
                    os.path.join(frame_dir, "geodesic_distances.npy"),
                ]
                if all(os.path.exists(p) for p in required):
                    self.samples.append({
                        "episode_dir": ep_dir,
                        "frame_dir": frame_dir,
                        "instruction_path": instr_path,
                    })

        print(f"[LangGeoNetDataset] {len(self.samples)} frames from "
              f"{len(set(s['episode_dir'] for s in self.samples))} episodes ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # -- Instruction --
        with open(sample["instruction_path"], "r") as f:
            instruction = f.read().strip()

        clip_text = self.clip_processor(
            text=instruction,
            max_length=self.max_instruction_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = clip_text["input_ids"].squeeze(0)
        attention_mask = clip_text["attention_mask"].squeeze(0)

        # -- RGB frame --
        image = Image.open(
            os.path.join(sample["frame_dir"], "rgb.png")
        ).convert("RGB")
        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = clip_inputs["pixel_values"].squeeze(0)

        # -- Masks, classes, GT geodesic --
        masks = np.load(os.path.join(sample["frame_dir"], "masks.npy"))
        class_ids = np.load(os.path.join(sample["frame_dir"], "class_ids.npy"))
        geodesic = np.load(os.path.join(sample["frame_dir"], "geodesic_distances.npy"))

        # Filter empty masks (all-zero rows) before passing to model
        valid = masks.any(axis=(1, 2))           # [K] bool
        masks = masks[valid]
        class_ids = class_ids[valid]
        geodesic = geodesic[valid]
        K = masks.shape[0]                        # actual number of valid objects

        masks = torch.from_numpy(masks).bool()
        class_ids = torch.from_numpy(class_ids).long()
        geodesic = torch.from_numpy(geodesic).float()

        return {
            "pixel_values": pixel_values,         # [3, 224, 224]
            "masks": masks,                        # [K, H, W]
            "class_ids": class_ids,                # [K]
            "input_ids": input_ids,                # [L]
            "attention_mask": attention_mask,       # [L]
            "geodesic_distances": geodesic,        # [K]
            "num_objects": K,
            "episode_dir": sample["episode_dir"],
            "frame_dir": sample["frame_dir"],
        }


def langgeonet_collate_fn(batch):
    """
    Custom collate: stack fixed-size tensors, keep variable-size as lists.
    """
    return {
        "pixel_values":       torch.stack([b["pixel_values"] for b in batch]),
        "input_ids":          torch.stack([b["input_ids"] for b in batch]),
        "attention_mask":     torch.stack([b["attention_mask"] for b in batch]),
        "masks_list":         [b["masks"] for b in batch],
        "class_ids_list":     [b["class_ids"] for b in batch],
        "geodesic_dists_list":[b["geodesic_distances"] for b in batch],
        "num_objects":        [b["num_objects"] for b in batch],
    }


def create_dataloaders(data_root, batch_size=8, num_workers=4,
                       clip_model="openai/clip-vit-base-patch16"):
    """Create train and val dataloaders."""
    train_ds = LangGeoNetDataset(data_root, clip_model, split="train")
    val_ds   = LangGeoNetDataset(data_root, clip_model, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=langgeonet_collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=langgeonet_collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# H5-backed dataset: RLE masks + PL scores + JSONL instructions
# ---------------------------------------------------------------------------

class H5MaskPLSDataset(Dataset):
    """
    Dataset that loads per-frame RGB images, instance-segmentation masks, and
    per-object path-length (PL) scores from an HDF5 file, paired with
    natural-language instructions supplied through a JSONL file.

    HDF5 structure
    --------------
    Top-level key  : "{ep_folder}_{frame_idx}"   e.g. "1S7LAXRdDqK_0000000_plant_42__0"
    Sub-datasets:
        img_masks/{i}  int64 1-D  – F-major alternating skip/set RLE for mask i
        img_pls        float64 [K] – per-object PL scores
        size           int64   [2] – [H, W] of the frame

    JSONL structure (one JSON object per line)
    ------------------------------------------
        {"filename": "<ep_folder_name>", "instruction": "natural language ..."}

    "filename" is matched to the episode-folder component of the H5 key.
    Trailing underscores are stripped on both sides before comparison, so
    "1S7LAXRdDqK_0000000_plant_42_" matches "1S7LAXRdDqK_0000000_plant_42_"
    (or the same string without the trailing underscore).

    Image path
    ----------
    {base_dir}/trajectories/{ep_folder}/images/{frame_idx:05d}.png
    """

    def __init__(
        self,
        h5_path: str,
        jsonl_path: str,
        base_dir: str,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        max_instruction_length: int = 77,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.traj_root = os.path.join(base_dir, "trajectories")
        self.max_instruction_length = max_instruction_length
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # H5 handle – opened lazily in each DataLoader worker (fork-safe).
        self._h5 = None

        # Build instruction lookup: stripped_ep_folder -> instruction
        instruction_map: dict[str, str] = {}
        with open(jsonl_path, "r") as fj:
            for line in fj:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                instruction_map[entry["file"].removesuffix(".gif").rstrip("_")] = entry["instruction"]

        # Index all H5 keys that have a matching instruction entry.
        self.samples: list[dict] = []
        with h5py.File(h5_path, "r") as h5:
            for sample_key in h5.keys():
                # Split off the frame index from the right.
                ep_folder, frame_str = sample_key.rsplit("_", 1)
                lookup_key = ep_folder.rstrip("_")
                if lookup_key not in instruction_map:
                    continue
                self.samples.append({
                    "sample_key":  sample_key,
                    "ep_folder":   ep_folder,
                    "frame_idx":   int(frame_str),
                    "instruction": instruction_map[lookup_key],
                })

        print(
            f"[H5MaskPLSDataset] {len(self.samples)} samples from "
            f"{len(set(s['ep_folder'] for s in self.samples))} episodes"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def _open_h5(self) -> "h5py.File":
        """Open the HDF5 file lazily (once per DataLoader worker process)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    @staticmethod
    def _decode_rle_mask(rle: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Decode a 1-D F-major alternating skip/set RLE into a (H, W) bool array.

        The RLE lists run lengths over a Fortran-order (column-major) flattened
        view of the image.  Runs at even positions (0, 2, …) are background
        (skip); runs at odd positions (1, 3, …) are foreground (set).
        """
        flat = np.zeros(H * W, dtype=bool)
        pos = 0
        for i, count in enumerate(rle):
            if i % 2 == 1:          # odd index → foreground run
                flat[pos: pos + count] = True
            pos += count
        return np.reshape(flat, (H, W), order="F")

    def __getitem__(self, idx: int) -> dict:
        meta = self.samples[idx]
        h5   = self._open_h5()
        grp  = h5[meta["sample_key"]]

        H, W = int(grp["size"][0]), int(grp["size"][1])

        # Decode all masks from F-major RLE.
        masks_grp = grp["img_masks"]
        n_masks   = len(masks_grp)
        masks = np.stack(
            [self._decode_rle_mask(masks_grp[str(i)][()], H, W)
             for i in range(n_masks)],
            axis=0,
        )  # [K, H, W]  bool

        # PL scores – min-max normalized to [0, 1] per frame.
        pls = grp["img_pls"][()].astype(np.float32)  # [K]
        pls_min, pls_max = pls.min(), pls.max()
        if pls_max - pls_min > 1e-6:
            pls = (pls - pls_min) / (pls_max - pls_min)
        else:
            pls = np.zeros_like(pls)

        # RGB image.
        img_path = os.path.join(
            self.traj_root,
            meta["ep_folder"],
            "images",
            f"{meta['frame_idx']:05d}.png",
        )
        image       = Image.open(img_path).convert("RGB")
        clip_img    = self.clip_processor(images=image, return_tensors="pt")
        pixel_values = clip_img["pixel_values"].squeeze(0)  # [3, 224, 224]

        # Instruction tokens.
        clip_text = self.clip_processor(
            text=meta["instruction"],
            max_length=self.max_instruction_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids     = clip_text["input_ids"].squeeze(0)      # [L]
        attention_mask = clip_text["attention_mask"].squeeze(0) # [L]

        return {
            "pixel_values":  pixel_values,                    # [3, 224, 224]
            "masks":         torch.from_numpy(masks).bool(),  # [K, H, W]
            "pls":           torch.from_numpy(pls),           # [K]
            "input_ids":     input_ids,                       # [L]
            "attention_mask": attention_mask,                  # [L]
            "num_objects":   n_masks,
            "sample_key":    meta["sample_key"],
            "ep_folder":     meta["ep_folder"],
            "frame_idx":     meta["frame_idx"],
        }


def h5_maskpls_collate_fn(batch: list[dict]) -> dict:
    """
    Collate for H5MaskPLSDataset.

    Fixed-size tensors are stacked; variable-size masks and pls are kept as
    lists (one element per sample in the batch).
    """
    return {
        "pixel_values":    torch.stack([b["pixel_values"]   for b in batch]),
        "input_ids":       torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask":  torch.stack([b["attention_mask"] for b in batch]),
        "masks_list":      [b["masks"] for b in batch],
        "pls_list":        [b["pls"]   for b in batch],
        "num_objects":     [b["num_objects"] for b in batch],
        "sample_keys":     [b["sample_key"]  for b in batch],
    }


def create_h5_maskpls_dataloader(
    h5_path: str,
    jsonl_path: str,
    base_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    clip_model: str = "openai/clip-vit-base-patch16",
) -> DataLoader:
    """
    Convenience factory that returns a single DataLoader backed by
    H5MaskPLSDataset.

    Parameters
    ----------
    h5_path    : path to the HDF5 file containing masks and PL scores.
    jsonl_path : path to the JSONL file with "filename" / "instruction" pairs.
    base_dir   : root directory that contains the ``trajectories/`` sub-folder.
    batch_size : samples per batch.
    shuffle    : whether to shuffle the dataset each epoch.
    num_workers: number of DataLoader worker processes.
    clip_model : HuggingFace model name used for image/text preprocessing.
    """
    ds = H5MaskPLSDataset(
        h5_path=h5_path,
        jsonl_path=jsonl_path,
        base_dir=base_dir,
        clip_model_name=clip_model,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=h5_maskpls_collate_fn,
        pin_memory=True,
        drop_last=shuffle,   # drop incomplete batches only during training
    )

#------------------xxxxxxxxxxxxx--------------------------------------

class H5EpisodePathLengthsDataset(Dataset):
    """
    Flat frame-level dataset for shuffled training.

    All per-frame data (node RLEs, pre-sliced path-length rows, instruction
    tokens) are extracted from episode graphs **once at construction time**.
    __getitem__ only reads the RGB image from HDF5, decodes masks, and
    returns a ready-to-use sample dict — no graph loading at runtime.

    This makes the dataset compatible with a standard shuffled DataLoader
    without any per-batch graph-unpickling overhead.
    """

    def __init__(self, h5_path: str, episode_ids: list = None):
        super().__init__()
        self.h5_path = h5_path
        self._h5: h5py.File = None   # opened lazily per DataLoader worker
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        with h5py.File(h5_path, 'r') as hf:
            all_keys = sorted(hf.keys())
            requested = episode_ids if episode_ids is not None else all_keys
            valid_episodes = [ep for ep in requested
                              if ep in hf and "frames" in hf[ep]]

        # Flat index: one entry per (episode, frame).  Lightweight — no RGB.
        # Each entry stores the pre-sliced path_rows [K, G] and the list of
        # raw RLE dicts so __getitem__ only needs to read RGB + decode masks.
        self.frames: list[dict] = []
        # Instruction tokens stored per-episode (shared across frames).
        self._ep_tokens: dict[str, tuple] = {}  # ep_id -> (input_ids, attn_mask)

        n_eps = len(valid_episodes)
        n_skipped = 0
        print(f"[H5EpisodePathLengthsDataset] indexing {n_eps} episodes …")
        for ep_idx, ep_id in enumerate(valid_episodes):
            try:
                ep_data = load_episode_from_hdf5(h5_path, ep_id)
            except Exception as exc:
                n_skipped += 1
                continue

            G         = ep_data['graph']
            all_paths = G.graph.get('all_paths_lengths')
            if all_paths is None:
                n_skipped += 1
                continue

            all_nodes   = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(all_nodes)}

            # Goal nodes = nodes belonging to the last frame of the episode.
            all_frame_idxs = [G.nodes[n]['map'][0] for n in all_nodes]
            goal_frame_idx = max(all_frame_idxs) if all_frame_idxs else 0
            goal_node_ids  = [n for n in all_nodes
                               if G.nodes[n]['map'][0] == goal_frame_idx]
            goal_indices   = [node_to_idx[n] for n in goal_node_ids]

            # Tokenise instruction once and store keyed by ep_id.
            if ep_id not in self._ep_tokens:
                tok = self.clip_processor(
                    text=ep_data.get('instruction', ''),
                    padding='max_length', truncation=True,
                    max_length=77, return_tensors='pt',
                )
                self._ep_tokens[ep_id] = (
                    tok['input_ids'].squeeze(0),
                    tok['attention_mask'].squeeze(0),
                )

            # Group nodes by frame index.
            frame_to_nodes: dict = {}
            for n in all_nodes:
                frame_to_nodes.setdefault(G.nodes[n]['map'][0], []).append(n)

            for frame_idx, nodes in frame_to_nodes.items():
                indices = [node_to_idx[n] for n in nodes]
                if indices and goal_indices:
                    path_rows = all_paths[np.ix_(indices, goal_indices)].astype(np.float32)
                else:
                    path_rows = np.zeros((len(indices), 0), dtype=np.float32)

                self.frames.append({
                    'ep_id':     ep_id,
                    'frame_idx': int(frame_idx),
                    'rle_list':  [G.nodes[n].get('segmentation') for n in nodes],
                    'path_rows': path_rows,        # [K, G] float32 — pre-sliced
                })

            if (ep_idx + 1) % 500 == 0 or ep_idx + 1 == n_eps:
                print(f"  {ep_idx + 1}/{n_eps} episodes indexed "
                      f"({len(self.frames)} frames) …")

        print(f"[H5EpisodePathLengthsDataset] ready — {len(self.frames)} frames "
              f"from {n_eps - n_skipped} episodes "
              f"({n_skipped} skipped — missing all_paths_lengths)")

    def __len__(self) -> int:
        return len(self.frames)

    def _open_h5(self) -> h5py.File:
        """Open the HDF5 file lazily (once per DataLoader worker process)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __getitem__(self, idx: int) -> dict:
        meta      = self.frames[idx]
        ep_id     = meta['ep_id']
        frame_idx = meta['frame_idx']
        rle_list  = meta['rle_list']
        path_rows = meta['path_rows']           # [K, G] float32

        # Only HDF5 I/O at runtime — no graph loading.
        h5  = self._open_h5()
        rgb = h5[ep_id]["frames"][f"{frame_idx:03d}"]["rgb"][()]
        H, W = rgb.shape[:2]

        # Decode per-node masks from stored RLE dicts.
        masks_list = []
        for rle in rle_list:
            if rle is None:
                masks_list.append(np.zeros((H, W), dtype=bool))
                continue
            try:
                if not isinstance(rle, dict) or 'counts' not in rle or 'size' not in rle:
                    raise ValueError(f"Malformed RLE: {type(rle)}")
                counts = rle['counts']
                if isinstance(counts, str):
                    counts = counts.encode('utf-8')
                rle_h, rle_w = int(rle['size'][0]), int(rle['size'][1])
                rle_use = {'size': [H, W] if (rle_h != H or rle_w != W) else rle['size'],
                           'counts': counts}
                comp = mask_utils.frPyObjects([rle_use], H, W)
                dec  = mask_utils.decode(comp)
                m    = (dec[..., 0] if dec.ndim == 3 else dec).astype(bool)
            except Exception as exc:
                import warnings
                warnings.warn(f"mask decode failed — zero mask. {type(exc).__name__}: {exc}",
                              RuntimeWarning, stacklevel=2)
                m = np.zeros((H, W), dtype=bool)
            masks_list.append(m)

        masks_arr = np.stack(masks_list) if masks_list else np.zeros((0, H, W), dtype=bool)

        # Normalised GT costs: min path length to nearest goal node → [0, 1].
        K = masks_arr.shape[0]
        if K > 0 and path_rows.shape[1] > 0:
            raw_costs = np.nanmean(path_rows.astype(np.float64), axis=1)  # [K]
        else:
            raw_costs = np.full(K, np.nan, dtype=np.float64)
        finite_mask = np.isfinite(raw_costs)
        if finite_mask.sum() > 1:
            c_min = float(raw_costs[finite_mask].min())
            c_max = float(raw_costs[finite_mask].max())
            if c_max > c_min:
                gt_costs = np.where(finite_mask, (raw_costs - c_min) / (c_max - c_min), np.nan)
            else:
                gt_costs = np.where(finite_mask, 0.0, np.nan)
        else:
            gt_costs = np.zeros(K, dtype=np.float64)

        pil_img      = Image.fromarray(rgb.astype(np.uint8))
        pixel_values = self.clip_processor(
            images=pil_img, return_tensors='pt')['pixel_values'].squeeze(0)

        input_ids, attn_mask = self._ep_tokens[ep_id]
        return {
            'episode_id':     ep_id,
            'frame_idx':      frame_idx,
            'frame_rgb':      rgb,                                             # [H, W, 3] uint8
            'pixel_values':   pixel_values,                                   # [3, 224, 224]
            'masks':          torch.from_numpy(masks_arr).bool(),             # [K, H, W]
            'gt_costs':       torch.from_numpy(gt_costs.astype(np.float32)), # [K]
            'input_ids':      input_ids,                                      # [77]
            'attention_mask': attn_mask,                                      # [77]
        }


def create_h5_episode_pathlengths_dataloader(
    h5_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple:
    """
    Build train and val DataLoaders from a single HDF5 file.

    Episodes are split at the episode level (not frame level), then all frames
    from each split form a flat dataset that is shuffled independently per epoch.
    Each sample carries the correct (image, masks, gt_costs, instruction) tuple
    from the same episode.
    """
    with h5py.File(h5_path, 'r') as hf:
        all_keys = sorted(hf.keys())

    rng = random.Random(seed)
    keys = list(all_keys)
    rng.shuffle(keys)
    split_idx = int(len(keys) * (1 - val_split))
    train_ids = keys[:split_idx]
    val_ids   = keys[split_idx:]

    train_ds = H5EpisodePathLengthsDataset(h5_path, episode_ids=train_ids)
    val_ds   = H5EpisodePathLengthsDataset(h5_path, episode_ids=val_ids)

    def _collate(batch: list[dict]) -> dict:
        return {
            "pixel_values":  torch.stack([b["pixel_values"]   for b in batch]),
            "input_ids":     torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "masks_list":    [b["masks"]    for b in batch],
            "gt_costs_list": [b["gt_costs"] for b in batch],
            "frame_rgbs":    [b["frame_rgb"] for b in batch],
            "episode_ids":   [b["episode_id"] for b in batch],
            "frame_idxs":    [b["frame_idx"]  for b in batch],
        }

    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# LMDB-backed dataset: fast random-access for large-scale training
# ---------------------------------------------------------------------------

def convert_h5_to_lmdb(
    h5_path: str,
    lmdb_path: str,
    clip_model_name: str = "openai/clip-vit-base-patch16",
    jpeg_quality: int = 95,
    map_size_gb: float = 200.0,
) -> None:
    """
    Pre-process every frame in an HDF5 episode file and write the results
    into an LMDB database for fast, random-access batch loading.

    All slow operations (graph unpickling, path-length slicing, CLIP
    tokenisation) are done **once here**, so that LMDBEpisodePathLengthsDataset
    __getitem__ only needs to do a single LMDB read, a JPEG decode, and mask
    decoding — no Python graph objects at training time.

    LMDB layout
    -----------
    Key                          Value (pickle)
    -------------------------    -----------------------------------------------
    b"__keys__"                  list[bytes] — sorted list of all frame keys
    b"{ep_id}/{frame_idx:03d}"   dict with fields:
                                   ep_id            : str
                                   frame_idx        : int
                                   rgb_jpeg         : bytes  (JPEG-encoded RGB)
                                   rle_list         : list[dict | None]
                                   path_rows        : bytes  (float32 ndarray,
                                                             shape [K, G])
                                   path_rows_shape  : tuple  (K, G)
                                   input_ids        : bytes  (int32 ndarray [77])
                                   attention_mask   : bytes  (int32 ndarray [77])
                                   nai_input_ids    : bytes  (int32 ndarray [77])
                                   nai_attention_mask: bytes (int32 ndarray [77])

    Parameters
    ----------
    h5_path        : Path to the source HDF5 file.
    lmdb_path      : Destination directory for the LMDB database.
    clip_model_name: HuggingFace CLIP model used for text tokenisation.
    jpeg_quality   : JPEG quality (1-95) used to compress RGB frames.
    map_size_gb    : LMDB virtual map size in GB.  On Linux this is only
                     virtual address space — actual disk usage is much smaller.
    """
    import io as _io
    import pickle as _pickle
    import lmdb

    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    map_size = int(map_size_gb * (1 << 30))

    with h5py.File(h5_path, 'r') as hf, \
         lmdb.open(lmdb_path, map_size=map_size, subdir=True, meminit=False,
                   map_async=True) as env:

        all_ep_ids = sorted(hf.keys())
        n_eps      = len(all_ep_ids)
        all_keys: list = []

        print(f"[convert_h5_to_lmdb] converting {n_eps} episodes …")
        n_skipped = 0

        for ep_idx, ep_id in enumerate(all_ep_ids):
            ep_grp = hf[ep_id]

            # -- instruction tokens (shared across all frames) ---------------
            raw_instr = ep_grp["instruction"][()]
            instruction = raw_instr.decode("utf-8") if isinstance(raw_instr, bytes) \
                          else str(raw_instr)
            tok = clip_processor(
                text=instruction,
                padding='max_length', truncation=True,
                max_length=77, return_tensors='pt',
            )
            input_ids_np     = tok['input_ids'].squeeze(0).numpy().astype(np.int32)
            attn_mask_np     = tok['attention_mask'].squeeze(0).numpy().astype(np.int32)
            input_ids_bytes  = input_ids_np.tobytes()
            attn_mask_bytes  = attn_mask_np.tobytes()

            # -- graph: unpickle once per episode ----------------------------
            if "graph" not in ep_grp:
                n_skipped += 1
                continue
            try:
                import pickle as _pickle_inner
                graph_bytes = ep_grp["graph"][()].tobytes()
                G           = _pickle_inner.loads(graph_bytes)
            except Exception:
                n_skipped += 1
                continue

            all_paths = G.graph.get('all_paths_lengths')
            if all_paths is None:
                n_skipped += 1
                continue

            all_nodes   = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(all_nodes)}

            all_frame_idxs = [G.nodes[n]['map'][0] for n in all_nodes]
            goal_frame_idx = max(all_frame_idxs) if all_frame_idxs else 0
            goal_indices   = [node_to_idx[n] for n in all_nodes
                              if G.nodes[n]['map'][0] == goal_frame_idx]

            # group nodes by frame index
            frame_to_nodes: dict = {}
            for nd in all_nodes:
                frame_to_nodes.setdefault(G.nodes[nd]['map'][0], []).append(nd)

            with env.begin(write=True) as txn:
                for frame_idx, nodes in frame_to_nodes.items():
                    frame_key_str = f"{ep_id}/{frame_idx:03d}"
                    frame_key     = frame_key_str.encode('utf-8')

                    # -- RGB: read from HDF5 and JPEG-compress ---------------
                    h5_frame_key = f"{frame_idx:03d}"
                    if h5_frame_key not in ep_grp["frames"]:
                        continue
                    rgb = ep_grp["frames"][h5_frame_key]["rgb"][()]
                    pil_img = Image.fromarray(rgb.astype(np.uint8))
                    jpeg_buf = _io.BytesIO()
                    pil_img.save(jpeg_buf, format='JPEG', quality=jpeg_quality)
                    rgb_jpeg = jpeg_buf.getvalue()

                    # -- path_rows: slice [K, G] -----------------------------
                    indices = [node_to_idx[n] for n in nodes]
                    if indices and goal_indices:
                        path_rows = all_paths[
                            np.ix_(indices, goal_indices)
                        ].astype(np.float32)
                    else:
                        path_rows = np.zeros(
                            (len(indices), 0), dtype=np.float32)

                    # -- RLE list and per-node MP3D category names -----------
                    rle_list       = [G.nodes[n].get('segmentation') for n in nodes]
                    category_names = [_node_category(dict(G.nodes[n])) for n in nodes]

                    # -- next_action_instruction tokens (per frame) ----------
                    frame_grp = ep_grp["frames"][h5_frame_key]
                    if "next_action_instruction" in frame_grp:
                        nai_raw = frame_grp["next_action_instruction"][()]
                        nai_text = nai_raw.decode("utf-8") if isinstance(nai_raw, bytes) \
                                   else str(nai_raw)
                    else:
                        nai_text = ""
                    nai_tok = clip_processor(
                        text=nai_text,
                        padding='max_length', truncation=True,
                        max_length=77, return_tensors='pt',
                    )
                    nai_input_ids_bytes  = nai_tok['input_ids'].squeeze(0).numpy().astype(np.int32).tobytes()
                    nai_attn_mask_bytes  = nai_tok['attention_mask'].squeeze(0).numpy().astype(np.int32).tobytes()

                    record = {
                        'ep_id':              ep_id,
                        'frame_idx':          int(frame_idx),
                        'rgb_jpeg':           rgb_jpeg,
                        'rle_list':           rle_list,
                        'mp3d_class_names':   category_names,
                        'path_rows':          path_rows.tobytes(),
                        'path_rows_shape':    path_rows.shape,
                        'input_ids':          input_ids_bytes,
                        'attention_mask':     attn_mask_bytes,
                        'nai_input_ids':      nai_input_ids_bytes,
                        'nai_attention_mask': nai_attn_mask_bytes,
                        'nai_text':           nai_text,
                    }
                    txn.put(frame_key, _pickle.dumps(record, protocol=4))
                    all_keys.append(frame_key)

            if (ep_idx + 1) % 50 == 0 or ep_idx + 1 == n_eps:
                print(f"  {ep_idx + 1}/{n_eps} episodes  "
                      f"({len(all_keys)} frames written) …")

        # write the key index
        with env.begin(write=True) as txn:
            txn.put(b"__keys__", _pickle.dumps(sorted(all_keys), protocol=4))

    print(f"[convert_h5_to_lmdb] done — {len(all_keys)} frames "
          f"from {n_eps - n_skipped} episodes written to {lmdb_path} "
          f"({n_skipped} episodes skipped)")


class LMDBEpisodePathLengthsDataset(Dataset):
    """
    Drop-in replacement for H5EpisodePathLengthsDataset that reads from LMDB.

    __getitem__ only does:
      1. One LMDB key lookup (memory-mapped, near-zero latency)
      2. JPEG decode → CLIP pixel_values
      3. pycocotools RLE mask decode

    All graph loading and path-length slicing was performed once at
    conversion time by convert_h5_to_lmdb().

    Returns the same dict schema as H5EpisodePathLengthsDataset.
    """

    def __init__(self, lmdb_path: str, episode_ids: list = None,
                 nai_text_contains: str = None, counterfactual: bool = False):
        super().__init__()
        import pickle as _pickle
        import lmdb

        self.lmdb_path      = lmdb_path
        self.counterfactual = counterfactual
        self._env           = None   # opened lazily per DataLoader worker
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16")

        # Load the key index from LMDB (read-only, main process).
        env = lmdb.open(lmdb_path, readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            raw = txn.get(b"__keys__")
            if raw is None:
                raise RuntimeError(
                    f"LMDB at {lmdb_path} has no '__keys__' entry. "
                    "Run convert_h5_to_lmdb() first.")
            all_keys: list = _pickle.loads(raw)   # list[bytes]

        # Filter by requested episode_ids if provided.
        if episode_ids is not None:
            ep_set = set(str(e) for e in episode_ids)
            all_keys = [k for k in all_keys
                        if k.decode('utf-8').split('/')[0] in ep_set]

        # Filter by next_action_instruction substring if requested.
        if nai_text_contains is not None:
            needle = nai_text_contains.lower()
            filtered = []
            for k in all_keys:
                rec_raw = None
                with env.begin(write=False) as txn:
                    rec_raw = txn.get(k)
                if rec_raw is None:
                    continue
                rec = _pickle.loads(rec_raw)
                nai = rec.get('nai_text', '')
                if needle in nai.lower():
                    filtered.append(k)
            all_keys = filtered
        env.close()

        self._keys = all_keys   # list[bytes]
        if episode_ids is None:
            ep_desc = "all episodes"
        else:
            n_ep = len({k.decode('utf-8').split('/')[0] for k in self._keys})
            ep_desc = f"{n_ep} episodes"
        filter_desc = f", nai contains '{nai_text_contains}'" if nai_text_contains else ""
        print(f"[LMDBEpisodePathLengthsDataset] {len(self._keys)} frames ({ep_desc}{filter_desc})")

    def __len__(self) -> int:
        return len(self._keys)

    def _open_env(self):
        """Open LMDB lazily once per DataLoader worker (fork-safe)."""
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self.lmdb_path,
                readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._env

    def __getitem__(self, idx: int) -> dict:
        import pickle as _pickle
        import io as _io

        key    = self._keys[idx]
        env    = self._open_env()
        with env.begin(write=False) as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"LMDB key {key} not found")
        rec = _pickle.loads(raw)

        ep_id     = rec['ep_id']
        frame_idx = rec['frame_idx']
        rle_list  = rec['rle_list']

        # -- JPEG decode → CLIP pixel_values ---------------------------------
        pil_img      = Image.open(_io.BytesIO(rec['rgb_jpeg'])).convert('RGB')
        H, W         = pil_img.height, pil_img.width
        pixel_values = self.clip_processor(
            images=pil_img, return_tensors='pt')['pixel_values'].squeeze(0)

        # -- decode RLE masks ------------------------------------------------
        masks_list = []
        for rle in rle_list:
            if rle is None:
                masks_list.append(np.zeros((H, W), dtype=bool))
                continue
            try:
                if not isinstance(rle, dict) or 'counts' not in rle or 'size' not in rle:
                    raise ValueError(f"Malformed RLE: {type(rle)}")
                counts = rle['counts']
                if isinstance(counts, str):
                    counts = counts.encode('utf-8')
                elif isinstance(counts, list):
                    # plain list RLE → COCO frPyObjects expects list-of-counts
                    pass
                rle_h, rle_w = int(rle['size'][0]), int(rle['size'][1])
                rle_use = {
                    'size':   [H, W] if (rle_h != H or rle_w != W) else rle['size'],
                    'counts': counts,
                }
                comp = mask_utils.frPyObjects([rle_use], H, W)
                dec  = mask_utils.decode(comp)
                m    = (dec[..., 0] if dec.ndim == 3 else dec).astype(bool)
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"mask decode failed — zero mask. {type(exc).__name__}: {exc}",
                    RuntimeWarning, stacklevel=2)
                m = np.zeros((H, W), dtype=bool)
            masks_list.append(m)

        masks_arr = np.stack(masks_list) if masks_list \
                    else np.zeros((0, H, W), dtype=bool)
        K = masks_arr.shape[0]

        # -- GT costs from pre-sliced path_rows ------------------------------
        K_stored, G_stored = rec['path_rows_shape']
        path_rows = np.frombuffer(rec['path_rows'], dtype=np.float32) \
                      .reshape(K_stored, G_stored)

        if K > 0 and path_rows.shape[1] > 0:
            raw_costs = np.nanmin(path_rows.astype(np.float64), axis=1)  # [K]
        else:
            raw_costs = np.full(K, np.nan, dtype=np.float64)

        finite = np.isfinite(raw_costs)
        if finite.sum() > 1:
            c_min, c_max = float(raw_costs[finite].min()), float(raw_costs[finite].max())
            if c_max > c_min:
                gt_costs = np.where(finite, (raw_costs - c_min) / (c_max - c_min), np.nan)
            else:
                gt_costs = np.where(finite, 0.0, np.nan)
        else:
            gt_costs = np.zeros(K, dtype=np.float64)

        # -- instruction tokens (pre-computed, stored as raw bytes) ----------
        input_ids  = torch.from_numpy(
            np.frombuffer(rec['input_ids'],    dtype=np.int32).copy()).long()
        attn_mask  = torch.from_numpy(
            np.frombuffer(rec['attention_mask'], dtype=np.int32).copy()).long()

        # -- next_action_instruction tokens ----------------------------------
        nai_input_ids = torch.from_numpy(
            np.frombuffer(rec['nai_input_ids'],      dtype=np.int32).copy()).long()
        nai_attn_mask = torch.from_numpy(
            np.frombuffer(rec['nai_attention_mask'], dtype=np.int32).copy()).long()

        frame_rgb = np.array(pil_img)  # [H, W, 3] uint8 — for visualisation

        # -- counterfactual NAI: noun-swapped distractor instruction ---------
        cf_nai_input_ids  = nai_input_ids
        cf_nai_attn_mask  = nai_attn_mask
        has_cf            = False
        nai_class_name    = ''
        cf_class_name_str = ''
        if self.counterfactual:
            nai_text = rec.get('nai_text', '')
            if nai_text:
                orig_class, m = _extract_mp3d_class(nai_text)
                if orig_class is not None:
                    cf_text, distractor = _gen_counterfactual_text(nai_text, orig_class, m)
                    if cf_text is not None:
                        cf_tok = self.clip_processor(
                            text=cf_text,
                            padding='max_length', truncation=True,
                            max_length=77, return_tensors='pt',
                        )
                        cf_nai_input_ids  = cf_tok['input_ids'].squeeze(0).long()
                        cf_nai_attn_mask  = cf_tok['attention_mask'].squeeze(0).long()
                        has_cf            = True
                        nai_class_name    = orig_class
                        cf_class_name_str = distractor

        # -- per-object class match: which objects belong to the referenced class --
        # mp3d_class_names is non-empty only if LMDB was built with the updated
        # convert_h5_to_lmdb (which calls _node_category per node).  Falls back
        # to all-False tensors so existing LMDBs continue to work unchanged.
        obj_categories = rec.get('mp3d_class_names', [''] * K)
        any_cat = any(c for c in obj_categories)
        class_match = (
            torch.tensor([c == nai_class_name    for c in obj_categories], dtype=torch.bool)
            if nai_class_name    and any_cat
            else torch.zeros(K, dtype=torch.bool)
        )
        cf_class_match = (
            torch.tensor([c == cf_class_name_str for c in obj_categories], dtype=torch.bool)
            if cf_class_name_str and any_cat
            else torch.zeros(K, dtype=torch.bool)
        )

        return {
            'episode_id':            ep_id,
            'frame_idx':             frame_idx,
            'frame_rgb':             frame_rgb,                                       # [H, W, 3] uint8
            'pixel_values':          pixel_values,                                    # [3, 224, 224]
            'masks':                 torch.from_numpy(masks_arr).bool(),              # [K, H, W]
            'gt_costs':              torch.from_numpy(gt_costs.astype(np.float32)),  # [K]
            'input_ids':             input_ids,                                       # [77]
            'attention_mask':        attn_mask,                                       # [77]
            'nai_input_ids':         nai_input_ids,                                   # [77]
            'nai_attention_mask':    nai_attn_mask,                                   # [77]
            'cf_nai_input_ids':      cf_nai_input_ids,                                # [77]
            'cf_nai_attention_mask': cf_nai_attn_mask,                                # [77]
            'has_counterfactual':    has_cf,
            'nai_class_name':        nai_class_name,
            'cf_class_name':         cf_class_name_str,
            'class_match':           class_match,      # [K] bool — referenced class objects
            'cf_class_match':        cf_class_match,   # [K] bool — distractor class objects
        }


def create_lmdb_episode_pathlengths_dataloader(
    lmdb_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    counterfactual: bool = False,
) -> tuple:
    """
    Build train and val DataLoaders from an LMDB database produced by
    convert_h5_to_lmdb().

    Episodes are split at the episode level so no frame leaks between splits.
    Returns (train_loader, val_loader).
    """
    import pickle as _pickle
    import lmdb

    # Enumerate all unique episode IDs from the LMDB key index.
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        all_keys: list = _pickle.loads(txn.get(b"__keys__"))
    env.close()

    all_ep_ids = sorted({k.decode('utf-8').split('/')[0] for k in all_keys})
    rng = random.Random(seed)
    ep_ids = list(all_ep_ids)
    rng.shuffle(ep_ids)
    split_idx = int(len(ep_ids) * (1 - val_split))
    train_ids = ep_ids[:split_idx]
    val_ids   = ep_ids[split_idx:]

    train_ds = LMDBEpisodePathLengthsDataset(lmdb_path, episode_ids=train_ids,
                                             counterfactual=counterfactual)
    val_ds   = LMDBEpisodePathLengthsDataset(lmdb_path, episode_ids=val_ids)

    def _collate(batch: list[dict]) -> dict:
        return {
            "pixel_values":          torch.stack([b["pixel_values"]          for b in batch]),
            "input_ids":             torch.stack([b["input_ids"]             for b in batch]),
            "attention_mask":        torch.stack([b["attention_mask"]        for b in batch]),
            "nai_input_ids":         torch.stack([b["nai_input_ids"]         for b in batch]),
            "nai_attention_mask":    torch.stack([b["nai_attention_mask"]    for b in batch]),
            "cf_nai_input_ids":      torch.stack([b["cf_nai_input_ids"]      for b in batch]),
            "cf_nai_attention_mask": torch.stack([b["cf_nai_attention_mask"] for b in batch]),
            "has_counterfactual":    [b["has_counterfactual"] for b in batch],
            "nai_class_names":       [b["nai_class_name"]     for b in batch],
            "cf_class_names":        [b["cf_class_name"]      for b in batch],
            "class_match_list":      [b["class_match"]        for b in batch],
            "cf_class_match_list":   [b["cf_class_match"]     for b in batch],
            "masks_list":            [b["masks"]    for b in batch],
            "gt_costs_list":         [b["gt_costs"] for b in batch],
            "frame_rgbs":            [b["frame_rgb"] for b in batch],
            "episode_ids":           [b["episode_id"] for b in batch],
            "frame_idxs":            [b["frame_idx"]  for b in batch],
        }

    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader

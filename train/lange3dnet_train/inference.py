"""
LangGeoNet Inference & Visualization.

Usage:
    # Single frame
    python inference.py --checkpoint best_model.pt \
        --image frame.png --masks masks.npy --class_ids class_ids.npy \
        --instruction "Go to the lamp on the nightstand" \
        --output ./output

    # Full dataset evaluation
    python inference.py --checkpoint best_model.pt \
        --data_root ./data --eval_split val

    # Episode inference (all frames, one instruction)
    python inference.py --checkpoint best_model.pt \
        --episode_dir ./data/episode_000 --output ./output
"""

import os
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor  # ← removed BertTokenizer

from .model import LangGeoNetV2


class LangGeoNetPredictor:
    """
    Inference wrapper. Handles preprocessing, prediction, and costmap generation.
    """

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        # Build model — no num_classes needed
        self.model = LangGeoNetV2(
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            clip_model_name=cfg["clip_model"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # CLIP processor handles both image and text tokenization
        self.clip_processor = CLIPProcessor.from_pretrained(cfg["clip_model"])

        print(f"[LangGeoNetPredictor] Loaded epoch {ckpt['epoch']} "
              f"(val MAE={ckpt.get('best_val_mae', '?'):.4f}) on {self.device}")

    # ----------------------------------------------------------
    # Core prediction
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict_frame(self, image, masks, instruction):
        """
        Args:
            image:       PIL.Image or np.ndarray [H, W, 3]
            masks:       np.ndarray [K, H, W]
            instruction: str

        Returns:
            distances: np.ndarray [K]
            costmap:   np.ndarray [H, W]
            attn_map:  np.ndarray [K, L_actual] or None
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        enc = self.clip_processor(
            images=image,
            text=instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        pixel_values   = enc["pixel_values"].to(self.device)   # [1, 3, 224, 224]
        input_ids      = enc["input_ids"].to(self.device)       # [1, 77]
        attention_mask = enc["attention_mask"].to(self.device)  # [1, 77]

        # Filter out empty masks (all-zero) — these cause spurious extra predictions
        valid = masks.any(axis=(1, 2))          # [K] bool
        masks = masks[valid]                    # [K_valid, H, W]

        if masks.shape[0] == 0:
            H, W = masks.shape[1], masks.shape[2] if masks.ndim == 3 else (image.height, image.width)
            return np.array([]), np.ones((H, W), dtype=np.float32), None

        masks_t = torch.from_numpy(masks.astype(bool)).to(self.device)  # [K_valid, H, W]

        predictions, attn_weights_all = self.model(
            images=pixel_values,
            masks_list=[masks_t],
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        distances = predictions[0].cpu().numpy()  # [K]

        K, H, W = masks.shape
        costmap = np.ones((H, W), dtype=np.float32)
        for k in range(K):
            costmap[masks[k] > 0] = distances[k]

        attn_map = None
        if attn_weights_all:
            last = attn_weights_all[-1]
            # L_actual = non-padding tokens; skip position 0 (CLIP SOT sink)
            # and the final EOS token so only content tokens are shown.
            L_total = int(attention_mask.sum().item())  # includes SOT + content + EOS
            # content tokens are positions 1 .. L_total-2  (exclude SOT=0 and EOS=L_total-1)
            content_start = 1
            content_end   = max(content_start + 1, L_total - 1)  # at least 1 token
            attn_map = last[0, :K, content_start:content_end].cpu().numpy()  # [K, L_content]

        return distances, costmap, attn_map

    # ----------------------------------------------------------
    # Episode-level
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict_episode(self, episode_dir):
        with open(os.path.join(episode_dir, "instruction.txt")) as f:
            instruction = f.read().strip()

        frame_dirs = sorted(glob.glob(os.path.join(episode_dir, "frame_*")))
        results = []

        for frame_dir in frame_dirs:
            rgb_path   = os.path.join(frame_dir, "rgb.png")
            masks_path = os.path.join(frame_dir, "masks.npy")

            if not all(os.path.exists(p) for p in [rgb_path, masks_path]):  # ← removed cids_path check
                continue

            image = Image.open(rgb_path).convert("RGB")
            masks = np.load(masks_path)
            # ← removed: class_ids = np.load(cids_path)

            distances, costmap, attn = self.predict_frame(image, masks, instruction)  # ← removed class_ids
            results.append({
                "frame_dir": frame_dir,
                "distances": distances,
                "costmap":   costmap,
                "attn_map":  attn,
            })

        return instruction, results

    # ----------------------------------------------------------
    # H5 inference
    # ----------------------------------------------------------

    @staticmethod
    def _decode_rle_mask(rle: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        Decode a 1-D F-major alternating skip/set RLE into a (H, W) bool array.

        Runs at even positions (0, 2, …) are background (skip);
        runs at odd positions (1, 3, …) are foreground (set).
        """
        flat = np.zeros(H * W, dtype=bool)
        pos  = 0
        for i, count in enumerate(rle):
            if i % 2 == 1:
                flat[pos: pos + count] = True
            pos += count
        return np.reshape(flat, (H, W), order="F")

    @torch.no_grad()
    def predict_from_h5(
        self,
        h5_path: str,
        episode_id: str,
        frame_idx: int,
        instruction: str,
        base_dir: str = None,
    ):
        """
        Load a single frame directly from an HDF5 file and run inference.

        HDF5 key format : ``{episode_id}_{frame_idx}``
        RGB image path  : ``{base_dir}/trajectories/{episode_id}/images/{frame_idx:05d}.png``

        ``base_dir`` defaults to the directory that contains the H5 file.

        Args:
            h5_path    : path to the HDF5 costmap file.
            episode_id : episode folder name as it appears in the H5 key
                         (e.g. ``"1S7LAXRdDqK_0000000_plant_42_"``).
            frame_idx  : 0-based frame index within the episode.
            instruction: navigation instruction string.
            base_dir   : root directory that contains the ``trajectories/``
                         sub-folder.  Defaults to the H5 file's parent dir.

        Returns:
            image_pil      : PIL.Image (RGB)
            masks          : np.ndarray [K_valid, H, W]  bool
            pred_distances : np.ndarray [K_valid]  model predictions in [0, 1]
            gt_distances   : np.ndarray [K_valid]  GT PL scores normalised to [0, 1]
            costmap        : np.ndarray [H, W]
            attn_map       : np.ndarray [K_valid, L_content] or None
        """
        import h5py

        ep_folder = episode_id
        h5_key    = f"{ep_folder}_{frame_idx}"

        with h5py.File(h5_path, "r") as h5:
            if h5_key not in h5:
                available = sorted(h5.keys())[:10]
                raise KeyError(
                    f"Key '{h5_key}' not found in {h5_path}. "
                    f"First available keys: {available}"
                )
            grp  = h5[h5_key]
            H, W = int(grp["size"][0]), int(grp["size"][1])

            # Decode RLE masks
            masks_grp = grp["img_masks"]
            n_masks   = len(masks_grp)
            masks_raw = np.stack(
                [self._decode_rle_mask(masks_grp[str(i)][()], H, W)
                 for i in range(n_masks)],
                axis=0,
            )  # [K, H, W]  bool

            # GT PL scores — min-max normalise per frame
            pls = grp["img_pls"][()].astype(np.float32)  # [K]

        pls_min, pls_max = float(pls.min()), float(pls.max())
        if pls_max - pls_min > 1e-6:
            gt_distances_raw = (pls - pls_min) / (pls_max - pls_min)
        else:
            gt_distances_raw = np.zeros_like(pls)

        # Locate RGB image: {base_dir}/trajectories/{ep_folder}/images/{frame:05d}.png
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(h5_path))
        img_path = os.path.join(
            base_dir, "trajectories", ep_folder, "images", f"{frame_idx:05d}.png"
        )
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"RGB image not found: {img_path}")
        image_pil = Image.open(img_path).convert("RGB")

        # Run inference (predict_frame filters empty masks internally)
        pred_distances, costmap, attn_map = self.predict_frame(
            image_pil, masks_raw, instruction
        )

        # Align GT with the same valid-mask filter used inside predict_frame
        valid        = masks_raw.any(axis=(1, 2))
        gt_distances = gt_distances_raw[valid]
        masks_valid  = masks_raw[valid]

        return image_pil, masks_valid, pred_distances, gt_distances, costmap, attn_map

    # ----------------------------------------------------------
    # Batch prediction
    # ----------------------------------------------------------

    @torch.no_grad()
    def predict_batch(self, images, masks_list, instruction):  # ← removed class_ids_list
        B = len(images)

        pil_images = [
            Image.fromarray(img) if isinstance(img, np.ndarray) else img
            for img in images
        ]

        enc = self.clip_processor(
            images=pil_images,
            text=[instruction] * B,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        pixel_values   = enc["pixel_values"].to(self.device)
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # Filter empty masks per batch item
        filtered_masks = []
        valid_indices = []
        for m in masks_list:
            valid = m.any(axis=(1, 2)) if isinstance(m, np.ndarray) else m.any(dim=(1, 2))
            filtered_masks.append(m[valid])
            valid_indices.append(valid)

        masks_t = [torch.from_numpy(m.astype(bool)).to(self.device) for m in filtered_masks]

        predictions, _ = self.model(
            images=pixel_values,
            masks_list=masks_t,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        all_distances, all_costmaps = [], []
        for b in range(B):
            dists = predictions[b].cpu().numpy()
            m = filtered_masks[b]               # already filtered
            K, H, W = m.shape
            costmap = np.ones((H, W), dtype=np.float32)
            for k in range(K):
                costmap[m[k] > 0] = dists[k]
            all_distances.append(dists)
            all_costmaps.append(costmap)

        return all_distances, all_costmaps


# ==============================================================
# Visualization
# ==============================================================

def visualize_costmap(image, costmap, save_path=None, title="WayObject Costmap"):
    """Overlay costmap on RGB. Green=close, Red=far."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if isinstance(image, Image.Image):
        image = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    cmap = cm.get_cmap("RdYlGn_r")
    colored = cmap(costmap)[:, :, :3]
    axes[1].imshow(colored)
    axes[1].set_title(title)
    axes[1].axis("off")
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=axes[1], fraction=0.046)

    overlay = 0.5 * (image / 255.0) + 0.5 * colored
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_per_object(image, masks, distances, class_names=None, save_path=None):
    """Bar chart of per-object predicted geodesic + overlay of top/bottom objects."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if isinstance(image, Image.Image):
        image = np.array(image)

    K = len(distances)
    labels = [class_names[k] if class_names else f"obj_{k}" for k in range(K)]
    order = np.argsort(distances)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart sorted by distance
    colors = cm.get_cmap("RdYlGn_r")(distances[order])
    axes[0].barh(range(K), distances[order], color=colors)
    axes[0].set_yticks(range(K))
    axes[0].set_yticklabels([labels[i] for i in order], fontsize=8)
    axes[0].set_xlabel("Predicted Geodesic Distance (normalized)")
    axes[0].set_title("Per-Object Distances")
    axes[0].set_xlim(0, 1)

    # Image with closest (green) and farthest (red) object highlighted
    overlay = image.copy().astype(np.float32) / 255.0
    closest_k = order[0]
    farthest_k = order[-1]

    green_overlay = np.zeros_like(overlay)
    green_overlay[:, :, 1] = 1.0
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 0] = 1.0

    mask_close = masks[closest_k] > 0
    mask_far = masks[farthest_k] > 0

    overlay[mask_close] = 0.5 * overlay[mask_close] + 0.5 * green_overlay[mask_close]
    overlay[mask_far] = 0.5 * overlay[mask_far] + 0.5 * red_overlay[mask_far]

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title(f"Closest: {labels[closest_k]} ({distances[closest_k]:.3f})  |  "
                       f"Farthest: {labels[farthest_k]} ({distances[farthest_k]:.3f})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_attention(instruction, attn_weights, object_labels, save_path=None):
    """Heatmap: which instruction tokens each object attends to."""
    import matplotlib.pyplot as plt

    # ← use CLIP tokenizer instead of BertTokenizer
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    tokens = tokenizer.tokenize(instruction)

    K, L = attn_weights.shape
    L_show = min(L, len(tokens))

    fig, ax = plt.subplots(figsize=(max(12, L_show * 0.5), max(3, K * 0.4)))
    im = ax.imshow(attn_weights[:, :L_show], cmap="Blues", aspect="auto")
    ax.set_xticks(range(L_show))
    ax.set_xticklabels(tokens[:L_show], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(object_labels, fontsize=9)
    ax.set_title("Cross-Attention: Object x Instruction Token")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ==============================================================
# Dataset-level Evaluation & CLI
# ==============================================================

def evaluate_dataset(predictor, data_root, split="val"):
    """
    Run evaluation on an entire dataset split.

    Returns:
        metrics dict
    """
    from dataset import LangGeoNetDataset
    from train import compute_metrics

    ds = LangGeoNetDataset(data_root, split=split)
    all_preds, all_gts = [], []

    for i in range(len(ds)):
        sample = ds[i]
        frame_dir = sample["frame_dir"]
        ep_dir = sample["episode_dir"]

        image = Image.open(os.path.join(frame_dir, "rgb.png")).convert("RGB")
        masks = sample["masks"].numpy()
        gt = sample["geodesic_distances"].numpy()

        with open(os.path.join(ep_dir, "instruction.txt")) as f:
            instruction = f.read().strip()

        dists, _, _ = predictor.predict_frame(image, masks, instruction)  # ← removed class_ids

        all_preds.append(dists)
        all_gts.append(gt)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)}")

    metrics = compute_metrics(all_preds, all_gts)

    print(f"\n{'='*40}")
    print(f"Evaluation on {split} ({len(ds)} frames)")
    print(f"{'='*40}")
    print(f"  MAE:              {metrics['mae']:.4f}")
    print(f"  RMSE:             {metrics['rmse']:.4f}")
    print(f"  Ranking Accuracy: {metrics['ranking_accuracy']:.4f}")
    print(f"  Spearman:         {metrics['spearman']:.4f}")
    print(f"  acc@0.05:         {metrics['acc@0.05']:.4f}")
    print(f"  acc@0.10:         {metrics['acc@0.10']:.4f}")
    print(f"  acc@0.20:         {metrics['acc@0.20']:.4f}")

    return metrics


# ==============================================================
# CLI
# ==============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LangGeoNet Inference")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--device", default=None)

    # Mode 1: single frame
    p.add_argument("--image", default=None, help="Path to RGB image")
    p.add_argument("--masks", default=None, help="Path to masks.npy [K, H, W]")
    p.add_argument("--instruction", default=None, help="Navigation instruction string")

    # Mode 2: episode
    p.add_argument("--episode_dir", default=None, help="Path to episode directory")

    # Mode 3: dataset eval
    p.add_argument("--data_root", default=None, help="Path to dataset root")
    p.add_argument("--eval_split", default="val")

    p.add_argument("--output", default="./output", help="Output directory")
    a = p.parse_args()

    predictor = LangGeoNetPredictor(a.checkpoint, device=a.device)
    os.makedirs(a.output, exist_ok=True)

    # ---- Mode 1: Single frame ----
    if a.image and a.masks and a.instruction:
        image = Image.open(a.image).convert("RGB")
        masks = np.load(a.masks)
        # ← removed: class_ids = np.load(a.class_ids) ...

        dists, costmap, attn = predictor.predict_frame(image, masks, a.instruction)  # ← removed class_ids

        print(f"\nInstruction: {a.instruction}")
        print(f"Objects: {masks.shape[0]}")
        print(f"\nPredicted geodesic distances (sorted by distance):")
        for k in np.argsort(dists):
            print(f"  obj_{k:2d}: {dists[k]:.4f}")  # ← removed class_ids[k] from print

        np.save(os.path.join(a.output, "distances.npy"), dists)
        np.save(os.path.join(a.output, "costmap.npy"), costmap)

        visualize_costmap(image, costmap, os.path.join(a.output, "costmap.png"))
        visualize_per_object(image, masks, dists, save_path=os.path.join(a.output, "per_object.png"))
        if attn is not None:
            labels = [f"obj_{k}" for k in range(masks.shape[0])]
            visualize_attention(a.instruction, attn, labels,
                                save_path=os.path.join(a.output, "attention.png"))

    # ---- Mode 2: Episode ----
    elif a.episode_dir:
        instruction, results = predictor.predict_episode(a.episode_dir)
        print(f"\nInstruction: {instruction}")
        print(f"Frames: {len(results)}")

        for i, r in enumerate(results):
            print(f"\n  Frame {i}: {r['distances'].shape[0]} objects, "
                  f"min_dist={r['distances'].min():.4f}, max_dist={r['distances'].max():.4f}")

            # Save per-frame outputs
            frame_out = os.path.join(a.output, f"frame_{i:03d}")
            os.makedirs(frame_out, exist_ok=True)
            np.save(os.path.join(frame_out, "distances.npy"), r["distances"])
            np.save(os.path.join(frame_out, "costmap.npy"), r["costmap"])

            # Visualize
            rgb_path = os.path.join(r["frame_dir"], "rgb.png")
            if os.path.exists(rgb_path):
                img = Image.open(rgb_path).convert("RGB")
                visualize_costmap(img, r["costmap"],
                                  os.path.join(frame_out, "costmap.png"),
                                  title=f"Frame {i} Costmap")

    # ---- Mode 3: Dataset eval ----
    elif a.data_root:
        evaluate_dataset(predictor, a.data_root, a.eval_split)

    else:
        print("Provide one of:")
        print("  --image + --masks + --instruction  (single frame)")
        print("  --episode_dir                      (full episode)")
        print("  --data_root                        (dataset eval)")

import numpy as np
import matplotlib
import h5py
from typing import Any, Dict

import torch
import torch.nn.functional as F


class TopoPaths:
    def __init__(
        self,
        maxRank=200,
        dims=16,
        w=160,
        h=120,
        precomputed_filename=None,
        pl_perturb_ratio=0.0,
        pl_perturb_type="max_val",
        mask_crop_ratio=1.0,
        use_mask_grad=False,
    ):
        self.w, self.h = w, h
        self.pl_outlier_value = 99
        self.pl_perturb_ratio = pl_perturb_ratio
        self.pl_perturb_type = pl_perturb_type
        self.mask_crop_ratio = mask_crop_ratio
        self.use_mask_grad = use_mask_grad
        self.precomputed_filename = precomputed_filename

        if dims == 1:
            self.rank_enc = np.arange(maxRank + 1).astype(float).reshape(-1, 1)
        else:
            self.rank_enc = generate_positional_encodings(maxRank, dims)
        self.default_enc = (
            np.ones((dims, self.h // 2, self.w // 2)) * self.rank_enc[0][:, None, None]
        )  # (D,H,W)

    def perturb_mask_pls(self, pls):
        perturbed_pls = pls.copy()
        if self.pl_perturb_ratio == 0:
            return pls

        num_masks_to_perturb = int(len(pls) * self.pl_perturb_ratio)
        indices_to_perturb = np.random.choice(
            len(pls), num_masks_to_perturb, replace=False
        )

        if self.pl_perturb_type == "max_val":
            perturbed_pls[indices_to_perturb] = self.pl_outlier_value
        elif self.pl_perturb_type == "rand_from_inliers":
            inliers = pls[pls < self.pl_outlier_value]
            perturbed_pls[indices_to_perturb] = np.random.choice(
                inliers, num_masks_to_perturb
            )
        else:
            raise ValueError(f"Invalid pl_perturb_type {self.pl_perturb_type}")

        return perturbed_pls

    def get_topo_path(self, trajName, imgIdx):
        key = f"{trajName}_{imgIdx}"
        with h5py.File(self.precomputed_filename, "r") as masks_pls_dict:
            if key not in masks_pls_dict:
                return self.create_input(None, None)
            else:
                key_data = masks_pls_dict[key]
                img_size = key_data["size"][()]
                img_masks = key_data["img_masks"]
                # read masks in order
                img_masks = [
                    {"size": img_size, "counts": img_masks[f"{mi}"][()]}
                    for mi in range(len(img_masks.keys()))
                ]
                img_pls = key_data["img_pls"][()]
                img_pls = self.perturb_mask_pls(img_pls)
            inputData = self.create_input(img_pls, img_masks, convertMask=True)
        return inputData

    def create_input(self, pls, masks, convertMask=False):
        img_enc = self.default_enc
        plWtColorImg = np.zeros((3, img_enc.shape[1], img_enc.shape[2]))
        if pls is None or masks is None:
            pass
        else:
            pls = normalize_pls(pls, outlier_value=self.pl_outlier_value)
            if convertMask:
                masks = np.array([rle_to_mask(m) for m in masks])
            masks = masks.transpose([1, 2, 0])[::2, ::2]  # (H,W,D)
            # for topological graphs of masks 320, 240
            if masks.shape[0] != self.h // 2:
                masks = masks[::2, ::2]
            if masks.shape[0] != self.h // 2:
                raise ValueError(
                    f"masks shape {masks.shape} does not match expected shape ({self.h//2},{self.w//2})"
                )

            # randomly crop masks
            if self.mask_crop_ratio != 1.0:
                masks = random_crop_and_reshape_torch(masks, self.mask_crop_ratio)

            deno = masks.sum(-1)
            deno[deno == 0] = 1
            colors, norm = value2color(pls, cmName="winter")
            plWtColorImg = (masks / deno[:, :, None] @ colors).transpose(2, 0, 1)
            enc = self.rank_enc[pls.astype(int)]
            img_enc = (masks @ enc).transpose(2, 0, 1)
        if self.use_mask_grad:
            grad = get_masks_gradient(masks.transpose(2, 0, 1))
            img_enc = np.concatenate([img_enc, grad[None]], axis=0)
        return img_enc, plWtColorImg

    def build_differentiable_goal(
        self,
        lang_pred_list: list,
        gnm_masks: "torch.Tensor",
        K_list: list,
        device: "torch.device",
    ) -> "torch.Tensor":
        """
        Build a differentiable goal encoding for the GNM from LangGeoNetV2
        raw logits, preserving gradient flow.

        Parameters
        ----------
        lang_pred_list : list[B] of Tensor[K_b]
            Raw logits from LangGeoNetV2 (in the computation graph).
        gnm_masks : Tensor[B, K_max, H//2, W//2] float32
            Pre-downsampled binary masks (padded with zeros to K_max).
        K_list : list[B] of int
            Actual number of objects per sample (unpadded).
        device : torch.device

        Returns
        -------
        goal_enc : Tensor[B, 3 + dims, H//2, W//2]
            Concatenation of:
              • channels 0–2 : detached visualisation (cost heat-map overlay)
              • channels 3– : differentiable goal encoding (grad flows through)
        """
        dims = self.rank_enc.shape[1]
        near_enc = torch.tensor(
            self.rank_enc[0], dtype=torch.float32, device=device
        )
        far_enc = torch.tensor(
            self.rank_enc[-1], dtype=torch.float32, device=device
        )
        results = []
        for b, (logits, K) in enumerate(zip(lang_pred_list, K_list)):
            masks_b = gnm_masks[b, :K].to(device)   # [K, Hh, Wh]
            Hh, Wh = masks_b.shape[1], masks_b.shape[2]

            if K == 0:
                img_enc = torch.zeros(dims, Hh, Wh, device=device)
                viz = torch.zeros(3, Hh, Wh, device=device)
                results.append(torch.cat([viz, img_enc], dim=0))
                continue

            lo = logits.min()
            costs = (logits - lo) / (logits.max() - lo + 1e-8)  # [K] ∈ [0,1]
            #   enc_k = (1 - cost_k) * near_enc + cost_k * far_enc   [K, dims]
            weights_near = (1.0 - costs).unsqueeze(1)
            weights_far  = costs.unsqueeze(1)
            obj_enc = weights_near * near_enc + weights_far * far_enc

            img_enc = torch.einsum("kd,khw->dhw", obj_enc, masks_b)

            # Detached visualisation (winter-like: near=blue, far=red).
            costs_d = costs.detach()
            r_ch = (masks_b * costs_d[:, None, None]).sum(0).clamp(0, 1)
            g_ch = (masks_b * (1.0 - costs_d)[:, None, None]).sum(0).clamp(0, 1)
            b_ch = torch.zeros_like(r_ch)
            viz = torch.stack([r_ch, g_ch, b_ch], dim=0).detach()

            results.append(torch.cat([viz, img_enc], dim=0))

        return torch.stack(results, dim=0)

    def create_input_from_predictions(
        self,
        pred_logits: "torch.Tensor",
        masks: "torch.Tensor",
    ):
        """
        Detached wrapper: convert predicted logits + half-res masks to the
        same ``(img_enc, plWtColorImg)`` tuple as ``create_input()``, for
        inference / visualisation without gradient tracking.

        Parameters
        ----------
        pred_logits : Tensor[K]  raw logits (min-max normalised to [0,1] for goal blending)
        masks       : Tensor[K, H//2, W//2] float32

        Returns
        -------
        img_enc      : ndarray [dims, H//2, W//2]
        plWtColorImg : ndarray [3, H//2, W//2]
        """
        logits_t = pred_logits.detach().cpu().float()
        lo = logits_t.min()
        costs = ((logits_t - lo) / (logits_t.max() - lo + 1e-8)).numpy()  # [K] ∈ [0,1]
        pls_approx = (costs * (self.rank_enc.shape[0] - 1)).astype(int)
        pls_approx = np.clip(pls_approx, 0, self.rank_enc.shape[0] - 1)
        masks_np = masks.detach().cpu().numpy()  # [K, Hh, Wh]
        # Call existing create_input with preconverted masks.
        # pls_approx passed directly (already int indices).
        return self.create_input(pls_approx, masks_np, convertMask=False)


def normalize_pls(pls, scale_factor=100, outlier_value=99, new_max_val=None):

    outliers = pls >= outlier_value
    # if all are outliers, set them to zero
    if sum(outliers) == len(pls):
        return np.zeros_like(pls)

    min_val = pls.min()
    if new_max_val is None:
        new_max_val = pls[~outliers].max() + 1
    else:
        assert (
            new_max_val > pls[~outliers].max()
        ), f"{new_max_val} <= {pls[~outliers].max()}"

    # else set outliers to max value of inliers + 1
    # so that when normalized, they are set to 0
    if sum(outliers) > 0:
        pls[outliers] = new_max_val

    # include a dummy value to ensure that new_max_val -> 0 after norm 'even for inliers'
    pls = np.concatenate([pls, [new_max_val]])

    # normalize so that outliers are set to 0; inliers \in (0, scale_factor]
    pls = scale_factor * (new_max_val - pls) / (new_max_val - min_val)
    return pls[:-1]


def get_masks_gradient(masks):
    """
    masks: [N, H, W]
    """
    dx = np.zeros_like(masks)
    dy = dx.copy()
    masks_f = masks.copy().astype(float)

    dx[:, 1:, :] = abs(masks_f[:, 1:, :] - masks_f[:, :-1, :])
    dy[:, :, 1:] = abs(masks_f[:, :, 1:] - masks_f[:, :, :-1])

    grad = ((dx + dy).sum(0).astype(bool)).astype(float)
    return grad


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def generate_positional_encodings(max_rank, d_model):
    """
    Generate positional encodings for ranks from 0 to max_rank.

    Parameters:
    max_rank (int): The maximum rank for which to generate encodings.
    d_model (int): The dimensionality of the encoding vector.

    Returns:
    dict: A dictionary with ranks as keys and positional encoding vectors as values.

    # Example usage:
    max_rank = 5
    d_model = 16
    positional_encodings = generate_positional_encodings(max_rank, d_model)
    print(positional_encodings)
    """
    reduceLater = False
    reduceLaterDim = d_model
    if d_model < 4:
        reduceLater = True
        d_model = 4
    ranks = np.arange(max_rank + 1)
    encodings = np.zeros((max_rank + 1, d_model))

    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    encodings[:, 0::2] = np.sin(ranks[:, np.newaxis] * div_term)
    encodings[:, 1::2] = np.cos(ranks[:, np.newaxis] * div_term)
    if reduceLater:
        encodings = encodings[:, :reduceLaterDim]
    return encodings


def random_crop_and_reshape_torch(masks_np, mask_crop_ratio=0.8):
    """
    Randomly crops binary masks with separate max crop % for height and width,
    then resizes back to original size. Aspect ratio can change.

    Args:
        masks_np (np.ndarray): Binary masks of shape (H, W, D)
        mask_crop_ratio (float): Max crop percent for height and width.

    Returns:
        np.ndarray: Resized masks of shape (H, W, D)
    """
    H, W, D = masks_np.shape

    if not (0 < mask_crop_ratio <= 1.0):
        raise ValueError("Crop percentages must be in (0, 1]")

    # Random crop percentages (from [max_pct, 1.0])
    crop_pct_h = np.random.uniform(mask_crop_ratio, 1.0)
    crop_pct_w = np.random.uniform(mask_crop_ratio, 1.0)

    crop_h = int(H * crop_pct_h)
    crop_w = int(W * crop_pct_w)

    top = np.random.randint(0, H - crop_h + 1)
    left = np.random.randint(0, W - crop_w + 1)

    # Torch: (D, 1, H, W)
    masks = torch.from_numpy(masks_np).permute(2, 0, 1).unsqueeze(1).float()

    # Crop and resize
    cropped = masks[:, :, top : top + crop_h, left : left + crop_w]
    resized = F.interpolate(cropped, size=(H, W), mode="nearest")

    return resized.squeeze(1).permute(1, 2, 0).to(torch.uint8).numpy()


def value2color(values, vmin=None, vmax=None, cmName="jet"):
    cmapPaths = matplotlib.cm.get_cmap(cmName)
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = np.array([cmapPaths(norm(value))[:3] for value in values])
    return colors, norm

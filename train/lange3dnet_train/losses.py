"""
Loss functions for LangGeoNet.

Combined loss:
    L_total = L_regression + lambda_rank * L_ranking + lambda_si * L_scale_invariant

    1. Geodesic Regression Loss   (SmoothL1) - primary distance supervision
    2. Ordinal Ranking Loss       (margin)   - correct ordering of objects
    3. Scale-Invariant Log Loss   (Eigen)    - handles varying scene scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicRegressionLoss(nn.Module):
    """SmoothL1 between predicted and GT normalized geodesic distances."""

    def __init__(self, beta=0.05):
        super().__init__()
        self.beta = beta

    def forward(self, predictions, targets):
        total_loss = 0.0
        total_objects = 0

        for pred, gt in zip(predictions, targets):
            if pred.numel() == 0:
                continue
            # Min-max normalise GT to [0, 1] so sentinel values (e.g. 1e6 for
            # unreachable objects stored in raw traj_data.pkl) do not explode
            # the regression target.  The normalised sentinel maps to 1.0,
            # correctly representing the highest-cost object.
            g_min, g_max = gt.min(), gt.max()
            if (g_max - g_min).abs() > 1e-8:
                gt_norm = (gt - g_min) / (g_max - g_min)
            else:
                gt_norm = torch.zeros_like(gt)
            # Predictions are raw logits; apply sigmoid before regression.
            total_loss += F.smooth_l1_loss(
                torch.sigmoid(pred), gt_norm, beta=self.beta, reduction='sum')
            total_objects += pred.numel()

        if total_objects == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / total_objects


class OrdinalRankingLoss(nn.Module):
    """
    For all pairs (i,j) where GT d_i < d_j, enforce pred_i < pred_j with margin.
    Uses hard negative mining for efficiency.
    """

    def __init__(self, margin=0.3, hard_fraction=0.5):
        super().__init__()
        self.margin = margin
        self.hard_fraction = hard_fraction

    def forward(self, predictions, targets):
        total_loss = 0.0
        total_pairs = 0

        for pred, gt in zip(predictions, targets):
            K = pred.shape[0]
            if K < 2:
                continue

            # Min-max normalise predictions to [0, 1] so pairwise differences
            # are bounded and the loss scale is comparable to the GT scale.
            p_min, p_max = pred.min(), pred.max()
            if (p_max - p_min).abs() > 1e-8:
                pred_norm = (pred - p_min) / (p_max - p_min)
            else:
                pred_norm = torch.zeros_like(pred)

            # GT is already min-max normalised [0, 1] by the dataset.
            g_min, g_max = gt.min(), gt.max()
            if (g_max - g_min).abs() > 1e-8:
                gt_norm = (gt - g_min) / (g_max - g_min)
            else:
                gt_norm = torch.zeros_like(gt)

            # Pairwise differences on normalised values
            pred_diff = pred_norm.unsqueeze(1) - pred_norm.unsqueeze(0)   # [K,K]: pred_i - pred_j
            gt_diff = gt_norm.unsqueeze(1) - gt_norm.unsqueeze(0)         # [K,K]: gt_i - gt_j
            valid = (gt_diff < -1e-6)  # gt_i < gt_j -> pred_i should be < pred_j

            if valid.sum() == 0:
                continue

            # Margin loss: max(0, pred_i - pred_j + margin) where gt_i < gt_j
            pair_losses = F.relu(pred_diff + self.margin)
            valid_losses = pair_losses[valid]

            violated = valid_losses[valid_losses > 0]
            if violated.numel() == 0:
                continue
            # Hard negative mining: topk over violated pairs only
            n_hard = max(1, int(violated.numel() * self.hard_fraction))
            if violated.numel() > n_hard:
                topk, _ = violated.topk(n_hard)
                total_loss += topk.sum()
                total_pairs += n_hard
            else:
                total_loss += violated.sum()
                total_pairs += violated.numel()

        if total_pairs == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / total_pairs


class ScaleInvariantLogLoss(nn.Module):
    """
    Scale-invariant loss in log space (Eigen et al., 2014).
    Allows uniform scale shift without penalty.
    """

    def __init__(self, variance_weight=0.5, eps=1e-6):
        super().__init__()
        self.lam = variance_weight
        self.eps = eps

    def forward(self, predictions, targets):
        total_loss = 0.0
        count = 0

        for pred, gt in zip(predictions, targets):
            if pred.numel() == 0:
                continue

            # Min-max normalise GT to [0, 1] to guard against sentinel values.
            g_min, g_max = gt.min(), gt.max()
            if (g_max - g_min).abs() > 1e-8:
                gt_norm = (gt - g_min) / (g_max - g_min)
            else:
                gt_norm = torch.zeros_like(gt)

            # Predictions are raw logits; apply sigmoid before log-space loss.
            log_pred = torch.log(torch.sigmoid(pred).clamp(min=self.eps))
            log_gt = torch.log(gt_norm.clamp(min=self.eps))
            diff = log_pred - log_gt

            total_loss += (diff ** 2).mean() - self.lam * (diff.mean()) ** 2
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / count


class PredictionDiversityLoss(nn.Module):
    """
    Penalise near-constant predictions within a single frame.

    Computes the std-dev of sigmoid(predictions) across all valid objects in
    each frame and penalises if it falls below `min_std`.  This provides a
    direct gradient signal to break the same-prediction collapse mode.
    """

    def __init__(self, min_std: float = 0.10):
        super().__init__()
        self.min_std = min_std

    def forward(self, predictions) -> torch.Tensor:
        loss  = torch.tensor(0.0, device=predictions[0].device)
        count = 0
        for pred in predictions:
            if pred.shape[0] < 2:
                continue
            std   = torch.sigmoid(pred).std()
            loss  = loss + F.relu(self.min_std - std)
            count += 1
        return loss / max(count, 1)


class ListNetRankingLoss(nn.Module):
    """
    ListNet top-1 approximation: cross-entropy between the predicted and
    ground-truth softmax score distributions.

    Works on raw logits — no sigmoid needed.  Lower GT value = higher
    probability (gt scores are costs, so we invert them).
    """

    def forward(self, predictions, targets):
        total_loss = 0.0
        count = 0

        for pred, gt in zip(predictions, targets):
            K = pred.shape[0]
            if K < 2:
                continue

            # Min-max normalise predictions to [0, 1] before computing
            # log-softmax so scale is consistent with the GT distribution.
            p_min, p_max = pred.min(), pred.max()
            if (p_max - p_min).abs() > 1e-8:
                pred_norm = (pred - p_min) / (p_max - p_min)
            else:
                pred_norm = torch.zeros_like(pred)

            # GT already in [0, 1]; normalise defensively.
            g_min, g_max = gt.min(), gt.max()
            if (g_max - g_min).abs() > 1e-8:
                gt_norm = (gt - g_min) / (g_max - g_min)
            else:
                gt_norm = torch.zeros_like(gt)

            # Invert costs so lower-cost objects get higher probability mass.
            true_probs = F.softmax(-gt_norm,    dim=0)   # [K]
            pred_log   = F.log_softmax(pred_norm, dim=0) # [K]
            total_loss -= (true_probs * pred_log).sum()
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=predictions[0].device, requires_grad=True)
        return total_loss / count


class ObjectGroundingContrastiveLoss(nn.Module):
    """
    Directional contrastive ranking loss for object grounding.

    Given a [K] bool ``class_match`` mask indicating which objects in a frame
    belong to the class mentioned in the instruction, enforces:

        pred[k+] < pred[k-] + margin   for all (k+, k-) pairs

    where k+ is a matched (referenced) object and k- is an unmatched object.

    Works on raw logits. A frame is skipped silently when class_match has no
    positives or no negatives (e.g. referenced class not visible in this frame,
    or LMDB built without mp3d_class_names).
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def _rank(self, preds: list, class_match_list: list) -> torch.Tensor:
        device = preds[0].device if preds else torch.device('cpu')
        loss   = torch.tensor(0.0, device=device)
        n      = 0
        for pred, match in zip(preds, class_match_list):
            if match is None or pred.numel() == 0:
                continue
            match   = match.to(pred.device)
            pos_idx = match.nonzero(as_tuple=True)[0]      # referenced objects [P]
            neg_idx = (~match).nonzero(as_tuple=True)[0]   # all others          [N]
            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue
            p     = pred[pos_idx].unsqueeze(1)   # [P, 1]
            q     = pred[neg_idx].unsqueeze(0)   # [1, N]
            loss += F.relu(p - q + self.margin).mean()
            n    += 1
        return loss / max(n, 1)

    def forward(self, preds: list, class_match_list: list) -> torch.Tensor:
        return self._rank(preds, class_match_list)


class CounterfactualContrastiveLoss(nn.Module):
    """
    Enforce that cost predictions diverge when the object noun in the NAI is
    replaced with a distractor class (counterfactual instruction).

    For frames where has_cf=True, the element-wise MSE between
    sigmoid(orig_preds) and sigmoid(cf_preds) must exceed `margin`.
    If it already does, no penalty is applied.

    Gradient flows only through orig_preds (cf_preds are computed with
    torch.no_grad in train_one_epoch), so the model learns to make its
    predictions under the real instruction more sensitive to the object noun.
    """

    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        orig_preds: list,
        cf_preds:   list,
        has_cf:     list,   # list[bool]
    ) -> torch.Tensor:
        device = orig_preds[0].device if orig_preds else torch.device('cpu')
        loss = torch.tensor(0.0, device=device)
        n = 0
        for orig, cf, hc in zip(orig_preds, cf_preds, has_cf):
            if not hc or orig.numel() < 1:
                continue
            orig_sig = torch.sigmoid(orig.float())
            cf_sig   = torch.sigmoid(cf.float())
            mse      = F.mse_loss(orig_sig, cf_sig)
            loss    += F.relu(self.margin - mse)
            n       += 1
        return loss / max(n, 1)


class LangGeoNetLoss(nn.Module):
    """
    Combined loss:
        L = L_reg  +  lambda_rank * (L_ordinal + L_listnet) / 2
              +  lambda_si   * L_si
              +  lambda_div  * L_diversity           (breaks same-prediction mode)
              +  lambda_aux  * (L_reg_geo + L_ordinal_geo)   (auxiliary on cost_head)
    """

    def __init__(self, lambda_rank=0.5, lambda_si=0.0, lambda_aux=0.5,
                 lambda_div=0.5):
        super().__init__()
        self.lambda_rank = lambda_rank
        self.lambda_si   = lambda_si
        self.lambda_aux  = lambda_aux
        self.lambda_div  = lambda_div
        self.regression  = GeodesicRegressionLoss()
        self.ranking     = OrdinalRankingLoss()
        self.listnet     = ListNetRankingLoss()
        self.scale_inv   = ScaleInvariantLogLoss()
        self.diversity   = PredictionDiversityLoss()

    def forward(self, predictions, targets, geo_preds=None):
        l_reg  = self.regression(predictions, targets)
        l_rank = self.ranking(predictions, targets)
        l_list = self.listnet(predictions, targets)
        l_si   = self.scale_inv(predictions, targets)
        l_div  = self.diversity(predictions)

        total = (l_reg
                 + self.lambda_rank * (l_rank + l_list) / 2.0
                 + self.lambda_si   * l_si
                 + self.lambda_div  * l_div)

        # Auxiliary supervision directly on the cost head (pre-refinement logits).
        l_aux = torch.tensor(0.0, device=l_reg.device)
        if geo_preds is not None and self.lambda_aux > 0:
            l_aux = (self.regression(geo_preds, targets)
                     + self.ranking(geo_preds, targets)
                     + self.diversity(geo_preds))
            total = total + self.lambda_aux * l_aux

        loss_dict = {
            "loss_total":           total.item(),
            "loss_regression":      l_reg.item(),
            "loss_ranking":         l_rank.item(),
            "loss_listnet":         l_list.item(),
            "loss_scale_invariant": l_si.item(),
            "loss_diversity":       l_div.item(),
            "loss_aux_geo":         l_aux.item(),
        }
        return total, loss_dict


def instruction_sensitivity_loss(
    model,
    pixel_values: torch.Tensor,
    masks_list: list,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    predictions: list,
) -> torch.Tensor:
    """
    Penalise the model for producing *similar* predictions when the instruction
    is swapped with a different sample's instruction from the same batch.

    Margin loss: push the per-sample MSE between original and negative
    predictions above a margin of 0.5.  If they are already more than 0.5
    apart, no penalty is applied.

    Within-batch permutation creates hard negatives cheaply without extra data.
    A random (non-identity) permutation is used so different pairs are seen
    across training steps.

    Returns a scalar loss.  Add it to the main loss weighted by ~0.3.
    """
    B = input_ids.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=input_ids.device)

    # Random permutation guaranteed to have no fixed points (derangement approx)
    perm = torch.randperm(B, device=input_ids.device)
    # If any position maps to itself, do a single cyclic shift of those slots
    identity = perm == torch.arange(B, device=input_ids.device)
    if identity.any():
        idx = identity.nonzero(as_tuple=True)[0]
        perm[idx] = perm[idx.roll(1)]

    neg_input_ids      = input_ids[perm]
    neg_attention_mask = attention_mask[perm]

    with torch.no_grad():
        neg_preds, _ = model(pixel_values, masks_list,
                             neg_input_ids, neg_attention_mask)

    loss = torch.tensor(0.0, device=input_ids.device)
    n = 0
    for orig, neg in zip(predictions, neg_preds):
        if orig.numel() < 1:
            continue
        # Compare in sigmoid (probability) space so the margin is scale-invariant.
        orig_sig = torch.sigmoid(orig.float())
        neg_sig  = torch.sigmoid(neg.float().detach())
        mse      = F.mse_loss(orig_sig, neg_sig)
        loss    += F.relu(0.05 - mse)   # margin in probability space
        n       += 1

    return loss / max(n, 1)

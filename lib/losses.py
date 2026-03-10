"""Loss functions and sample weighting for fire detection training."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

NDArrayFloat = npt.NDArray[np.floating[Any]]
WeightCounts = dict[str, int]


class SoftErrorRateLoss(nn.Module):
    """Differentiable approximation of (FN+FP)/P error rate.

    P_total is precomputed once from the full training set so that
    each mini-batch's FP/FN contributions are normalized consistently.

    Optional gt_fp_penalty adds extra loss for false positives on
    ground-truth (pre-burn) pixels when gt_mask is provided.
    """

    def __init__(self, P_total: float, gt_fp_penalty: float = 0.0):
        super().__init__()
        self.P_total = max(P_total, 1.0)
        self.gt_fp_penalty = gt_fp_penalty

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        soft_FP = (w * p * (1 - y)).sum()
        soft_FN = (w * (1 - p) * y).sum()
        loss = (soft_FN + soft_FP) / self.P_total
        if gt_mask is not None and self.gt_fp_penalty > 0:
            gt_FP = (p * (1 - y) * gt_mask.float()).sum()
            loss = loss + self.gt_fp_penalty * gt_FP / self.P_total
        return loss


class TverskyLoss(nn.Module):
    """Tversky loss -- generalised Dice with tunable FP/FN asymmetry.

    alpha < beta penalises false negatives more (recall-biased).
    alpha=beta=0.5 reduces to standard Dice loss.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0, gt_fp_penalty: float = 0.0,
                 P_total: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.gt_fp_penalty = gt_fp_penalty
        self.P_total = max(P_total, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        TP_soft = (w * p * y).sum()
        FP_soft = (w * p * (1 - y)).sum()
        FN_soft = (w * (1 - p) * y).sum()
        tversky = (TP_soft + self.smooth) / (
            TP_soft + self.alpha * FP_soft + self.beta * FN_soft + self.smooth
        )
        loss = 1.0 - tversky
        if gt_mask is not None and self.gt_fp_penalty > 0:
            gt_FP = (p * (1 - y) * gt_mask.float()).sum()
            loss = loss + self.gt_fp_penalty * gt_FP / self.P_total
        return loss


class FocalErrorRateLoss(nn.Module):
    """Error-rate loss with focal modulation on hard examples.

    Each sample's FP/FN contribution is weighted by (1 - p_correct)^gamma.
    gamma=0 reduces exactly to SoftErrorRateLoss.
    """

    def __init__(self, P_total: float, gamma: float = 2.0,
                 gt_fp_penalty: float = 0.0):
        super().__init__()
        self.P_total = max(P_total, 1.0)
        self.gamma = gamma
        self.gt_fp_penalty = gt_fp_penalty

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pt = y * p + (1 - y) * (1 - p)      # correct-class probability
        modulator = (1 - pt) ** self.gamma   # high for misclassified
        soft_FP = (modulator * w * p * (1 - y)).sum()
        soft_FN = (modulator * w * (1 - p) * y).sum()
        loss = (soft_FN + soft_FP) / self.P_total
        if gt_mask is not None and self.gt_fp_penalty > 0:
            gt_FP = (modulator * p * (1 - y) * gt_mask.float()).sum()
            loss = loss + self.gt_fp_penalty * gt_FP / self.P_total
        return loss


class CombinedLoss(nn.Module):
    """Weighted combination of BCE and soft error-rate losses.

    loss = lam * weighted_BCE + (1-lam) * SoftErrorRate
    gt_fp_penalty applies to the error-rate component.
    """

    def __init__(self, P_total: float, lam: float = 0.5,
                 gt_fp_penalty: float = 0.0):
        super().__init__()
        self.P_total = max(P_total, 1.0)
        self.lam = lam
        self.gt_fp_penalty = gt_fp_penalty
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
        gt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # BCE component (weighted per-sample)
        bce_component = (self.bce(logits, y) * w).mean()

        # Error-rate component
        p = torch.sigmoid(logits)
        soft_FP = (w * p * (1 - y)).sum()
        soft_FN = (w * (1 - p) * y).sum()
        er_component = (soft_FN + soft_FP) / self.P_total

        loss = self.lam * bce_component + (1 - self.lam) * er_component

        if gt_mask is not None and self.gt_fp_penalty > 0:
            gt_FP = (p * (1 - y) * gt_mask.float()).sum()
            loss = loss + self.gt_fp_penalty * gt_FP / self.P_total
        return loss


def compute_pixel_weights(
    y: NDArrayFloat,
    flight_source: npt.NDArray[np.str_],
    ground_truth_flight: str = '24-801-03',
    importance_gt: float = 10.0,
    importance_fire: float = 5.0,
    importance_other: float = 1.0,
) -> tuple[NDArrayFloat, WeightCounts]:
    """Compute normalized pixel-wise weights for loss function.

    Implements importance x inverse-frequency weighting to handle class
    imbalance and prioritize high-confidence samples. The formula:

        w_i = importance_i x (N / category_count_i)
        w_normalized = w / mean(w)

    Args:
        y: Labels array (0 = no fire, 1 = fire).
        flight_source: Array of flight IDs for each sample.
        ground_truth_flight: Flight ID with known no-fire ground truth.
        importance_gt: FP penalty on ground truth flight. Default 10.0.
        importance_fire: FN penalty on confirmed fire pixels. Default 5.0.
        importance_other: Penalty on uncertain non-fire in burn flights. Default 1.0.

    Returns:
        Tuple of (weights, counts) where:
            - weights: Array same shape as y, normalized so mean=1
            - counts: Dict with n_gt, n_fire, n_other category counts
    """
    n_total = len(y)

    # Identify categories
    is_gt = flight_source == ground_truth_flight
    is_fire = (y == 1) & ~is_gt
    is_other = (y == 0) & ~is_gt

    n_gt = int(is_gt.sum())
    n_fire = int(is_fire.sum())
    n_other = int(is_other.sum())

    # Weight = importance x inverse-frequency
    weights = np.ones(n_total, dtype=np.float32)
    if n_gt > 0:
        weights[is_gt] = importance_gt * (n_total / n_gt)
    if n_fire > 0:
        weights[is_fire] = importance_fire * (n_total / n_fire)
    if n_other > 0:
        weights[is_other] = importance_other * (n_total / n_other)

    # Normalize to mean=1 (keeps gradient scale stable)
    weights = weights / weights.mean()

    return weights, {'n_gt': n_gt, 'n_fire': n_fire, 'n_other': n_other}

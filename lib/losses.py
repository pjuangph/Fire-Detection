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
    """

    def __init__(self, P_total: float):
        super().__init__()
        self.P_total = max(P_total, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)
        soft_FP = (w * p * (1 - y)).sum()
        soft_FN = (w * (1 - p) * y).sum()
        return (soft_FN + soft_FP) / self.P_total


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

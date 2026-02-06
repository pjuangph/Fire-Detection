"""Model evaluation and device utilities for fire detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from lib.inference import FireMLP

NDArrayFloat = npt.NDArray[np.floating[Any]]
Metrics = dict[str, int | float]


def get_device() -> torch.device:
    """Return the best available device: MPS (Mac) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate(
    model: FireMLP,
    X: NDArrayFloat,
    y: NDArrayFloat,
    threshold: float = 0.5,
) -> tuple[Metrics, NDArrayFloat]:
    """Evaluate model with absolute count metrics.

    Computes confusion matrix (TP, FP, FN, TN) and derived metrics.
    Uses absolute counts per user preference (not percentages).

    Args:
        model: Trained model to evaluate.
        X: Normalized feature array of shape (N, 12).
        y: Ground truth labels of shape (N,).
        threshold: Classification threshold for P(fire). Default 0.5.

    Returns:
        Tuple of (metrics, probs) where:
            - metrics: Dict with TP, FP, FN, TN, precision, recall
            - probs: Array of P(fire) predictions of shape (N,)
    """
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()

    preds = (probs >= threshold).astype(np.float32)

    TP = int(np.sum((preds == 1) & (y == 1)))
    FP = int(np.sum((preds == 1) & (y == 0)))
    FN = int(np.sum((preds == 0) & (y == 1)))
    TN = int(np.sum((preds == 0) & (y == 0)))

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall,
    }, probs


def print_metrics(metrics: Metrics, label: str = '') -> None:
    """Print evaluation metrics to stdout.

    Args:
        metrics: Dict with TP, FP, FN, TN, precision, recall.
        label: Optional label prefix for output.
    """
    m = metrics
    header = f'  {label} ' if label else '  '
    print(f'{header}Absolute counts:')
    print(f'    TP (correct fire):     {m["TP"]:>8,}')
    print(f'    FP (false alarm):      {m["FP"]:>8,}')
    print(f'    FN (missed fire):      {m["FN"]:>8,}')
    print(f'    TN (correct no-fire):  {m["TN"]:>8,}')
    print(f'{header}Rates:')
    print(f'    Precision: {m["precision"]:.4f}')
    print(f'    Recall:    {m["recall"]:.4f}')

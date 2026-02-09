"""ML fire detection inference: model definition, loading, and prediction."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]

FEATURE_NAMES = [
    'T4_max', 'T4_mean', 'T11_mean', 'dT_max',
    'SWIR_max', 'SWIR_mean',
    'Red_mean', 'NIR_mean',
    'NDVI_min', 'NDVI_mean', 'NDVI_drop', 'obs_count',
]


class FireMLP(nn.Module):
    """MLP fire detector from aggregate features.

    Variable-depth architecture: n_features -> [hidden layers] -> 1.
    Output is raw logits; use BCEWithLogitsLoss or sigmoid for probabilities.

    Attributes:
        hidden_layers (list[int]): Hidden layer sizes used to build the network.
        net (nn.Sequential): The neural network layers.
    """

    def __init__(
        self,
        n_features: int = 12,
        hidden_layers: list[int] | None = None,
    ) -> None:
        """Initialize FireMLP.

        Args:
            n_features: Number of input features. Default 12.
            hidden_layers: List of hidden layer sizes, e.g. [64, 32, 16, 8].
                Default [64, 32] for backward compatibility.
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]
        self.hidden_layers = list(hidden_layers)

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features).

        Returns:
            torch.Tensor: Raw logits of shape (batch,).
        """
        return self.net(x).squeeze(-1)


def load_model(
    model_path: str = 'checkpoint/fire_detector.pt',
) -> tuple[FireMLP, StandardScaler]:
    """Load trained model and scaler from checkpoint.

    Args:
        model_path (str): Path to checkpoint file.

    Returns:
        tuple[FireMLP, StandardScaler]: Loaded model and fitted scaler.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = FireMLP(
        n_features=checkpoint.get('n_features', 12),
        hidden_layers=checkpoint.get('hidden_layers', [64, 32]),
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    scaler = checkpoint.get('scaler')
    if scaler is None:
        scaler = StandardScaler()
        scaler.mean_ = checkpoint['mean']
        scaler.scale_ = checkpoint['std']
        scaler.var_ = checkpoint['std'] ** 2
        scaler.n_features_in_ = len(checkpoint['mean'])

    return model, scaler


def predict(
    model: FireMLP,
    scaler: StandardScaler,
    X: NDArrayFloat,
    threshold: float = 0.5,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Run inference on feature matrix.

    Args:
        model (FireMLP): Trained model.
        scaler (StandardScaler): Fitted scaler for normalization.
        X (NDArrayFloat): Raw feature array of shape (N, 12).
        threshold (float): Classification threshold. Default 0.5.

    Returns:
        tuple[NDArrayFloat, NDArrayFloat]: (predictions, probabilities) where
            predictions is bool array and probabilities is float array.
    """
    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    X_norm = scaler.transform(X_clean).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_norm, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()

    preds = probs >= threshold
    return preds, probs


def _find_checkpoint(model_path: str | None = None) -> str | None:
    """Find a checkpoint file, checking explicit path then auto-discovery.

    Search order: explicit path > fire_detector_bce.pt > fire_detector_error-rate.pt
    > legacy fire_detector.pt.
    """
    if model_path and os.path.isfile(model_path):
        return model_path
    for candidate in [
        'checkpoint/fire_detector_best.pt',
        'checkpoint/fire_detector_bce.pt',
        'checkpoint/fire_detector_error-rate.pt',
        'checkpoint/fire_detector.pt',
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_fire_model(
    model_path: str | None = None,
) -> _MLFireDetector | None:
    """Load trained ML fire detector for realtime use. Returns None if not found."""
    path = _find_checkpoint(model_path)
    if path is None:
        return None
    return _MLFireDetector(path)


class _MLFireDetector:
    """Wrapper for grid-state-based inference in realtime_fire.py.

    Loads a trained model and predicts fire from running accumulators
    stored in grid state (gs). Used by realtime_fire.py.
    """

    def __init__(self, model_path: str) -> None:
        from lib.fire import compute_aggregate_features
        self._compute_features = compute_aggregate_features

        ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
        self.model = FireMLP(
            n_features=ckpt.get('n_features', 12),
            hidden_layers=ckpt.get('hidden_layers', [64, 32]),
        )
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.mean = ckpt['mean']
        self.std = ckpt['std']
        self.threshold = ckpt.get('threshold', 0.5)

    def predict_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Compute aggregate features from gs accumulators, run MLP.

        Returns:
            bool fire mask (nrows x ncols).
        """
        features, valid_mask = self._compute_features(gs)
        fire_mask = np.zeros((gs['nrows'], gs['ncols']), dtype=bool)

        if features.shape[0] == 0:
            return fire_mask

        x = (features - self.mean) / self.std
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)

        with torch.no_grad():
            logits = self.model(torch.tensor(x))
            probs = torch.sigmoid(logits).squeeze(-1).numpy()

        fire_mask[valid_mask] = probs >= self.threshold
        return fire_mask

    def predict_proba_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Return P(fire) grid (nrows x ncols), NaN where no data."""
        features, valid_mask = self._compute_features(gs)
        prob_grid = np.full((gs['nrows'], gs['ncols']), np.nan, dtype=np.float32)

        if features.shape[0] == 0:
            return prob_grid

        x = (features - self.mean) / self.std
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)

        with torch.no_grad():
            logits = self.model(torch.tensor(x))
            probs = torch.sigmoid(logits).squeeze(-1).numpy()

        prob_grid[valid_mask] = probs
        return prob_grid

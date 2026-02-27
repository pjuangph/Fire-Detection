"""ML fire detection inference: loading, and prediction."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler

from models.firemlp import FireMLP
from lib.constants import (
    T_IGNITION_DRY_WOOD, THERMAL_FEATURE_INDICES, NON_THERMAL_FEATURE_INDICES,
)

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]

FEATURE_NAMES = [
    'T4_max', 'T4_mean', 'T11_mean', 'dT_max',
    'SWIR_max', 'SWIR_mean',
    'Red_mean', 'NIR_mean',
    'NDVI_min', 'NDVI_mean', 'NDVI_drop', 'obs_count',
]


def _hybrid_normalize(
    X: NDArrayFloat,
    scaler: StandardScaler,
    T_ignition: float = T_IGNITION_DRY_WOOD,
) -> NDArrayFloat:
    """Apply hybrid normalization: thermal / T_ignition, non-thermal via scaler.

    Args:
        X: Clean feature array (N, 12), NaN already replaced with 0.
        scaler: StandardScaler fitted on non-thermal features (8 cols).
        T_ignition: Ignition temperature for thermal normalization.

    Returns:
        Normalized feature array (N, 12).
    """
    X_norm = np.empty_like(X)
    X_norm[:, THERMAL_FEATURE_INDICES] = X[:, THERMAL_FEATURE_INDICES] / T_ignition
    X_norm[:, NON_THERMAL_FEATURE_INDICES] = scaler.transform(
        X[:, NON_THERMAL_FEATURE_INDICES]).astype(np.float32)
    return X_norm


def load_model(
    model_path: str = 'checkpoint/fire_detector.pt',
) -> tuple[FireMLP, StandardScaler, float]:
    """Load trained model, scaler, and T_ignition from checkpoint.

    Args:
        model_path (str): Path to checkpoint file.

    Returns:
        tuple[FireMLP, StandardScaler, float]: Loaded model, fitted scaler,
            and T_ignition value.

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

    T_ignition = checkpoint.get('T_ignition', T_IGNITION_DRY_WOOD)

    return model, scaler, T_ignition


def predict(
    model: FireMLP,
    scaler: StandardScaler,
    X: NDArrayFloat,
    threshold: float = 0.5,
    T_ignition: float = T_IGNITION_DRY_WOOD,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Run inference on feature matrix with hybrid normalization.

    Args:
        model (FireMLP): Trained model.
        scaler (StandardScaler): Fitted scaler for non-thermal features.
        X (NDArrayFloat): Raw feature array of shape (N, 12).
        threshold (float): Classification threshold. Default 0.5.
        T_ignition (float): Ignition temperature for thermal normalization.

    Returns:
        tuple[NDArrayFloat, NDArrayFloat]: (predictions, probabilities) where
            predictions is bool array and probabilities is float array.
    """
    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    X_norm = _hybrid_normalize(X_clean, scaler, T_ignition)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_norm, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()

    preds = probs >= threshold
    return preds, probs


def _find_checkpoint(
    model_path: str | None = None,
    model_type: str = 'firemlp',
) -> str | None:
    """Find a checkpoint file, checking explicit path then auto-discovery.

    Args:
        model_path: Explicit path. If valid file, returned immediately.
        model_type: 'firemlp' or 'tabpfn' — controls auto-discovery candidates.
    """
    if model_path and os.path.isfile(model_path):
        return model_path

    if model_type == 'tabpfn_regression':
        candidates = [
            'checkpoint/fire_detector_tabpfn_regression_best.pt',
        ]
    elif model_type == 'tabpfn':
        candidates = [
            'checkpoint/fire_detector_tabpfn_best.pt',
        ]
    else:
        candidates = [
            'checkpoint/fire_detector_mlp_best.pt',
            'checkpoint/fire_detector_best.pt',
            'checkpoint/fire_detector_bce.pt',
            'checkpoint/fire_detector_error-rate.pt',
            'checkpoint/fire_detector.pt',
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_fire_model(
    model_path: str | None = None,
    threshold: float | None = None,
    model_type: str = 'firemlp',
) -> '_MLFireDetector | _TabPFNFireDetector | _TabPFNRegressionDetector | None':
    """Load trained ML fire detector for realtime use. Returns None if not found.

    Args:
        model_path: Explicit checkpoint path, or None for auto-discovery.
        threshold: Override classification threshold (default: use checkpoint value).
        model_type: 'firemlp', 'tabpfn', or 'tabpfn_regression'.
    """
    path = _find_checkpoint(model_path, model_type=model_type)
    if path is None:
        return None
    if model_type == 'tabpfn_regression':
        return _TabPFNRegressionDetector(path, threshold=threshold)
    if model_type == 'tabpfn':
        return _TabPFNFireDetector(path, threshold=threshold)
    return _MLFireDetector(path, threshold=threshold)


class _MLFireDetector:
    """Wrapper for grid-state-based inference in realtime scripts.

    Loads a trained model and predicts fire from running accumulators
    stored in grid state (gs).
    """

    def __init__(self, model_path: str, threshold: float | None = None) -> None:
        from lib.fire import compute_aggregate_features
        self._compute_features = compute_aggregate_features

        ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
        self.model = FireMLP(
            n_features=ckpt.get('n_features', 12),
            hidden_layers=ckpt.get('hidden_layers', [64, 32]),
        )
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.scaler = ckpt.get('scaler')
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.mean_ = ckpt['mean']
            self.scaler.scale_ = ckpt['std']
            self.scaler.var_ = ckpt['std'] ** 2
            self.scaler.n_features_in_ = len(ckpt['mean'])
        self.T_ignition = ckpt.get('T_ignition', T_IGNITION_DRY_WOOD)
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply hybrid normalization to raw features."""
        x = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        return _hybrid_normalize(x, self.scaler, self.T_ignition)

    def predict_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Compute aggregate features from gs accumulators, run MLP.

        Returns:
            bool fire mask (nrows x ncols).
        """
        features, valid_mask = self._compute_features(gs)
        fire_mask = np.zeros((gs['nrows'], gs['ncols']), dtype=bool)

        if features.shape[0] == 0:
            return fire_mask

        x = self._normalize(features)

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

        x = self._normalize(features)

        with torch.no_grad():
            logits = self.model(torch.tensor(x))
            probs = torch.sigmoid(logits).squeeze(-1).numpy()

        prob_grid[valid_mask] = probs
        return prob_grid


class _TabPFNFireDetector:
    """Wrapper for grid-state-based TabPFN inference in realtime scripts.

    Supports two checkpoint formats:
      - `.pt` (from-scratch): reconstructs TabPFNClassifier from random init,
        loads trained weights, fits on saved representative context.
      - `.pkl` (legacy joblib): loads pre-fitted TabPFNClassifier directly.
    """

    def __init__(self, model_path: str, threshold: float | None = None) -> None:
        from lib.fire import compute_aggregate_features
        self._compute_features = compute_aggregate_features

        if model_path.endswith('.pkl'):
            self._load_legacy(model_path, threshold)
        else:
            self._load_from_scratch(model_path, threshold)

    def _load_legacy(self, model_path: str, threshold: float | None) -> None:
        """Load pre-fitted TabPFNClassifier from joblib checkpoint."""
        import joblib
        ckpt = joblib.load(model_path)
        self.model = ckpt['model']
        self.scaler = ckpt['scaler']
        self.T_ignition = ckpt.get('T_ignition', T_IGNITION_DRY_WOOD)
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

    def _load_from_scratch(self, model_path: str, threshold: float | None) -> None:
        """Load from-scratch trained TabPFN from torch checkpoint.

        Reconstructs the classifier with random weights, loads the trained
        state dict, clones for evaluation, and fits on the saved context.
        """
        from tabpfn import TabPFNClassifier
        from tabpfn.finetuning.train_util import clone_model_for_evaluation

        ckpt = torch.load(model_path, weights_only=False, map_location='cpu')

        self.scaler = ckpt['scaler']
        self.T_ignition = ckpt.get('T_ignition', T_IGNITION_DRY_WOOD)
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

        # Reconstruct classifier with random weights, then load trained weights
        from lib.evaluation import auto_device
        init_args = ckpt['classifier_init'].copy()
        init_args['device'] = auto_device()  # override checkpoint device
        classifier = TabPFNClassifier(
            **init_args,
            fit_mode="batched",
            differentiable_input=False,
        )
        classifier._initialize_model_variables()
        classifier.models_[0].load_state_dict(ckpt['model_state_dict'])

        # Clone for clean evaluation and fit on saved context
        eval_init = {k: v for k, v in init_args.items() if k != 'model_path'}
        eval_clf = clone_model_for_evaluation(
            classifier, eval_init, TabPFNClassifier)
        eval_clf.fit(ckpt['context_X'], ckpt['context_y'])

        self.model = eval_clf  # has predict_proba()

    def _batched_predict_proba(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Predict probabilities in batches to avoid MPS OOM."""
        parts = []
        for i in range(0, len(X), batch_size):
            parts.append(self.model.predict_proba(X[i:i + batch_size])[:, 1])
        return np.concatenate(parts)

    def predict_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Compute aggregate features from gs accumulators, run TabPFN.

        Returns:
            bool fire mask (nrows x ncols).
        """
        features, valid_mask = self._compute_features(gs)
        fire_mask = np.zeros((gs['nrows'], gs['ncols']), dtype=bool)

        if features.shape[0] == 0:
            return fire_mask

        X = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        X = _hybrid_normalize(X, self.scaler, self.T_ignition)
        probs = self._batched_predict_proba(X)

        fire_mask[valid_mask] = probs >= self.threshold
        return fire_mask

    def predict_proba_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Return P(fire) grid (nrows x ncols), NaN where no data."""
        features, valid_mask = self._compute_features(gs)
        prob_grid = np.full((gs['nrows'], gs['ncols']), np.nan, dtype=np.float32)

        if features.shape[0] == 0:
            return prob_grid

        X = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        X = _hybrid_normalize(X, self.scaler, self.T_ignition)
        probs = self._batched_predict_proba(X)

        prob_grid[valid_mask] = probs
        return prob_grid


class _TabPFNRegressionDetector:
    """Wrapper for grid-state-based TabPFN regression inference.

    Loads a from-scratch trained TabPFNRegressor checkpoint, reconstructs
    the model, and uses predict() (continuous output clipped to [0,1]).
    """

    def __init__(self, model_path: str, threshold: float | None = None) -> None:
        from lib.fire import compute_aggregate_features
        self._compute_features = compute_aggregate_features

        self._load_from_scratch(model_path, threshold)

    def _load_from_scratch(self, model_path: str, threshold: float | None) -> None:
        """Load from-scratch trained TabPFN regressor from torch checkpoint."""
        from tabpfn import TabPFNRegressor
        from tabpfn.finetuning.train_util import clone_model_for_evaluation

        ckpt = torch.load(model_path, weights_only=False, map_location='cpu')

        self.scaler = ckpt['scaler']
        self.T_ignition = ckpt.get('T_ignition', T_IGNITION_DRY_WOOD)
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

        # Reconstruct regressor with random weights, then load trained weights
        from lib.evaluation import auto_device
        init_args = ckpt['regressor_init'].copy()
        init_args['device'] = auto_device()  # override checkpoint device
        regressor = TabPFNRegressor(
            **init_args,
            fit_mode="batched",
            differentiable_input=False,
        )
        regressor._initialize_model_variables()
        regressor.models_[0].load_state_dict(ckpt['model_state_dict'])

        # Clone for clean evaluation and fit on saved context
        eval_init = {k: v for k, v in init_args.items() if k != 'model_path'}
        eval_reg = clone_model_for_evaluation(
            regressor, eval_init, TabPFNRegressor)
        eval_reg.fit(ckpt['context_X'], ckpt['context_y'])

        self.model = eval_reg  # has predict()

    def _batched_predict(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Predict in batches to avoid MPS OOM."""
        parts = []
        for i in range(0, len(X), batch_size):
            parts.append(self.model.predict(X[i:i + batch_size]))
        return np.concatenate(parts)

    def predict_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Compute aggregate features from gs accumulators, run TabPFN regressor.

        Returns:
            bool fire mask (nrows x ncols).
        """
        features, valid_mask = self._compute_features(gs)
        fire_mask = np.zeros((gs['nrows'], gs['ncols']), dtype=bool)

        if features.shape[0] == 0:
            return fire_mask

        X = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        X = _hybrid_normalize(X, self.scaler, self.T_ignition)
        probs = np.clip(self._batched_predict(X), 0.0, 1.0)

        fire_mask[valid_mask] = probs >= self.threshold
        return fire_mask

    def predict_proba_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Return P(fire) grid (nrows x ncols), NaN where no data."""
        features, valid_mask = self._compute_features(gs)
        prob_grid = np.full((gs['nrows'], gs['ncols']), np.nan, dtype=np.float32)

        if features.shape[0] == 0:
            return prob_grid

        X = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        X = _hybrid_normalize(X, self.scaler, self.T_ignition)
        probs = np.clip(self._batched_predict(X), 0.0, 1.0)

        prob_grid[valid_mask] = probs
        return prob_grid

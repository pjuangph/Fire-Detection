"""ML fire detection inference: loading, and prediction."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.preprocessing import StandardScaler

from models.firemlp import FireMLP

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]

FEATURE_NAMES = [
    'T4_max', 'T4_mean', 'T11_mean', 'dT_max',
    'SWIR_max', 'SWIR_mean',
    'Red_mean', 'NIR_mean',
    'NDVI_min', 'NDVI_mean', 'NDVI_drop', 'obs_count',
]


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
        self.mean = ckpt['mean']
        self.std = ckpt['std']
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

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
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

        # Reconstruct classifier with random weights, then load trained weights
        init_args = ckpt['classifier_init']
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
        X = self.scaler.transform(X)
        probs = self.model.predict_proba(X)[:, 1]

        fire_mask[valid_mask] = probs >= self.threshold
        return fire_mask

    def predict_proba_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Return P(fire) grid (nrows x ncols), NaN where no data."""
        features, valid_mask = self._compute_features(gs)
        prob_grid = np.full((gs['nrows'], gs['ncols']), np.nan, dtype=np.float32)

        if features.shape[0] == 0:
            return prob_grid

        X = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        X = self.scaler.transform(X)
        probs = self.model.predict_proba(X)[:, 1]

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
        self.threshold = threshold if threshold is not None else ckpt.get('threshold', 0.5)

        # Reconstruct regressor with random weights, then load trained weights
        init_args = ckpt['regressor_init']
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
        X = self.scaler.transform(X)
        probs = np.clip(self.model.predict(X), 0.0, 1.0)

        fire_mask[valid_mask] = probs >= self.threshold
        return fire_mask

    def predict_proba_from_gs(self, gs: dict[str, Any]) -> np.ndarray:
        """Return P(fire) grid (nrows x ncols), NaN where no data."""
        features, valid_mask = self._compute_features(gs)
        prob_grid = np.full((gs['nrows'], gs['ncols']), np.nan, dtype=np.float32)

        if features.shape[0] == 0:
            return prob_grid

        X = np.where(np.isfinite(features), features, 0.0).astype(np.float32)
        X = self.scaler.transform(X)
        probs = np.clip(self.model.predict(X), 0.0, 1.0)

        prob_grid[valid_mask] = probs
        return prob_grid

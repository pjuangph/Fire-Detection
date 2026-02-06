"""tune_fire_prediction.py - Train MLP fire detector with accumulated observations.

Trains an MLP that predicts fire from 8 aggregate features computed across
all observations of each pixel:

  T4_max    — peak temperature (fire spike)
  T4_mean   — average thermal state (normalizes the peak)
  T11_mean  — background temperature (stable reference)
  dT_max    — strongest T4-T11 difference (fire signature)
  NDVI_min  — lowest vegetation (burn scar indicator)
  NDVI_mean — average vegetation (normalizes the drop)
  NDVI_drop — NDVI_first - NDVI_min (temporal vegetation loss)
  obs_count — number of observations (reliability)

These aggregate features normalize the data by putting each pixel in its
own temporal context. The MLP learns patterns like: high T4_max vs T4_mean
(thermal anomaly) + large NDVI_drop (vegetation loss) = fire.

Loss: BCEWithLogitsLoss (binary cross-entropy) → sigmoid → P(fire).

Train/test split by flight:
    Train: flights 03 (pre-burn), 04 (day burn), 05 (night burn)
    Test:  flight 06 (day burn, unseen)

Usage:
    python tune_fire_prediction.py
"""

from __future__ import annotations

import os
import pickle
import gzip
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import trange
from lib import (
    group_files_by_flight, compute_grid_extent,
    build_pixel_table, compute_location_stats,
)

# Type aliases for clarity
NDArrayFloat = npt.NDArray[np.floating[Any]]
NDArrayBool = npt.NDArray[np.bool_]
FlightFeatures = dict[str, dict[str, Any]]
WeightCounts = dict[str, int]
Metrics = dict[str, int | float]


# ── Feature Engineering ───────────────────────────────────────


def build_location_features(
    pixel_df: pd.DataFrame,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute 8 aggregate features per grid-cell location from pixel table.

    Groups by (lat, lon) and computes running statistics that mirror
    what process_sweep() maintains in gs accumulators. Features are designed
    to normalize each pixel's temporal context for fire detection.

    Feature selection based on MODIS fire detection literature:
    - Giglio et al. (2003) "An Enhanced Contextual Fire Detection Algorithm"
    - Schroeder et al. (2014) "The New VIIRS 375m Active Fire Detection"

    The 8 aggregate features:
        1. T4_max: Peak 3.9μm brightness temperature (fire spike)
        2. T4_mean: Average thermal state (normalizes peak)
        3. T11_mean: Background 11μm temperature (stable reference)
        4. dT_max: Strongest T4-T11 difference (fire signature)
        5. NDVI_min: Lowest vegetation index (burn scar indicator)
        6. NDVI_mean: Average vegetation (normalizes drop)
        7. NDVI_drop: First NDVI - min NDVI (temporal vegetation loss)
        8. obs_count: Number of observations (reliability indicator)

    Args:
        pixel_df (pd.DataFrame): DataFrame from build_pixel_table() with columns
            lat, lon, T4, T11, dT, SWIR, NDVI, fire.

    Returns:
        tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]: Tuple of
            (X, y, lats, lons) where:
            - X: (N_locations, 8) float32 feature matrix
            - y: (N_locations,) float32 labels (1 if any observation was fire)
            - lats: (N_locations,) latitude coordinates
            - lons: (N_locations,) longitude coordinates
    """
    grouped = pixel_df.groupby(['lat', 'lon'])

    T4_max = np.asarray(grouped['T4'].max().values)
    T4_mean = np.asarray(grouped['T4'].mean().values)
    T11_mean = np.asarray(grouped['T11'].mean().values)
    dT_max = np.asarray(grouped['dT'].max().values)

    # NDVI features: only from daytime (finite NDVI) observations
    ndvi_min = np.asarray(grouped['NDVI'].min().values)
    ndvi_mean = np.asarray(grouped['NDVI'].mean().values)

    # NDVI_drop: first NDVI - min NDVI (vegetation loss over time)
    def ndvi_drop_fn(g: pd.Series) -> float:
        finite = g[np.isfinite(g)]
        if len(finite) < 2:
            return 0.0
        return float(finite.iloc[0] - finite.min())
    ndvi_drop = np.asarray(grouped['NDVI'].apply(ndvi_drop_fn).values)

    obs_count = np.asarray(grouped['T4'].count().values)

    # Label: 1 if any observation at this location was fire
    fire_rate = np.asarray(grouped['fire'].mean().values)
    y = (fire_rate > 0).astype(np.float32)

    # Fill NaN NDVI features with 0 (night-only pixels)
    ndvi_min = np.where(np.isfinite(ndvi_min), ndvi_min, 0.0)
    ndvi_mean = np.where(np.isfinite(ndvi_mean), ndvi_mean, 0.0)
    ndvi_drop = np.where(np.isfinite(ndvi_drop), ndvi_drop, 0.0)

    X = np.stack([
        T4_max, T4_mean, T11_mean, dT_max,
        ndvi_min, ndvi_mean, ndvi_drop, obs_count,
    ], axis=1).astype(np.float32)

    # Replace any remaining non-finite with 0
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)

    coords = grouped['T4'].count().reset_index()
    lats = np.asarray(coords['lat'].values)
    lons = np.asarray(coords['lon'].values)

    return X, y, lats, lons


# ── Pixel-Wise Weighting ──────────────────────────────────────


def compute_pixel_weights(
    y: NDArrayFloat,
    flight_source: npt.NDArray[np.str_],
    ground_truth_flight: str = '24-801-03',
) -> tuple[NDArrayFloat, WeightCounts]:
    """Compute normalized pixel-wise weights for BCE loss.

    Implements importance × inverse-frequency weighting to handle class
    imbalance and prioritize high-confidence samples. The formula:

        w_i = importance_i × (N / category_count_i)
        w_normalized = w / mean(w)

    This approach is inspired by focal loss (Lin et al., 2017) and
    cost-sensitive learning for imbalanced datasets.

    Importance factors (empirically tuned for fire detection):
        - Ground truth no-fire (flight 03): 10.0 — FP here is definitely wrong
        - Fire pixels in burn flights: 5.0 — confirmed detections, high priority
        - Non-fire in burn flights: 1.0 — uncertain baseline

    Assumptions:
        - Flight 03 is pre-burn with no fire (ground truth negative)
        - Threshold detector labels in burn flights are pseudo-ground-truth
        - Inverse-frequency balances category sizes so small categories
          (rare fire pixels) aren't drowned out by large categories

    Args:
        y (NDArrayFloat): Labels array (0 = no fire, 1 = fire).
        flight_source (npt.NDArray[np.str_]): Array of flight IDs for each sample.
        ground_truth_flight (str): Flight ID with known no-fire ground truth.
            Default is '24-801-03' (FIREX-AQ pre-burn flight).

    Returns:
        tuple[NDArrayFloat, WeightCounts]: Tuple of (weights, counts) where:
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

    # Importance factors (how bad is an error on this category?)
    IMPORTANCE_GT = 10.0      # FP on ground truth = definitely wrong
    IMPORTANCE_FIRE = 5.0     # FN on fire = missed detection
    IMPORTANCE_OTHER = 1.0    # Uncertain baseline

    # Weight = importance × inverse-frequency
    weights = np.ones(n_total, dtype=np.float32)
    if n_gt > 0:
        weights[is_gt] = IMPORTANCE_GT * (n_total / n_gt)
    if n_fire > 0:
        weights[is_fire] = IMPORTANCE_FIRE * (n_total / n_fire)
    if n_other > 0:
        weights[is_other] = IMPORTANCE_OTHER * (n_total / n_other)

    # Normalize to mean=1 (keeps gradient scale stable)
    weights = weights / weights.mean()

    return weights, {'n_gt': n_gt, 'n_fire': n_fire, 'n_other': n_other}


# ── Data Pipeline ─────────────────────────────────────────────


def load_all_data(
    flights: dict[str, dict[str, Any]],
) -> FlightFeatures:
    """Build pixel tables for all flights and compute location features.

    Processes HDF files for each flight, extracts observations, and computes
    8 aggregate features per grid-cell location.

    Args:
        flights (dict[str, dict[str, Any]]): Flight metadata from
            group_files_by_flight(). Keys are flight IDs (e.g., '24-801-03'),
            values contain 'files', 'day_night', 'comment'.

    Returns:
        FlightFeatures: Dict mapping flight_num to feature dict containing:
            - 'X': (N, 8) float32 feature matrix
            - 'y': (N,) float32 labels
            - 'lats': (N,) latitude coordinates
            - 'lons': (N,) longitude coordinates
            - 'pixel_df': Raw DataFrame with all observations
            - 'day_night': 'day' or 'night'
            - 'comment': Flight description
    """
    flight_features = {}
    for fnum, info in sorted(flights.items()):
        files = info['files']
        day_night = info['day_night']
        print(f'  Flight {fnum} ({len(files)} files, {day_night})...')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        pixel_df = build_pixel_table(
            files, lat_min, lat_max, lon_min, lon_max,
            day_night=day_night, flight_num=fnum)

        n_pixels = len(pixel_df)
        n_fire = int(pixel_df['fire'].sum())
        print(f'    {n_pixels:,} observations, {n_fire:,} fire detections')

        X, y, lats, lons = build_location_features(pixel_df)
        n_locs = len(y)
        n_fire_locs = int(y.sum())
        print(f'    {n_locs:,} locations, {n_fire_locs:,} with fire')

        flight_features[fnum] = {
            'X': X, 'y': y, 'lats': lats, 'lons': lons,
            'pixel_df': pixel_df, 'day_night': day_night,
            'comment': info['comment'],
        }
    return flight_features


def extract_train_test(
    flight_features: FlightFeatures,
    train_flights: list[str],
    test_flights: list[str],
    ground_truth_flight: str = '24-801-03',
    gt_test_ratio: float = 0.2,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Split ground truth flight between train/test, then add burn flights.

    Flight 03 (pre-burn, no fire) is split 80/20 between train/test.
    This ensures test set has ground truth "no fire" data for proper
    false positive evaluation.

    Split strategy:
        - Train: 80% of ground truth + burn flights from train_flights
        - Test: 20% of ground truth + burn flights from test_flights

    Assumptions:
        - Ground truth flight has NO fire (pre-burn baseline)
        - Random split with seed=42 for reproducibility
        - Pixel-wise weights computed via importance × inverse-frequency

    Args:
        flight_features (FlightFeatures): Dict from load_all_data().
        train_flights (list[str]): Flight IDs for training (burn flights).
        test_flights (list[str]): Flight IDs for testing (burn flights).
        ground_truth_flight (str): Flight ID with known no-fire ground truth.
            Default '24-801-03' is FIREX-AQ pre-burn flight.
        gt_test_ratio (float): Fraction of ground truth to put in test set.
            Default 0.2 (80/20 split).

    Returns:
        tuple[NDArrayFloat, ...]: Six arrays:
            - X_train: (N_train, 8) training features
            - y_train: (N_train,) training labels
            - w_train: (N_train,) training weights (normalized mean=1)
            - X_test: (N_test, 8) test features
            - y_test: (N_test,) test labels
            - w_test: (N_test,) test weights (normalized mean=1)
    """
    rng = np.random.default_rng(42)

    # Split ground truth flight (03) 80/20
    gt = flight_features[ground_truth_flight]
    n_gt = len(gt['X'])
    n_gt_test = int(n_gt * gt_test_ratio)
    perm = rng.permutation(n_gt)
    gt_test_idx = perm[:n_gt_test]
    gt_train_idx = perm[n_gt_test:]

    # Train: 80% of ground truth + burn flights from train_flights
    train_X = [gt['X'][gt_train_idx]]
    train_y = [gt['y'][gt_train_idx]]
    train_flight_src = [np.full(len(gt_train_idx), ground_truth_flight)]
    for f in train_flights:
        if f != ground_truth_flight:
            train_X.append(flight_features[f]['X'])
            train_y.append(flight_features[f]['y'])
            train_flight_src.append(np.full(len(flight_features[f]['y']), f))

    # Test: 20% of ground truth + burn flights from test_flights
    test_X = [gt['X'][gt_test_idx]]
    test_y = [gt['y'][gt_test_idx]]
    test_flight_src = [np.full(len(gt_test_idx), ground_truth_flight)]
    for f in test_flights:
        if f != ground_truth_flight:
            test_X.append(flight_features[f]['X'])
            test_y.append(flight_features[f]['y'])
            test_flight_src.append(np.full(len(flight_features[f]['y']), f))

    X_train = np.concatenate(train_X)
    y_train = np.concatenate(train_y)
    flight_src_train = np.concatenate(train_flight_src)

    X_test = np.concatenate(test_X)
    y_test = np.concatenate(test_y)
    flight_src_test = np.concatenate(test_flight_src)

    # Compute pixel-wise weights
    w_train, train_counts = compute_pixel_weights(
        y_train, flight_src_train, ground_truth_flight)
    w_test, test_counts = compute_pixel_weights(
        y_test, flight_src_test, ground_truth_flight)

    print(f'  Train weights: gt={train_counts["n_gt"]:,}, '
          f'fire={train_counts["n_fire"]:,}, other={train_counts["n_other"]:,}')
    print(f'  Test weights:  gt={test_counts["n_gt"]:,}, '
          f'fire={test_counts["n_fire"]:,}, other={test_counts["n_other"]:,}')

    return (X_train, y_train, w_train, X_test, y_test, w_test)


def oversample_minority(
    X: NDArrayFloat,
    y: NDArrayFloat,
    w: NDArrayFloat,
    ratio: float = 1.0,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Oversample fire locations to balance training data, preserving weights.

    Implements random oversampling with replacement to address class imbalance.
    After oversampling, weights are re-normalized so mean=1 to maintain
    stable gradient magnitudes.

    This approach follows Chawla et al. (2002) recommendations for handling
    imbalanced datasets, with the addition of weight preservation.

    Args:
        X (NDArrayFloat): Feature array of shape (N, 8).
        y (NDArrayFloat): Label array of shape (N,).
        w (NDArrayFloat): Weight array of shape (N,), pixel-wise weights.
        ratio (float): Target fire:no-fire ratio. Default 1.0 (balanced).

    Returns:
        tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]: Tuple of
            (X_bal, y_bal, w_bal) with balanced and shuffled arrays.
            Weights are re-normalized so mean=1.
    """
    fire_mask = y == 1
    n_fire = int(fire_mask.sum())
    n_nofire = int((~fire_mask).sum())
    if n_fire == 0 or n_fire >= n_nofire * ratio:
        return X, y, w

    target_fire = int(n_nofire * ratio)
    rng = np.random.default_rng(42)
    repeat_idx = rng.choice(
        np.where(fire_mask)[0], target_fire - n_fire, replace=True)

    X_bal = np.concatenate([X, X[repeat_idx]], axis=0)
    y_bal = np.concatenate([y, y[repeat_idx]], axis=0)
    w_bal = np.concatenate([w, w[repeat_idx]], axis=0)

    # Re-normalize weights so mean=1 after oversampling
    w_bal = w_bal / w_bal.mean()

    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm], w_bal[perm]


# ── Model ─────────────────────────────────────────────────────


FEATURE_NAMES = [
    'T4_max', 'T4_mean', 'T11_mean', 'dT_max',
    'NDVI_min', 'NDVI_mean', 'NDVI_drop', 'obs_count',
]


class FireMLP(nn.Module):
    """MLP fire detector from aggregate features.

    A simple multi-layer perceptron for binary fire classification.
    Architecture: 8 → 64 → 32 → 1 (2,465 parameters).

    Design choices:
        - Two hidden layers with ReLU activation (universal approximation)
        - No dropout (dataset is large enough, regularization via weighting)
        - Output is raw logits (use BCEWithLogitsLoss for numerical stability)

    Attributes:
        net (nn.Sequential): The neural network layers.
    """

    def __init__(
        self,
        n_features: int = 8,
        hidden1: int = 64,
        hidden2: int = 32,
    ) -> None:
        """Initialize FireMLP.

        Args:
            n_features (int): Number of input features. Default 8.
            hidden1 (int): First hidden layer size. Default 64.
            hidden2 (int): Second hidden layer size. Default 32.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, n_features).

        Returns:
            torch.Tensor: Raw logits of shape (batch,).
        """
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────


def get_device() -> torch.device:
    """Get best available device: MPS (Mac), CUDA (NVIDIA), or CPU.

    Returns:
        torch.device: The best available compute device.
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_model(
    X_train: NDArrayFloat,
    y_train: NDArrayFloat,
    w_train: NDArrayFloat,
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 1024,
) -> tuple[FireMLP, NDArrayFloat]:
    """Train FireMLP with pixel-wise weighted BCEWithLogitsLoss.

    Uses MPS on Mac, CUDA on NVIDIA, or CPU as fallback.

    Loss function:
        L = (1/N) * sum_i(w_i * BCE(logit_i, y_i))

    where w_i is the pixel-wise weight encoding sample importance.

    Training hyperparameters (empirically tuned):
        - 300 epochs: sufficient for convergence on this dataset
        - lr=1e-3: Adam default, works well for MLPs
        - batch_size=1024: balances GPU utilization and gradient noise

    Args:
        X_train (NDArrayFloat): Normalized feature array of shape (N, 8).
        y_train (NDArrayFloat): Label array of shape (N,).
        w_train (NDArrayFloat): Pixel-wise weights of shape (N,),
            normalized so mean=1.
        n_epochs (int): Number of training epochs. Default 300.
        lr (float): Learning rate for Adam optimizer. Default 1e-3.
        batch_size (int): Batch size for training. Default 1024.

    Returns:
        tuple[FireMLP, NDArrayFloat]: Tuple of (model, loss_history) where:
            - model: Trained FireMLP moved to CPU
            - loss_history: Array of per-epoch average loss values
    """
    device = get_device()
    print(f'  Using device: {device}')

    model = FireMLP().to(device)
    # Use reduction='none' so we can apply per-sample weights
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    w_t = torch.tensor(w_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)
    avg_loss = 1000

    loss_history = np.zeros((n_epochs, 1))
    pbar = trange(n_epochs, desc="Training Model, Pixel Weighted BCE")
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for X_batch, y_batch, w_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            # Per-sample loss, then apply weights
            loss_per_sample = criterion(logits, y_batch)
            loss = (loss_per_sample * w_batch).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(X_t)
        loss_history[epoch] = avg_loss
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
    # Move model back to CPU for saving and inference
    model = model.cpu()
    return model, loss_history


# ── Evaluation ────────────────────────────────────────────────


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
        model (FireMLP): Trained model to evaluate.
        X (NDArrayFloat): Normalized feature array of shape (N, 8).
        y (NDArrayFloat): Ground truth labels of shape (N,).
        threshold (float): Classification threshold for P(fire). Default 0.5.

    Returns:
        tuple[Metrics, NDArrayFloat]: Tuple of (metrics, probs) where:
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
        metrics (Metrics): Dict with TP, FP, FN, TN, precision, recall.
        label (str): Optional label prefix for output.
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


# ── Visualization ─────────────────────────────────────────────


def plot_training_loss(loss_history: NDArrayFloat) -> None:
    """Plot weighted BCE training loss curve.

    Saves plot to plots/tune_training_loss.png.

    Args:
        loss_history (NDArrayFloat): Array of per-epoch average loss values.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', linewidth=1.5, label='BCE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence \u2014 BCE Loss (Aggregate Features)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/tune_training_loss.png', dpi=150, bbox_inches='tight')
    print('  Saved plots/tune_training_loss.png')
    plt.close()


def plot_probability_hist(
    probs_fire: NDArrayFloat,
    probs_nofire: NDArrayFloat,
) -> None:
    """Plot P(fire) histogram for fire vs non-fire locations.

    Visualizes model calibration by showing probability distributions
    for each class. Well-calibrated model should show separation.

    Saves plot to plots/tune_probability_hist.png.

    Args:
        probs_fire (NDArrayFloat): P(fire) for true fire locations.
        probs_nofire (NDArrayFloat): P(fire) for true non-fire locations.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs_nofire, bins=50, alpha=0.6, label='No fire', color='gray',
            density=True)
    ax.hist(probs_fire, bins=50, alpha=0.6, label='Fire', color='red',
            density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
               label='Threshold (0.5)')
    ax.set_xlabel('P(fire)')
    ax.set_ylabel('Density')
    ax.set_title('Fire Probability Distribution (Test Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/tune_probability_hist.png', dpi=150, bbox_inches='tight')
    print('  Saved plots/tune_probability_hist.png')
    plt.close()


def plot_prediction_map(
    flight_features: FlightFeatures,
    model: FireMLP,
    scaler: StandardScaler,
    flight_num: str,
) -> None:
    """Plot spatial fire predictions for one flight.

    Creates 2x2 subplot comparing ML predictions vs threshold detector:
        - Top-left: ML predictions (red = fire)
        - Top-right: Threshold detector labels
        - Bottom-left: Agreement (green=both, blue=ML only, orange=thresh only)
        - Bottom-right: P(fire) heatmap

    Saves plot to plots/tune_prediction_map_{flight_num}.png.

    Args:
        flight_features (FlightFeatures): Dict from load_all_data().
        model (FireMLP): Trained model.
        scaler (StandardScaler): Fitted scaler for feature normalization.
        flight_num (str): Flight ID to plot (e.g., '24-801-06').
    """
    d = flight_features[flight_num]
    X, y = d['X'], d['y']
    lats, lons = d['lats'], d['lons']

    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    X_norm = scaler.transform(X_clean).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_norm))
        probs = torch.sigmoid(logits).numpy()

    ml_fire = probs >= 0.5
    thresh_fire = y > 0.5

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'ML vs Threshold \u2014 Flight {flight_num} ({d["comment"]})',
        fontsize=14, fontweight='bold')

    # Top-left: ML predictions
    ax = axes[0, 0]
    ax.scatter(lons[~ml_fire], lats[~ml_fire], s=0.1, c='gray', alpha=0.2)
    if ml_fire.any():
        ax.scatter(lons[ml_fire], lats[ml_fire], s=1, c='red', alpha=0.7)
    ax.set_title(f'ML Predictions ({int(ml_fire.sum()):,} fire locations)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Top-right: Threshold labels
    ax = axes[0, 1]
    ax.scatter(lons[~thresh_fire], lats[~thresh_fire], s=0.1, c='gray',
               alpha=0.2)
    if thresh_fire.any():
        ax.scatter(lons[thresh_fire], lats[thresh_fire], s=1, c='red',
                   alpha=0.7)
    ax.set_title(f'Threshold Labels ({int(thresh_fire.sum()):,} fire locations)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Bottom-left: Agreement
    ax = axes[1, 0]
    both = ml_fire & thresh_fire
    ml_only = ml_fire & ~thresh_fire
    thresh_only = ~ml_fire & thresh_fire
    neither = ~ml_fire & ~thresh_fire
    ax.scatter(lons[neither], lats[neither], s=0.1, c='gray', alpha=0.1)
    if both.any():
        ax.scatter(lons[both], lats[both], s=1.5, c='green', alpha=0.7,
                   label=f'Both ({int(both.sum()):,})')
    if ml_only.any():
        ax.scatter(lons[ml_only], lats[ml_only], s=1.5, c='blue', alpha=0.7,
                   label=f'ML only ({int(ml_only.sum()):,})')
    if thresh_only.any():
        ax.scatter(lons[thresh_only], lats[thresh_only], s=1.5, c='orange',
                   alpha=0.7, label=f'Thresh only ({int(thresh_only.sum()):,})')
    ax.set_title('Agreement')
    ax.legend(fontsize=9, markerscale=5)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Bottom-right: P(fire) heatmap
    ax = axes[1, 1]
    sc = ax.scatter(lons, lats, s=0.5, c=probs, cmap='hot', vmin=0, vmax=1,
                    alpha=0.7)
    plt.colorbar(sc, ax=ax, label='P(fire)')
    ax.set_title('ML Fire Probability')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    plt.tight_layout()
    outname = f'plots/tune_prediction_map_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    """Train and evaluate ML fire detector.

    Pipeline:
        1. Load HDF data and compute per-location aggregate features
        2. Save/load dataset (skip if exists)
        3. Train/test split with ground truth in both sets
        4. Oversample minority class (fire pixels)
        5. Normalize features with StandardScaler
        6. Train with pixel-wise weighted BCE loss
        7. Evaluate on train and test sets
        8. Save model checkpoint with scaler
        9. Generate diagnostic plots

    Data source: FIREX-AQ campaign (2019), MASTER instrument
        - Flight 03: Pre-burn (ground truth no fire)
        - Flights 04, 05, 06: Active burn flights

    Outputs:
        - checkpoint/fire_detector.pt: Model + scaler
        - dataset/fire_features.pkl.gz: Cached features
        - plots/tune_*.png: Diagnostic visualizations
    """
    print('=' * 60)
    print('ML Fire Detection — Accumulated Observation Features')
    print('=' * 60)

    # Step 1: Load flight data
    print('\n--- Step 1: Building per-location datasets ---')
    flights = group_files_by_flight()
    flight_features = load_all_data(flights)

    # Step 2: Save dataset (skip if already exists)
    print('\n--- Step 2: Dataset ---')
    os.makedirs('dataset', exist_ok=True)
    dataset_path = 'dataset/fire_features.pkl.gz'
    if os.path.exists(dataset_path):
        print(f'  Using existing {dataset_path}')
    else:
        with gzip.open(dataset_path, 'wb') as f:
            pickle.dump(flight_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f'  Saved {dataset_path} ({size_mb:.1f} MB)')

    # Step 3: Train/test split
    # Flight 03 (pre-burn, ground truth no fire) split 80/20 between train/test
    # Train: 80% of 03 + flights 04, 05 (burn flights)
    # Test: 20% of 03 + flight 06 (burn flight)
    print('\n--- Step 3: Train/test split ---')
    print('  Ground truth (flight 03, no fire): 80% train, 20% test')
    train_flights = ['24-801-04', '24-801-05']  # burn flights for training
    test_flights = ['24-801-06']  # burn flight for testing
    X_train, y_train, w_train, X_test, y_test, w_test = extract_train_test(
        flight_features, train_flights, test_flights,
        ground_truth_flight='24-801-03', gt_test_ratio=0.2)
    print(f'  Train: {len(X_train):,} locations '
          f'({int(y_train.sum()):,} fire, '
          f'{len(y_train) - int(y_train.sum()):,} no-fire)')
    print(f'  Test:  {len(X_test):,} locations '
          f'({int(y_test.sum()):,} fire, '
          f'{len(y_test) - int(y_test.sum()):,} no-fire)')

    # Step 4: Oversample
    print('\n--- Step 4: Oversampling fire class ---')
    X_train_bal, y_train_bal, w_train_bal = oversample_minority(
        X_train, y_train, w_train, ratio=1.0)
    print(f'  Balanced: {len(X_train_bal):,} locations '
          f'({int(y_train_bal.sum()):,} fire, '
          f'{len(y_train_bal) - int(y_train_bal.sum()):,} no-fire)')
    print(f'  Weight range: [{w_train_bal.min():.3f}, {w_train_bal.max():.3f}], '
          f'mean={w_train_bal.mean():.3f}')

    # Step 5: Normalize using sklearn StandardScaler on entire dataset
    print('\n--- Step 5: Feature normalization (StandardScaler) ---')
    # Fit scaler on ALL flights for consistent normalization
    all_X = np.concatenate([ff['X'] for ff in flight_features.values()])
    # Replace non-finite values before fitting scaler
    all_X = np.where(np.isfinite(all_X), all_X, 0.0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(all_X)

    # Transform train and test sets
    X_train_bal_clean = np.where(np.isfinite(X_train_bal), X_train_bal, 0.0).astype(np.float32)
    X_test_clean = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)
    X_train_norm = scaler.transform(X_train_bal_clean).astype(np.float32)
    X_test_norm = scaler.transform(X_test_clean).astype(np.float32)

    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name:12s}: mean={scaler.mean_[i]:10.3f}, '
              f'std={scaler.scale_[i]:10.3f}')

    # Step 6: Train with weighted BCE
    print('\n--- Step 6: Training (Weighted BCE Loss, 300 epochs) ---')
    model, loss_history = train_model(
        X_train_norm, y_train_bal, w_train_bal, n_epochs=300, lr=1e-3)

    # Step 7: Evaluate
    print('\n--- Step 7: Evaluation ---')
    X_train_clean = np.where(np.isfinite(X_train), X_train, 0.0).astype(np.float32)
    X_train_orig_norm = scaler.transform(X_train_clean).astype(np.float32)

    print('\n  Training set (flights 03+04+05):')
    train_metrics, _ = evaluate(model, X_train_orig_norm, y_train)
    print_metrics(train_metrics)

    print(f'\n  Test set (flight 06):')
    test_metrics, test_probs = evaluate(model, X_test_norm, y_test)
    print_metrics(test_metrics)

    # Step 8: Save model with scaler for inference
    print('\n--- Step 8: Saving model ---')
    os.makedirs('checkpoint', exist_ok=True)
    model_path = 'checkpoint/fire_detector.pt'
    torch.save({
        'model_state': model.state_dict(),
        'mean': scaler.mean_,
        'std': scaler.scale_,
        'scaler': scaler,  # Save full scaler for sklearn compatibility
        'n_features': 8,
        'threshold': 0.5,
        'feature_names': FEATURE_NAMES,
    }, model_path)
    print(f'  Saved {model_path}')

    # Step 9: Plots
    print('\n--- Step 9: Generating plots ---')
    os.makedirs('plots', exist_ok=True)
    plot_training_loss(loss_history)

    fire_probs = test_probs[y_test == 1]
    nofire_probs = test_probs[y_test == 0]
    if len(fire_probs) > 0:
        plot_probability_hist(fire_probs, nofire_probs)

    plot_prediction_map(flight_features, model, scaler, '24-801-06')

    # Summary with threshold vs ML comparison
    print('\n' + '=' * 60)
    print('Summary:')
    print(f'  Features:  {len(FEATURE_NAMES)} aggregate features')
    print(f'  Model:     8 \u2192 64 \u2192 32 \u2192 1 ({sum(p.numel() for p in model.parameters()):,} params)')

    # Compute ML predictions on test set
    ml_preds = (test_probs >= 0.5).astype(bool)
    thresh_preds = (y_test >= 0.5).astype(bool)

    # Comparison metrics
    n_thresh_fire = int(thresh_preds.sum())
    n_ml_fire = int(ml_preds.sum())
    both_fire = int((ml_preds & thresh_preds).sum())
    ml_only = int((ml_preds & ~thresh_preds).sum())
    thresh_only = int((~ml_preds & thresh_preds).sum())
    both_nofire = int((~ml_preds & ~thresh_preds).sum())

    print('\n  Threshold vs ML Comparison (Test Set):')
    print(f'    Threshold fire detections:  {n_thresh_fire:,}')
    print(f'    ML fire detections:         {n_ml_fire:,}')
    print(f'    Both agree (fire):          {both_fire:,}')
    print(f'    Both agree (no fire):       {both_nofire:,}')
    print(f'    ML only (thresh missed):    {ml_only:,}')
    print(f'    Threshold only (ML missed): {thresh_only:,}')
    agreement = 100.0 * (both_fire + both_nofire) / len(y_test)
    print(f'    Agreement rate:             {agreement:.1f}%')

    print('\n  ML Model Performance (vs threshold labels):')
    print(f'    TP: {test_metrics["TP"]:,}  (ML + threshold both say fire)')
    print(f'    TN: {test_metrics["TN"]:,}  (ML + threshold both say no fire)')
    print(f'    FP: {test_metrics["FP"]:,}  (ML says fire, threshold says no)')
    print(f'    FN: {test_metrics["FN"]:,}  (ML says no fire, threshold says fire)')
    print(f'    Precision: {test_metrics["precision"]:.4f}')
    print(f'    Recall:    {test_metrics["recall"]:.4f}')

    print(f'\n  Checkpoint: {model_path}')
    print(f'  Dataset:    {dataset_path}')
    print(f'\n  To use in realtime_fire.py:')
    print(f'    Model auto-detected from {model_path}')
    print('Done.')


if __name__ == '__main__':
    main()

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

import numpy as np
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


# ── Feature Engineering ───────────────────────────────────────


def build_location_features(pixel_df):
    """Compute 8 aggregate features per grid-cell location from pixel table.

    Groups by (lat, lon) and computes running statistics that mirror
    what process_sweep() maintains in gs accumulators.

    Args:
        pixel_df: DataFrame from build_pixel_table() with columns
                  lat, lon, T4, T11, dT, SWIR, NDVI, fire.

    Returns:
        X: (N_locations, 8) float32 features
        y: (N_locations,) float32 labels (1 if any observation was fire)
        lats, lons: (N_locations,) coordinate arrays
    """
    grouped = pixel_df.groupby(['lat', 'lon'])

    T4_max = grouped['T4'].max().values
    T4_mean = grouped['T4'].mean().values
    T11_mean = grouped['T11'].mean().values
    dT_max = grouped['dT'].max().values

    # NDVI features: only from daytime (finite NDVI) observations
    ndvi_min = grouped['NDVI'].min().values
    ndvi_mean = grouped['NDVI'].mean().values

    # NDVI_drop: first NDVI - min NDVI (vegetation loss over time)
    def ndvi_drop_fn(g):
        finite = g[np.isfinite(g)]
        if len(finite) < 2:
            return 0.0
        return float(finite.iloc[0] - finite.min())
    ndvi_drop = grouped['NDVI'].apply(ndvi_drop_fn).values

    obs_count = grouped['T4'].count().values

    # Label: 1 if any observation at this location was fire
    fire_rate = grouped['fire'].mean().values
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
    lats = coords['lat'].values
    lons = coords['lon'].values

    return X, y, lats, lons


# ── Pixel-Wise Weighting ──────────────────────────────────────


def compute_pixel_weights(y, flight_source, ground_truth_flight='24-801-03'):
    """Compute normalized pixel-wise weights for BCE loss.

    Weight = importance × inverse_frequency, then normalize to mean=1.

    This ensures:
    - Ground truth no-fire pixels (flight 03) are heavily penalized for FP
    - Fire pixels in burn flights are prioritized for recall
    - All categories contribute proportionally regardless of size

    Args:
        y: labels array (0 = no fire, 1 = fire)
        flight_source: array of flight IDs for each sample
        ground_truth_flight: flight ID with known no-fire ground truth

    Returns:
        weights: array same shape as y, normalized so mean=1
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


def load_all_data(flights):
    """Build pixel tables for all flights and compute location features.

    Returns:
        flight_features: {flight_num: {'X': arr, 'y': arr, 'lats': arr,
                          'lons': arr, 'pixel_df': DataFrame}}
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


def extract_train_test(flight_features, train_flights, test_flights,
                       ground_truth_flight='24-801-03', gt_test_ratio=0.2):
    """Split ground truth flight between train/test, then add burn flights.

    Flight 03 (pre-burn, no fire) is split 80/20 between train/test.
    This ensures test set has ground truth "no fire" data for proper FP evaluation.

    Also computes pixel-wise weights using importance × inverse-frequency.

    Args:
        flight_features: dict from load_all_data()
        train_flights: list of flight numbers for training (burn flights)
        test_flights: list of flight numbers for testing (burn flights)
        ground_truth_flight: flight with known no-fire ground truth
        gt_test_ratio: fraction of ground truth to put in test set

    Returns:
        X_train, y_train, w_train, X_test, y_test, w_test
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


def oversample_minority(X, y, w, ratio=1.0):
    """Oversample fire locations to balance training data, preserving weights.

    After oversampling, weights are re-normalized so mean=1.

    Args:
        X: feature array
        y: label array
        w: weight array (pixel-wise weights)
        ratio: target fire:no-fire ratio

    Returns:
        X_bal, y_bal, w_bal: balanced arrays with shuffled order
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

    Architecture: 8 → 64 → 32 → 1 (2,465 parameters).
    """

    def __init__(self, n_features=8, hidden1=64, hidden2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────


def get_device():
    """Get best available device: MPS (Mac), CUDA (NVIDIA), or CPU."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_model(X_train, y_train, w_train, n_epochs=300, lr=1e-3, batch_size=1024):
    """Train FireMLP with pixel-wise weighted BCEWithLogitsLoss.

    Uses MPS on Mac, CUDA on NVIDIA, or CPU as fallback.

    The loss for each sample is weighted by w_train, which encodes:
    - Ground truth no-fire pixels: high weight (penalize FP heavily)
    - Fire pixels in burn flights: medium-high weight (capture real fires)
    - Other pixels: baseline weight

    Args:
        X_train: normalized feature array (N, 8)
        y_train: label array (N,)
        w_train: pixel-wise weights (N,), normalized so mean=1
        n_epochs: number of training epochs
        lr: learning rate
        batch_size: batch size for training

    Returns:
        model: trained FireMLP (moved back to CPU)
        loss_history: list of per-epoch average loss values
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
    for epoch in trange(n_epochs, desc=f"Training Model, Weighted BCE"):
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

    # Move model back to CPU for saving and inference
    model = model.cpu()
    return model, loss_history


# ── Evaluation ────────────────────────────────────────────────


def evaluate(model, X, y, threshold=0.5):
    """Evaluate with absolute count metrics."""
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


def print_metrics(metrics, label=''):
    """Print evaluation metrics."""
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


def plot_training_loss(loss_history):
    """Plot BCE training loss curve."""
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


def plot_probability_hist(probs_fire, probs_nofire):
    """Plot P(fire) histogram for fire vs non-fire locations."""
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


def plot_prediction_map(flight_features, model, scaler, flight_num):
    """Plot spatial fire predictions for one flight.

    2x2: ML pred, threshold pred, agreement, P(fire) heatmap.
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


def main():
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

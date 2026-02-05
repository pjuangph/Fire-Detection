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


def extract_train_test(flight_features, train_flights, test_flights):
    """Concatenate features from specified flights."""
    def concat_flights(fnums):
        Xs, ys = [], []
        for f in fnums:
            Xs.append(flight_features[f]['X'])
            ys.append(flight_features[f]['y'])
        return np.concatenate(Xs), np.concatenate(ys)

    X_train, y_train = concat_flights(train_flights)
    X_test, y_test = concat_flights(test_flights)
    return X_train, y_train, X_test, y_test


def oversample_minority(X, y, ratio=1.0):
    """Oversample fire locations to balance training data."""
    fire_mask = y == 1
    n_fire = int(fire_mask.sum())
    n_nofire = int((~fire_mask).sum())
    if n_fire == 0 or n_fire >= n_nofire * ratio:
        return X, y

    target_fire = int(n_nofire * ratio)
    rng = np.random.default_rng(42)
    repeat_idx = rng.choice(
        np.where(fire_mask)[0], target_fire - n_fire, replace=True)

    X_bal = np.concatenate([X, X[repeat_idx]], axis=0)
    y_bal = np.concatenate([y, y[repeat_idx]], axis=0)
    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


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


def train_model(X_train, y_train, n_epochs=300, lr=1e-3, batch_size=4096):
    """Train FireMLP with BCEWithLogitsLoss.

    Returns:
        model: trained FireMLP
        loss_history: list of per-epoch average loss values
    """
    model = FireMLP()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(X_t)
        loss_history.append(avg_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:4d}/{n_epochs}: '
                  f'BCE Loss = {avg_loss:.4f}')

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


def plot_prediction_map(flight_features, model, train_mean, train_std,
                        flight_num):
    """Plot spatial fire predictions for one flight.

    2x2: ML pred, threshold pred, agreement, P(fire) heatmap.
    """
    d = flight_features[flight_num]
    X, y = d['X'], d['y']
    lats, lons = d['lats'], d['lons']

    X_norm = (X - train_mean) / train_std
    X_norm = np.where(np.isfinite(X_norm), X_norm, 0.0).astype(np.float32)

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

    # Step 2: Save dataset
    print('\n--- Step 2: Saving dataset ---')
    os.makedirs('dataset', exist_ok=True)
    dataset_path = 'dataset/fire_features.pkl.gz'
    with gzip.open(dataset_path, 'wb') as f:
        pickle.dump(flight_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
    print(f'  Saved {dataset_path} ({size_mb:.1f} MB)')

    # Step 3: Train/test split
    print('\n--- Step 3: Train/test split ---')
    train_flights = ['24-801-03', '24-801-04', '24-801-05']
    test_flights = ['24-801-06']
    X_train, y_train, X_test, y_test = extract_train_test(
        flight_features, train_flights, test_flights)
    print(f'  Train: {len(X_train):,} locations '
          f'({int(y_train.sum()):,} fire, '
          f'{len(y_train) - int(y_train.sum()):,} no-fire)')
    print(f'  Test:  {len(X_test):,} locations '
          f'({int(y_test.sum()):,} fire, '
          f'{len(y_test) - int(y_test.sum()):,} no-fire)')

    # Step 4: Oversample
    print('\n--- Step 4: Oversampling fire class ---')
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train, ratio=1.0)
    print(f'  Balanced: {len(X_train_bal):,} locations '
          f'({int(y_train_bal.sum()):,} fire, '
          f'{len(y_train_bal) - int(y_train_bal.sum()):,} no-fire)')

    # Step 5: Normalize
    print('\n--- Step 5: Feature normalization ---')
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    # Prevent division by zero for constant features
    train_std = np.where(train_std > 0, train_std, 1.0)
    X_train_norm = (X_train_bal - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    # Replace any non-finite values after normalization
    X_train_norm = np.where(np.isfinite(X_train_norm), X_train_norm, 0.0).astype(np.float32)
    X_test_norm = np.where(np.isfinite(X_test_norm), X_test_norm, 0.0).astype(np.float32)

    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name:12s}: mean={train_mean[i]:10.3f}, '
              f'std={train_std[i]:10.3f}')

    # Step 6: Train
    print('\n--- Step 6: Training (BCE Loss, 300 epochs) ---')
    model, loss_history = train_model(
        X_train_norm, y_train_bal, n_epochs=300, lr=1e-3)

    # Step 7: Evaluate
    print('\n--- Step 7: Evaluation ---')
    X_train_orig_norm = (X_train - train_mean) / train_std
    X_train_orig_norm = np.where(
        np.isfinite(X_train_orig_norm), X_train_orig_norm, 0.0
    ).astype(np.float32)

    print('\n  Training set (flights 03+04+05):')
    train_metrics, _ = evaluate(model, X_train_orig_norm, y_train)
    print_metrics(train_metrics)

    print(f'\n  Test set (flight 06):')
    test_metrics, test_probs = evaluate(model, X_test_norm, y_test)
    print_metrics(test_metrics)

    # Step 8: Save model
    print('\n--- Step 8: Saving model ---')
    os.makedirs('checkpoint', exist_ok=True)
    model_path = 'checkpoint/fire_detector.pt'
    torch.save({
        'model_state': model.state_dict(),
        'mean': train_mean,
        'std': train_std,
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

    plot_prediction_map(
        flight_features, model, train_mean, train_std, '24-801-06')

    # Summary
    print('\n' + '=' * 60)
    print('Summary:')
    print(f'  Features:  {len(FEATURE_NAMES)} aggregate features')
    print(f'  Model:     8 \u2192 64 \u2192 32 \u2192 1 ({sum(p.numel() for p in model.parameters()):,} params)')
    print(f'  Test TP:   {test_metrics["TP"]:,}')
    print(f'  Test FP:   {test_metrics["FP"]:,}')
    print(f'  Test FN:   {test_metrics["FN"]:,}')
    print(f'  Checkpoint: {model_path}')
    print(f'  Dataset:    {dataset_path}')
    print(f'\n  To use in realtime_fire.py:')
    print(f'    Model auto-detected from {model_path}')
    print('Done.')


if __name__ == '__main__':
    main()

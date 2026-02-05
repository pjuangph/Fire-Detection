"""fire_ml.py - ML-based fire detection using Dice Loss.

Trains a small neural network to classify fire vs. non-fire pixels from
MASTER L1B data using 4 features:
    - T4:   Brightness temperature at 3.9 μm (fire channel) [K]
    - T11:  Brightness temperature at 11.25 μm (background channel) [K]
    - ΔT:   T4 − T11 spectral difference [K]
    - SWIR: Radiance at 2.16 μm (solar reflection channel) [W/m²/sr/μm]

SWIR helps distinguish solar reflection false positives from real fire:
sun-heated rock reflects strongly in SWIR, while fire emission at 2.16 μm
is relatively low compared to its 3.9 μm signal.

The loss function is Soft Dice Loss:
    Loss = 1 - 2·TP / (2·TP + FP + FN)

This loss operates on absolute TP/FP/FN counts — total pixel count never
appears, so it is not diluted by adding more background pixels. Every
false positive and every missed fire directly degrades the score.

Train/test split is by flight:
    Train: flights 03 (pre-burn), 04 (day burn), 05 (night burn)
    Test:  flight 06 (day burn, unseen)

Labels are pseudo-labels from the threshold detector (T4 > 325K/310K AND
ΔT > 10K), so the ML model learns a more nuanced decision boundary than
the hard thresholds.

Usage:
    python fire_ml.py
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from lib import group_files_by_flight, compute_grid_extent, build_mosaic


# ── Data Pipeline ──────────────────────────────────────────────


def load_flight_data(flights):
    """Build mosaics for all flights, return dict of grids and metadata.

    Returns:
        {flight_num: {'T4': grid, 'T11': grid, 'dT': grid,
                      'labels': grid, 'lat_axis': arr, 'lon_axis': arr,
                      'day_night': str, 'comment': str}}
    """
    flight_data = {}
    for fnum, info in sorted(flights.items()):
        files = info['files']
        day_night = info['day_night']
        print(f'  Flight {fnum} ({len(files)} files, {day_night})...')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        mosaic = build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night)
        grid_dT = mosaic['T4'] - mosaic['T11']

        valid = np.sum(np.isfinite(mosaic['T4']))
        fire = np.sum(mosaic['fire'])
        print(f'    {valid:,} valid pixels, {fire:,} fire labels')

        flight_data[fnum] = {
            'T4': mosaic['T4'], 'T11': mosaic['T11'],
            'SWIR': mosaic['SWIR'], 'NDVI': mosaic['NDVI'], 'dT': grid_dT,
            'labels': mosaic['fire'].astype(np.float32),
            'lat_axis': mosaic['lat_axis'], 'lon_axis': mosaic['lon_axis'],
            'day_night': day_night, 'comment': info['comment'],
        }
    return flight_data


def extract_pixels(flight_data, flight_nums):
    """Extract valid pixels from specified flights into flat arrays.

    Returns:
        X: (N, 4) float32 — features [T4, T11, ΔT, SWIR]
        y: (N,) float32 — binary labels {0, 1}
    """
    all_X, all_y = [], []
    for fnum in flight_nums:
        d = flight_data[fnum]
        valid = (np.isfinite(d['T4']) & np.isfinite(d['T11']) &
                 np.isfinite(d['SWIR']))
        T4 = d['T4'][valid]
        T11 = d['T11'][valid]
        dT = d['dT'][valid]
        SWIR = d['SWIR'][valid]
        labels = d['labels'][valid]
        features = np.stack([T4, T11, dT, SWIR], axis=1)
        all_X.append(features)
        all_y.append(labels)
    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    assert not np.any(np.isnan(X)), 'NaN in features after filtering'
    return X, y


def oversample_minority(X, y, ratio=1.0):
    """Oversample the minority (fire) class to balance training data.

    Args:
        ratio: target fire/no-fire ratio. 1.0 = equal counts.

    Returns:
        X_bal, y_bal: balanced arrays with fire pixels repeated.
    """
    fire_mask = y == 1
    n_fire = int(fire_mask.sum())
    n_nofire = int((~fire_mask).sum())
    if n_fire == 0 or n_fire >= n_nofire * ratio:
        return X, y

    target_fire = int(n_nofire * ratio)
    rng = np.random.default_rng(42)
    repeat_idx = rng.choice(np.where(fire_mask)[0], target_fire - n_fire, replace=True)

    X_bal = np.concatenate([X, X[repeat_idx]], axis=0)
    y_bal = np.concatenate([y, y[repeat_idx]], axis=0)

    # Shuffle
    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


# ── Model ──────────────────────────────────────────────────────


class FireDetector(nn.Module):
    """Pixel-level fire detection MLP.

    Architecture: 4 → 64 → 32 → 1 (2,337 parameters).
    Learns a nonlinear decision boundary in (T4, T11, ΔT, SWIR) space.
    """

    def __init__(self, n_features=4, hidden1=64, hidden2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        # With oversampled training data, class prior is ~50/50 so
        # default zero bias is appropriate. No special initialization needed.

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Loss ───────────────────────────────────────────────────────


class SoftDiceLoss(nn.Module):
    """Differentiable Dice Loss for binary classification.

    Loss = 1 - (2·TP + smooth) / (2·TP + FP + FN + smooth)

    Soft TP/FP/FN use sigmoid probabilities, making the loss
    fully differentiable for gradient descent.

    True Negatives do NOT appear in the formula — the loss is
    not diluted by background pixel count. 100 FP out of 1,000
    pixels gives the same loss as 100 FP out of 1,000,000 pixels.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()

        dice = (2.0 * tp + self.smooth) / (2.0 * tp + fp + fn + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss.

    Dice Loss drives the global TP/FP/FN metric toward optimal.
    BCE provides per-pixel gradient signals that help early training
    converge when Dice alone gets stuck in a local minimum.

    Loss = dice_weight * DiceLoss + bce_weight * BCE
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice(logits, targets) +
                self.bce_weight * self.bce(logits, targets))


# ── Training ───────────────────────────────────────────────────


def train_model(X_train, y_train, n_epochs=200, lr=1e-3, batch_size=65536):
    """Train FireDetector with Dice+BCE Loss.

    Uses mini-batch training with large batches (64K) for stable
    Dice Loss gradients while remaining memory-efficient.

    Returns:
        model: trained FireDetector
        loss_history: list of per-epoch average loss values
    """
    model = FireDetector()
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5, smooth=1.0)
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
                  f'Loss = {avg_loss:.4f}')

    return model, loss_history


# ── Evaluation ─────────────────────────────────────────────────


def evaluate(model, X, y, threshold=0.5):
    """Evaluate model with absolute count metrics.

    Returns:
        metrics: dict with TP, FP, FN, TN, precision, recall, dice_score
        probs: (N,) predicted probabilities
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
    dice = 2 * TP / max(2 * TP + FP + FN, 1)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall, 'dice_score': dice,
    }, probs


def print_metrics(metrics, label=''):
    """Print evaluation metrics with absolute counts."""
    m = metrics
    header = f'  {label} ' if label else '  '
    print(f'{header}Absolute counts:')
    print(f'    TP (correct fire):     {m["TP"]:>8,}')
    print(f'    FP (false alarm):      {m["FP"]:>8,}')
    print(f'    FN (missed fire):      {m["FN"]:>8,}')
    print(f'    TN (correct no-fire):  {m["TN"]:>8,}')
    total = m['TP'] + m['FP'] + m['FN'] + m['TN']
    print(f'    Total:                 {total:>8,}')
    print(f'{header}Rates:')
    print(f'    Precision: TP/(TP+FP) = {m["precision"]:.4f} ({m["precision"]:.1%})')
    print(f'    Recall:    TP/(TP+FN) = {m["recall"]:.4f} ({m["recall"]:.1%})')
    print(f'    Dice:  2TP/(2TP+FP+FN) = {m["dice_score"]:.4f} ({m["dice_score"]:.1%})')


def sklearn_baseline(X_train, y_train, X_test, y_test):
    """Logistic regression baseline for comparison."""
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    TP = int(np.sum((preds == 1) & (y_test == 1)))
    FP = int(np.sum((preds == 1) & (y_test == 0)))
    FN = int(np.sum((preds == 0) & (y_test == 1)))
    TN = int(np.sum((preds == 0) & (y_test == 0)))

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    dice = 2 * TP / max(2 * TP + FP + FN, 1)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall, 'dice_score': dice,
    }


# ── Visualization ──────────────────────────────────────────────


def plot_training_loss(loss_history):
    """Plot Dice Loss training curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', linewidth=1.5, label='Dice Loss')
    ax.plot(epochs, [1 - l for l in loss_history], 'r-', linewidth=1.5,
            alpha=0.7, label='Dice Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Training Convergence — Soft Dice Loss')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    plt.savefig('plots/ml_training_loss.png', dpi=150, bbox_inches='tight')
    print('  Saved plots/ml_training_loss.png')
    plt.close()


def plot_decision_boundary(model, train_mean, train_std, X_test, y_test,
                           day_night='D'):
    """Plot learned decision boundary in T4 vs ΔT space.

    Shows the ML boundary (colored contour) alongside the hard threshold
    lines used by the current detector, with scatter data overlaid.
    """
    T4_range = np.linspace(250, 750, 400)
    dT_range = np.linspace(-20, 400, 350)
    T4_grid, dT_grid = np.meshgrid(T4_range, dT_range)
    T11_grid = T4_grid - dT_grid
    # Use mean SWIR for the 2D decision boundary slice
    swir_grid = np.full_like(T4_grid, train_mean[3])

    feat_grid = np.stack([T4_grid.ravel(), T11_grid.ravel(), dT_grid.ravel(),
                          swir_grid.ravel()], axis=1).astype(np.float32)
    feat_norm = (feat_grid - train_mean) / train_std

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(feat_norm))
        prob_grid = torch.sigmoid(logits).numpy().reshape(T4_grid.shape)

    fig, ax = plt.subplots(figsize=(10, 8))

    cf = ax.contourf(T4_range, dT_range, prob_grid, levels=20,
                     cmap='RdYlBu_r', alpha=0.4)
    plt.colorbar(cf, ax=ax, label='P(fire)')
    ax.contour(T4_range, dT_range, prob_grid, levels=[0.5],
               colors='blue', linewidths=2, linestyles='-')

    # Subsample scatter data
    rng = np.random.default_rng(42)
    T4_vals = X_test[:, 0]
    dT_vals = X_test[:, 2]

    bg_mask = y_test == 0
    fire_mask = y_test == 1
    n_bg = min(5000, int(bg_mask.sum()))
    if n_bg > 0:
        idx = rng.choice(int(bg_mask.sum()), n_bg, replace=False)
        ax.scatter(T4_vals[bg_mask][idx], dT_vals[bg_mask][idx],
                   s=1, c='gray', alpha=0.3, label='Background')
    n_fire = min(5000, int(fire_mask.sum()))
    if n_fire > 0:
        idx = rng.choice(int(fire_mask.sum()), n_fire, replace=False)
        ax.scatter(T4_vals[fire_mask][idx], dT_vals[fire_mask][idx],
                   s=3, c='red', alpha=0.7, label=f'Fire ({int(fire_mask.sum()):,})')

    T4_thresh = 310.0 if day_night == 'N' else 325.0
    ax.axvline(T4_thresh, color='orange', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Threshold T4={T4_thresh:.0f} K')
    ax.axhline(10, color='green', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Threshold ΔT=10 K')

    ax.set_xlabel('T4 [K]')
    ax.set_ylabel('ΔT = T4 − T11 [K]')
    ax.set_title('ML Decision Boundary vs Threshold Detection')
    ax.set_xlim(250, 750)
    ax.set_ylim(-20, 400)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig('plots/ml_decision_boundary.png', dpi=150, bbox_inches='tight')
    print('  Saved plots/ml_decision_boundary.png')
    plt.close()


def plot_prediction_map(flight_data, model, train_mean, train_std, flight_num):
    """Plot spatial fire predictions on a flight mosaic.

    2x2 layout:
        Top-left:     ML fire predictions (red) on gray T4 background
        Top-right:    Threshold fire labels (red) on gray T4 background
        Bottom-left:  Agreement/disagreement map
        Bottom-right: ML probability heatmap
    """
    d = flight_data[flight_num]
    grid_shape = d['T4'].shape
    valid = (np.isfinite(d['T4']) & np.isfinite(d['T11']) &
             np.isfinite(d['SWIR']))

    features = np.stack([d['T4'][valid], d['T11'][valid], d['dT'][valid],
                         d['SWIR'][valid]], axis=1).astype(np.float32)
    features_norm = (features - train_mean) / train_std

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features_norm))
        probs = torch.sigmoid(logits).numpy()

    prob_grid = np.full(grid_shape, np.nan, dtype=np.float32)
    prob_grid[valid] = probs
    ml_fire = np.zeros(grid_shape, dtype=bool)
    ml_fire[valid] = probs >= 0.5

    thresh_fire = d['labels'] > 0.5
    lat_axis = d['lat_axis']
    lon_axis = d['lon_axis']
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    valid_T4 = d['T4'][np.isfinite(d['T4'])]
    vmin, vmax = (np.percentile(valid_T4, [2, 98]) if len(valid_T4) > 0
                  else (280, 320))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'ML vs Threshold — Flight {flight_num} ({d["comment"]})',
                 fontsize=14, fontweight='bold')

    # Top-left: ML predictions
    ax = axes[0, 0]
    ax.imshow(d['T4'], extent=extent, aspect='equal', cmap='gray',
              vmin=vmin, vmax=vmax)
    ml_count = np.sum(ml_fire)
    if ml_count > 0:
        rows, cols = np.where(ml_fire)
        lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * rows / (len(lat_axis) - 1)
        lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * cols / (len(lon_axis) - 1)
        ax.scatter(lons, lats, s=0.3, c='red', alpha=0.7,
                   label=f'ML fire ({ml_count:,})')
        ax.legend(loc='upper right', markerscale=15, fontsize=9)
    ax.set_title(f'ML Predictions ({ml_count:,} pixels)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Top-right: Threshold predictions
    ax = axes[0, 1]
    ax.imshow(d['T4'], extent=extent, aspect='equal', cmap='gray',
              vmin=vmin, vmax=vmax)
    thresh_count = np.sum(thresh_fire)
    if thresh_count > 0:
        rows, cols = np.where(thresh_fire)
        lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * rows / (len(lat_axis) - 1)
        lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * cols / (len(lon_axis) - 1)
        ax.scatter(lons, lats, s=0.3, c='red', alpha=0.7,
                   label=f'Threshold ({thresh_count:,})')
        ax.legend(loc='upper right', markerscale=15, fontsize=9)
    ax.set_title(f'Threshold Labels ({thresh_count:,} pixels)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Bottom-left: Agreement/disagreement
    ax = axes[1, 0]
    agree_grid = np.full(grid_shape + (3,), 0.2, dtype=np.float32)  # dark gray default
    # Both detect (green)
    both = ml_fire & thresh_fire
    agree_grid[both] = [0.0, 0.8, 0.0]
    # ML only (blue) — ML detects, threshold does not
    ml_only = ml_fire & ~thresh_fire
    agree_grid[ml_only] = [0.2, 0.4, 1.0]
    # Threshold only (orange) — threshold detects, ML misses
    thresh_only = ~ml_fire & thresh_fire
    agree_grid[thresh_only] = [1.0, 0.5, 0.0]
    # Not data (white)
    agree_grid[~valid] = [1.0, 1.0, 1.0]

    ax.imshow(agree_grid, extent=extent, aspect='equal')
    ax.set_title(f'Agreement: green=both ({np.sum(both):,}), '
                 f'blue=ML only ({np.sum(ml_only):,}), '
                 f'orange=thresh only ({np.sum(thresh_only):,})')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Bottom-right: ML probability heatmap
    ax = axes[1, 1]
    im = ax.imshow(prob_grid, extent=extent, aspect='equal',
                   cmap='hot', vmin=0, vmax=1)
    ax.set_title('ML Fire Probability')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P(fire)')

    plt.tight_layout()
    outname = f'plots/ml_prediction_map_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()


def plot_fp_fn_comparison(flight_data, model, train_mean, train_std,
                          flight_num):
    """Plot spatial locations of FP and FN pixels for a flight.

    Compared against the threshold pseudo-labels:
        - TP (green): both ML and threshold detect
        - FP (blue): ML detects but threshold does not
        - FN (orange): threshold detects but ML misses
    """
    d = flight_data[flight_num]
    grid_shape = d['T4'].shape
    valid = (np.isfinite(d['T4']) & np.isfinite(d['T11']) &
             np.isfinite(d['SWIR']))

    features = np.stack([d['T4'][valid], d['T11'][valid], d['dT'][valid],
                         d['SWIR'][valid]], axis=1).astype(np.float32)
    features_norm = (features - train_mean) / train_std

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features_norm))
        probs = torch.sigmoid(logits).numpy()

    ml_fire = np.zeros(grid_shape, dtype=bool)
    ml_fire[valid] = probs >= 0.5
    thresh_fire = d['labels'] > 0.5

    tp_mask = ml_fire & thresh_fire
    fp_mask = ml_fire & ~thresh_fire
    fn_mask = ~ml_fire & thresh_fire

    lat_axis = d['lat_axis']
    lon_axis = d['lon_axis']
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]
    valid_T4 = d['T4'][np.isfinite(d['T4'])]
    vmin, vmax = (np.percentile(valid_T4, [2, 98]) if len(valid_T4) > 0
                  else (280, 320))

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(d['T4'], extent=extent, aspect='equal', cmap='gray',
              vmin=vmin, vmax=vmax)

    def scatter_mask(mask, color, label, s=0.5):
        if np.sum(mask) > 0:
            rows, cols = np.where(mask)
            lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * rows / (len(lat_axis) - 1)
            lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * cols / (len(lon_axis) - 1)
            ax.scatter(lons, lats, s=s, c=color, alpha=0.7, label=label)

    scatter_mask(tp_mask, 'green', f'TP — both detect ({np.sum(tp_mask):,})')
    scatter_mask(fn_mask, 'orange', f'FN — ML misses ({np.sum(fn_mask):,})', s=2)
    scatter_mask(fp_mask, 'blue', f'FP — ML only ({np.sum(fp_mask):,})', s=2)

    ax.set_title(f'FP/FN Map — Flight {flight_num}\n'
                 f'TP={np.sum(tp_mask):,}  FP={np.sum(fp_mask):,}  '
                 f'FN={np.sum(fn_mask):,}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', markerscale=15, fontsize=9)

    plt.tight_layout()
    outname = f'plots/ml_fpfn_comparison_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()


# ── Main ───────────────────────────────────────────────────────


def main():
    print('=' * 60)
    print('ML Fire Detection with Dice Loss')
    print('=' * 60)

    # Step 1: Load all flight mosaics
    print('\n--- Step 1: Building flight mosaics ---')
    flights = group_files_by_flight()
    flight_data = load_flight_data(flights)

    # Step 2: Train/test split by flight
    print('\n--- Step 2: Train/test split ---')
    train_flights = ['24-801-03', '24-801-04', '24-801-05']
    test_flights = ['24-801-06']
    X_train, y_train = extract_pixels(flight_data, train_flights)
    X_test, y_test = extract_pixels(flight_data, test_flights)
    print(f'  Train: {len(X_train):,} pixels '
          f'({int(y_train.sum()):,} fire, {len(y_train) - int(y_train.sum()):,} no-fire) '
          f'from flights {train_flights}')
    print(f'  Test:  {len(X_test):,} pixels '
          f'({int(y_test.sum()):,} fire, {len(y_test) - int(y_test.sum()):,} no-fire) '
          f'from flights {test_flights}')

    # Step 3: Oversample fire pixels for balanced training
    print('\n--- Step 3: Oversampling fire class ---')
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train, ratio=1.0)
    print(f'  Balanced train: {len(X_train_bal):,} pixels '
          f'({int(y_train_bal.sum()):,} fire, '
          f'{len(y_train_bal) - int(y_train_bal.sum()):,} no-fire)')

    # Step 4: Normalize features using original train statistics
    print('\n--- Step 4: Feature normalization ---')
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    X_train_norm = (X_train_bal - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std
    print(f'  Train mean: T4={train_mean[0]:.1f} K, '
          f'T11={train_mean[1]:.1f} K, ΔT={train_mean[2]:.1f} K, '
          f'SWIR={train_mean[3]:.4f} W/m²/sr/μm')
    print(f'  Train std:  T4={train_std[0]:.1f} K, '
          f'T11={train_std[1]:.1f} K, ΔT={train_std[2]:.1f} K, '
          f'SWIR={train_std[3]:.4f} W/m²/sr/μm')

    # Step 5: Unit test Dice Loss
    print('\n--- Step 5: Dice Loss unit tests ---')
    criterion = SoftDiceLoss(smooth=1.0)
    test_cases = [
        ('Perfect',     torch.tensor([10., 10., -10., -10.]),
                        torch.tensor([1., 1., 0., 0.])),
        ('All wrong',   torch.tensor([-10., -10., 10., 10.]),
                        torch.tensor([1., 1., 0., 0.])),
        ('Half right',  torch.tensor([10., -10., -10., -10.]),
                        torch.tensor([1., 1., 0., 0.])),
    ]
    for name, logits, targets in test_cases:
        loss = criterion(logits, targets)
        print(f'  {name:12s}: Loss = {loss.item():.4f}, '
              f'Dice = {1 - loss.item():.4f}')

    # Step 6: Train model on balanced data
    print('\n--- Step 6: Training (Dice+BCE Loss, 500 epochs, balanced) ---')
    model, loss_history = train_model(X_train_norm, y_train_bal, n_epochs=500, lr=1e-3)

    # Step 7: Evaluate on original (unbalanced) train and test sets
    print('\n--- Step 7: Evaluation ---')
    # Evaluate on ORIGINAL train data (not oversampled) for fair comparison
    X_train_orig_norm = (X_train - train_mean) / train_std
    print('\n  Training set (flights 03+04+05):')
    train_metrics, _ = evaluate(model, X_train_orig_norm, y_train)
    print_metrics(train_metrics)

    print(f'\n  Test set (flight 06):')
    test_metrics, test_probs = evaluate(model, X_test_norm, y_test)
    print_metrics(test_metrics)

    # Step 8: Sklearn baseline
    print('\n--- Step 8: Logistic Regression baseline ---')
    baseline_metrics = sklearn_baseline(X_train_orig_norm, y_train, X_test_norm, y_test)
    print_metrics(baseline_metrics, label='Baseline')

    print('\n  Comparison (test set):')
    print(f'    {"Metric":<12s} {"ML (Dice)":>10s} {"Baseline":>10s}')
    print(f'    {"─" * 12} {"─" * 10} {"─" * 10}')
    for key in ['TP', 'FP', 'FN', 'precision', 'recall', 'dice_score']:
        ml_val = test_metrics[key]
        bl_val = baseline_metrics[key]
        if isinstance(ml_val, int):
            print(f'    {key:<12s} {ml_val:>10,} {bl_val:>10,}')
        else:
            print(f'    {key:<12s} {ml_val:>10.4f} {bl_val:>10.4f}')

    # Step 9: Plots
    print('\n--- Step 9: Generating plots ---')
    os.makedirs('plots', exist_ok=True)
    plot_training_loss(loss_history)
    plot_decision_boundary(model, train_mean, train_std, X_test, y_test,
                           day_night=flight_data['24-801-06']['day_night'])
    plot_prediction_map(flight_data, model, train_mean, train_std, '24-801-06')
    plot_fp_fn_comparison(flight_data, model, train_mean, train_std, '24-801-06')

    print('\nDone.')


if __name__ == '__main__':
    main()

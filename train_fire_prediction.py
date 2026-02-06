"""train_fire_prediction.py - Train MLP fire detector with accumulated observations.

Trains an MLP that predicts fire from 12 aggregate features computed across
all observations of each pixel. See lib/features.py for feature definitions.

Usage:
    python train_fire_prediction.py                    # default: BCE loss
    python train_fire_prediction.py --loss error-rate   # soft (FN+FP)/P loss
"""

from __future__ import annotations

import argparse
import os
import pickle
import gzip
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from lib import (
    group_files_by_flight, compute_grid_extent, build_pixel_table,
)
from lib.features import build_location_features
from lib.losses import SoftErrorRateLoss, compute_pixel_weights
from lib.inference import FireMLP, FEATURE_NAMES
from lib.evaluation import get_device, evaluate, print_metrics
from lib.plotting import (
    plot_training_loss, plot_probability_hist, plot_prediction_map,
)

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]


# ── Data Pipeline ─────────────────────────────────────────────


def load_all_data(
    flights: dict[str, dict[str, Any]],
) -> FlightFeatures:
    """Build pixel tables for all flights and compute location features.

    Args:
        flights: Flight metadata from group_files_by_flight().

    Returns:
        Dict mapping flight_num to feature dict with 'X', 'y', 'lats',
        'lons', 'pixel_df', 'day_night', 'comment'.
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
    Ground truth labels are forced to 0 (no real fire on pre-burn flight).

    Returns:
        Tuple of (X_train, y_train, w_train, X_test, y_test, w_test).
    """
    rng = np.random.default_rng(42)

    # Split ground truth flight (03) 80/20
    gt = flight_features[ground_truth_flight]
    n_gt = len(gt['X'])
    n_gt_test = int(n_gt * gt_test_ratio)
    perm = rng.permutation(n_gt)
    gt_test_idx = perm[:n_gt_test]
    gt_train_idx = perm[n_gt_test:]

    # Train: 80% of ground truth + burn flights
    # Force ground truth labels to 0 (no real fire on pre-burn)
    train_X = [gt['X'][gt_train_idx]]
    train_y = [np.zeros(len(gt_train_idx), dtype=np.float32)]
    train_flight_src = [np.full(len(gt_train_idx), ground_truth_flight)]
    for f in train_flights:
        if f != ground_truth_flight:
            train_X.append(flight_features[f]['X'])
            train_y.append(flight_features[f]['y'])
            train_flight_src.append(np.full(len(flight_features[f]['y']), f))

    # Test: 20% of ground truth + burn flights
    test_X = [gt['X'][gt_test_idx]]
    test_y = [np.zeros(len(gt_test_idx), dtype=np.float32)]
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
    """Oversample fire locations to balance training data, preserving weights."""
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


# ── Training ─────────────────────────────────────────────────


def train_model(
    X_train: NDArrayFloat,
    y_train: NDArrayFloat,
    w_train: NDArrayFloat,
    n_epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 4096,
    loss_fn: str = 'bce',
    hidden_layers: list[int] | None = None,
) -> tuple[FireMLP, NDArrayFloat]:
    """Train FireMLP with the selected loss function.

    Loss functions:
        - 'bce': Pixel-wise weighted BCEWithLogitsLoss
        - 'error-rate': Soft differentiable (FN+FP)/P

    Returns:
        Tuple of (model, loss_history).
    """
    device = get_device()
    print(f'  Using device: {device}')

    model = FireMLP(hidden_layers=hidden_layers).to(device)
    use_error_rate = loss_fn == 'error-rate'
    if use_error_rate:
        P_total = float(y_train.sum())
        criterion = SoftErrorRateLoss(P_total).to(device)
        loss_label = f'Soft (FN+FP)/P, P={P_total:.0f}'
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_label = 'Pixel Weighted BCE'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    w_t = torch.tensor(w_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    loss_history = np.zeros((n_epochs, 1))
    pbar = trange(n_epochs, desc=f'Training Model, {loss_label}')
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for X_batch, y_batch, w_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            if use_error_rate:
                loss = criterion(logits, y_batch, w_batch)
            else:
                loss_per_sample = criterion(logits, y_batch)
                loss = (loss_per_sample * w_batch).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(X_t)
        loss_history[epoch] = avg_loss
        pbar.set_postfix({'loss': f'{avg_loss:.4e}'})

    model = model.cpu()
    return model, loss_history


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    """Train and evaluate ML fire detector."""
    parser = argparse.ArgumentParser(
        description='Train MLP fire detector with accumulated observations')
    parser.add_argument(
        '--loss', choices=['bce', 'error-rate'], default='bce',
        help='Loss function: "bce" (weighted BCE) or "error-rate" (soft (FN+FP)/P)')
    parser.add_argument(
        '--layers', type=int, nargs='+', default=[64, 32],
        help='Hidden layer sizes, e.g. --layers 64 32 16 8 (default: 64 32)')
    args = parser.parse_args()

    arch_str = ' -> '.join(str(h) for h in args.layers)
    print('=' * 60)
    print('ML Fire Detection — Accumulated Observation Features')
    print(f'Loss: {args.loss}')
    print(f'Architecture: 12 -> {arch_str} -> 1')
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
    print('\n--- Step 3: Train/test split ---')
    print('  Ground truth (flight 03, no fire): 80% train, 20% test')
    train_flights = ['24-801-04', '24-801-05']
    test_flights = ['24-801-06']
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

    # Step 5: Normalize
    print('\n--- Step 5: Feature normalization (StandardScaler) ---')
    all_X = np.concatenate([ff['X'] for ff in flight_features.values()])
    all_X = np.where(np.isfinite(all_X), all_X, 0.0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(all_X)

    X_train_bal_clean = np.where(np.isfinite(X_train_bal), X_train_bal, 0.0).astype(np.float32)
    X_test_clean = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)
    X_train_norm = scaler.transform(X_train_bal_clean).astype(np.float32)
    X_test_norm = scaler.transform(X_test_clean).astype(np.float32)

    for i, name in enumerate(FEATURE_NAMES):
        print(f'  {name:12s}: mean={scaler.mean_[i]:10.3f}, '
              f'std={scaler.scale_[i]:10.3f}')

    # Step 6: Train
    loss_label = 'Soft (FN+FP)/P' if args.loss == 'error-rate' else 'Weighted BCE'
    print(f'\n--- Step 6: Training ({loss_label}, 100 epochs) ---')
    model, loss_history = train_model(
        X_train_norm, y_train_bal, w_train_bal,
        n_epochs=100, lr=1e-3, loss_fn=args.loss,
        hidden_layers=args.layers)

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

    # Step 8: Save model
    print('\n--- Step 8: Saving model ---')
    os.makedirs('checkpoint', exist_ok=True)
    model_path = f'checkpoint/fire_detector_{args.loss}.pt'
    torch.save({
        'model_state': model.state_dict(),
        'mean': scaler.mean_,
        'std': scaler.scale_,
        'scaler': scaler,
        'n_features': 12,
        'hidden_layers': model.hidden_layers,
        'threshold': 0.5,
        'feature_names': FEATURE_NAMES,
        'loss_fn': args.loss,
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

    # Summary
    print('\n' + '=' * 60)
    print('Summary:')
    print(f'  Features:  {len(FEATURE_NAMES)} aggregate features')
    arch = ' \u2192 '.join(['12'] + [str(h) for h in model.hidden_layers] + ['1'])
    print(f'  Model:     {arch} ({sum(p.numel() for p in model.parameters()):,} params)')
    print(f'  Checkpoint: {model_path}')
    print(f'  Dataset:    {dataset_path}')
    print(f'\n  To compare detectors per flight:')
    print(f'    python compare_fire_detectors.py')
    print(f'\n  To use in realtime_fire.py:')
    print(f'    python realtime_fire.py --detector ml --model {model_path}')
    print('Done.')


if __name__ == '__main__':
    main()

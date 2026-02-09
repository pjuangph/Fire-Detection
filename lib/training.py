"""Training pipeline: data loading, splitting, oversampling, and model training."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from tqdm import trange

from lib import group_files_by_flight, compute_grid_extent, build_pixel_table
from lib.features import build_location_features
from lib.losses import SoftErrorRateLoss, compute_pixel_weights
from lib.inference import FireMLP, FEATURE_NAMES
from lib.evaluation import get_device

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]
TrainResult = dict[str, Any]


# ── Data Pipeline ─────────────────────────────────────────────


def load_all_data(
    flights: dict[str, dict[str, Any]],
) -> FlightFeatures:
    """Build pixel tables for all flights and compute location features."""
    flight_features: FlightFeatures = {}
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
    importance_gt: float = 10.0,
    importance_fire: float = 5.0,
    importance_other: float = 1.0,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Split ground truth flight between train/test, then add burn flights.

    Flight 03 (pre-burn, no fire) is split 80/20 between train/test.
    Ground truth labels are forced to 0 (no real fire on pre-burn flight).
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

    # Compute pixel-wise weights with configurable importance
    w_train, _ = compute_pixel_weights(
        y_train, flight_src_train, ground_truth_flight,
        importance_gt=importance_gt,
        importance_fire=importance_fire,
        importance_other=importance_other)
    w_test, _ = compute_pixel_weights(
        y_test, flight_src_test, ground_truth_flight,
        importance_gt=importance_gt,
        importance_fire=importance_fire,
        importance_other=importance_other)

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
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 4096,
    loss_fn: str = 'bce',
    hidden_layers: list[int] | None = None,
    P_total: float | None = None,
    quiet: bool = False,
    resume_from: str | None = None,
) -> TrainResult:
    """Train FireMLP with the selected loss function.

    Args:
        quiet: If True, suppress per-epoch progress bar.
        resume_from: Checkpoint path to resume training from.

    Returns:
        Dict with keys: model, loss_history, epochs_completed,
        optimizer_state, scheduler_state.
    """
    device = get_device()
    if not quiet:
        print(f'  Device: {device}')

    model = FireMLP(hidden_layers=hidden_layers).to(device)
    use_error_rate = loss_fn == 'error-rate'
    if use_error_rate:
        if P_total is None:
            P_total = float(y_train.sum())
        criterion = SoftErrorRateLoss(P_total).to(device)
        loss_label = f'Soft (FN+FP)/P, P={P_total:.0f}'
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_label = 'Pixel Weighted BCE'
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt.get('epochs_completed', 0)
        if not quiet:
            print(f'  Resuming from epoch {start_epoch}/{n_epochs}')

    if start_epoch >= n_epochs:
        model = model.cpu()
        return {
            'model': model,
            'loss_history': np.zeros((0, 1)),
            'epochs_completed': n_epochs,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    w_t = torch.tensor(w_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    remaining = n_epochs - start_epoch
    loss_history = np.zeros((remaining, 1))
    pbar = trange(remaining, desc=loss_label, disable=quiet)
    for i in pbar:
        epoch = start_epoch + i
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

        scheduler.step()
        avg_loss = total_loss / len(X_t)
        loss_history[i] = avg_loss
        cur_lr = scheduler.get_last_lr()[0]
        if not quiet:
            pbar.set_postfix({'loss': f'{avg_loss:.4e}', 'lr': f'{cur_lr:.1e}',
                              'ep': f'{epoch + 1}/{n_epochs}'})

    model = model.cpu()
    return {
        'model': model,
        'loss_history': loss_history,
        'epochs_completed': n_epochs,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }

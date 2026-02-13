"""Shared training data pipeline: loading, splitting, oversampling, and grid search helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from lib import group_files_by_flight, compute_grid_extent, build_pixel_table
from lib.features import build_location_features
from lib.losses import compute_pixel_weights

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]


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


# ── Grid Search Helpers ──────────────────────────────────────


def load_existing_results(results_path: str) -> list[dict[str, Any]]:
    """Load previously completed results from JSON (for restart)."""
    if not os.path.isfile(results_path):
        return []
    with open(results_path) as f:
        data = json.load(f)
    return data.get('results', [])


def save_incremental(results: list[dict[str, Any]], cfg: dict[str, Any],
                     config_path: str, results_path: str) -> None:
    """Save results to JSON after each run (crash-safe)."""
    os.makedirs(os.path.dirname(results_path) or '.', exist_ok=True)
    metric = cfg.get('metric', 'error_rate')
    output = {
        'config': config_path,
        'timestamp': datetime.now().isoformat(),
        'metric': metric,
        'results': results,
    }
    if results:
        best = min(results, key=lambda r: r.get(metric, float('inf')))
        output['best_run_id'] = best['run_id']
        output[f'best_{metric}'] = best[metric]
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)


def build_representative_context(
    X: NDArrayFloat, y: NDArrayFloat, batch_size: int, seed: int,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Build a balanced (stratified) context subset for inference checkpoint."""
    rng = np.random.default_rng(seed)
    fire_idx = np.where(y == 1)[0]
    nofire_idx = np.where(y == 0)[0]

    half = batch_size // 2
    n_fire = min(half, len(fire_idx))
    n_nofire = min(batch_size - n_fire, len(nofire_idx))

    fire_sel = rng.choice(fire_idx, n_fire, replace=len(fire_idx) < n_fire)
    nofire_sel = rng.choice(nofire_idx, n_nofire, replace=len(nofire_idx) < n_nofire)

    idx = np.concatenate([fire_sel, nofire_sel])
    rng.shuffle(idx)
    return X[idx].copy(), y[idx].copy()


def safe_split(*args: Any, stratify: Any = None, **kwargs: Any) -> Any:
    """train_test_split that falls back to non-stratified when a class has < 2 members."""
    try:
        return train_test_split(*args, stratify=stratify, **kwargs)
    except ValueError:
        return train_test_split(*args, stratify=None, **kwargs)


def make_splitter(test_size: float = 0.2, seed: int = 0) -> Any:
    """Return a partial-applied safe_split for use with TabPFN data utils."""
    return partial(safe_split, test_size=test_size, random_state=seed)

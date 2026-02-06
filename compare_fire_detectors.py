"""compare_fire_detectors.py - Compare ML vs threshold fire detector per flight.

Auto-discovers trained models from checkpoint/fire_detector_*.pt, rebuilds
per-location features from HDF data, and prints a per-flight comparison
table for each model.

Usage:
    python compare_fire_detectors.py                                        # all models
    python compare_fire_detectors.py --model checkpoint/fire_detector_bce.pt  # one model
    python compare_fire_detectors.py --threshold 0.3                         # custom threshold
"""

from __future__ import annotations

import argparse
import glob as globmod
from typing import Any

import numpy as np
import numpy.typing as npt

from lib import group_files_by_flight, compute_grid_extent, build_pixel_table
from lib.inference import FireMLP, FEATURE_NAMES, load_model, predict
from lib.features import build_location_features

# StandardScaler type used in function signatures
from sklearn.preprocessing import StandardScaler

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]


# ── Data Loading ─────────────────────────────────────────────


def load_flight_features() -> FlightFeatures:
    """Load HDF data and compute per-location aggregate features.

    Returns:
        FlightFeatures: Dict mapping flight_num to feature dict containing
            'X', 'y', 'lats', 'lons', 'day_night', 'comment'.
    """
    flights = group_files_by_flight()
    flight_features: FlightFeatures = {}

    for fnum, info in sorted(flights.items()):
        files = info['files']
        day_night = info['day_night']
        print(f'  Flight {fnum} ({len(files)} files, {day_night})...')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        pixel_df = build_pixel_table(
            files, lat_min, lat_max, lon_min, lon_max,
            day_night=day_night, flight_num=fnum)

        X, y, lats, lons = build_location_features(pixel_df)
        n_locs = len(y)
        n_fire = int(y.sum())
        print(f'    {n_locs:,} locations, {n_fire:,} with fire')

        flight_features[fnum] = {
            'X': X, 'y': y, 'lats': lats, 'lons': lons,
            'day_night': day_night,
            'comment': info['comment'],
        }

    return flight_features


# ── Comparison ───────────────────────────────────────────────


GROUND_TRUTH_FLIGHT = '24-801-03'


def compare_detectors(
    flight_features: FlightFeatures,
    model: FireMLP,
    scaler: StandardScaler,
    threshold: float = 0.5,
    model_label: str = 'ML',
) -> None:
    """Compare ML detector vs detect_fire_simple for every flight.

    For each flight, prints a row showing:
    - Total grid-cell locations
    - Ground truth fire count
    - Threshold fire detections (raw detect_fire_simple labels)
    - ML fire detections (from trained model)
    - Confusion matrix (TP, FP, FN, TN) of ML vs ground truth

    Flight 03 (pre-burn) labels are forced to 0 because there is no real
    fire — any threshold detections on that flight are false positives.

    Args:
        flight_features: Dict from load_flight_features().
        model: Trained model (on CPU).
        scaler: Fitted scaler for feature normalization.
        threshold: ML classification threshold. Default 0.5.
        model_label: Short label for the ML model column (e.g. 'BCE', 'ER').
    """
    print('\n' + '=' * 80)
    print(f'Per-Flight Detector Comparison: Threshold vs {model_label} (threshold={threshold})')
    print('=' * 80)

    header = (f'  {"Flight":<14s} {"Comment":<22s} {"Locations":>9s}  '
              f'{"GT":>6s}  {"Thresh":>6s}  {model_label:>6s}  '
              f'{"TP":>6s}  {"FP":>6s}  {"FN":>6s}  {"TN":>8s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    model.eval()
    totals: dict[str, int] = {
        'locs': 0, 'gt': 0, 'thresh': 0, 'ml': 0,
        'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
    }

    for fnum in sorted(flight_features.keys()):
        d = flight_features[fnum]
        X = d['X']
        raw_y = d['y']
        comment = d.get('comment', '')[:22]

        # Ground truth: flight 03 is no-fire; burn flights use threshold labels
        if fnum == GROUND_TRUTH_FLIGHT:
            gt = np.zeros_like(raw_y)
        else:
            gt = raw_y
        gt_fire = gt > 0.5

        # Raw threshold detector output (before ground truth override)
        raw_thresh = raw_y > 0.5

        # ML predictions
        ml_fire, probs = predict(model, scaler, X, threshold=threshold)

        n_locs = len(gt)
        n_gt = int(gt_fire.sum())
        n_thresh = int(raw_thresh.sum())
        n_ml = int(ml_fire.sum())

        # Confusion matrix: ML vs ground truth
        TP = int((ml_fire & gt_fire).sum())
        FP = int((ml_fire & ~gt_fire).sum())
        FN = int((~ml_fire & gt_fire).sum())
        TN = int((~ml_fire & ~gt_fire).sum())

        print(f'  {fnum:<14s} {comment:<22s} {n_locs:>9,}  '
              f'{n_gt:>6,}  {n_thresh:>6,}  {n_ml:>6,}  '
              f'{TP:>6,}  {FP:>6,}  {FN:>6,}  {TN:>8,}')

        totals['locs'] += n_locs
        totals['gt'] += n_gt
        totals['thresh'] += n_thresh
        totals['ml'] += n_ml
        totals['TP'] += TP
        totals['FP'] += FP
        totals['FN'] += FN
        totals['TN'] += TN

    print('  ' + '-' * (len(header) - 2))
    print(f'  {"TOTAL":<14s} {"":<22s} {totals["locs"]:>9,}  '
          f'{totals["gt"]:>6,}  {totals["thresh"]:>6,}  {totals["ml"]:>6,}  '
          f'{totals["TP"]:>6,}  {totals["FP"]:>6,}  {totals["FN"]:>6,}  {totals["TN"]:>8,}')

    # Summary stats
    total_P = totals['TP'] + totals['FN']
    if total_P > 0:
        error_rate = (totals['FN'] + totals['FP']) / total_P
        print(f'\n  Error rate (FN+FP)/P: {error_rate:.4f}  '
              f'(P = {total_P:,} ground truth fire locations)')

    precision = totals['TP'] / max(totals['TP'] + totals['FP'], 1)
    recall = totals['TP'] / max(totals['TP'] + totals['FN'], 1)
    print(f'  Overall Precision: {precision:.4f}')
    print(f'  Overall Recall:    {recall:.4f}')


# ── Main ─────────────────────────────────────────────────────


def _discover_checkpoints() -> list[str]:
    """Find all fire_detector_*.pt checkpoints, sorted alphabetically."""
    return sorted(globmod.glob('checkpoint/fire_detector_*.pt'))


def main() -> None:
    """Load model(s), build features, and compare detectors."""
    parser = argparse.ArgumentParser(
        description='Compare ML vs threshold fire detector per flight.')
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='ML classification threshold (default: 0.5)')
    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to a specific model checkpoint (default: auto-discover all)')
    args = parser.parse_args()

    # Determine which checkpoints to compare
    if args.model:
        model_paths = [args.model]
    else:
        model_paths = _discover_checkpoints()
        if not model_paths:
            print('No checkpoints found in checkpoint/fire_detector_*.pt')
            print('Train a model first: python train_fire_prediction.py')
            return

    print(f'Found {len(model_paths)} model(s): {", ".join(model_paths)}')

    print('\nBuilding per-location features...')
    flight_features = load_flight_features()

    for model_path in model_paths:
        # Derive label from filename: fire_detector_bce.pt -> BCE
        stem = model_path.rsplit('/', 1)[-1]          # fire_detector_bce.pt
        loss_name = stem.replace('fire_detector_', '').replace('.pt', '')  # bce
        label = loss_name.upper().replace('ERROR-RATE', 'ER')

        print(f'\nLoading {model_path}...')
        model, scaler = load_model(model_path)
        print(f'  Model: {sum(p.numel() for p in model.parameters()):,} params')

        compare_detectors(
            flight_features, model, scaler,
            threshold=args.threshold, model_label=label)


if __name__ == '__main__':
    main()

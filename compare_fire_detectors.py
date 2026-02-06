"""compare_fire_detectors.py - Compare ML vs threshold fire detector per flight.

Loads the trained ML model from checkpoint/fire_detector.pt, rebuilds
per-location features from HDF data, and prints a per-flight comparison
table showing how many fires each detector predicts.

Usage:
    python compare_fire_detectors.py
    python compare_fire_detectors.py --threshold 0.3   # custom ML threshold
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt

from lib import group_files_by_flight, compute_grid_extent, build_pixel_table
from lib.inference import FireMLP, FEATURE_NAMES, load_model, predict
from tune_fire_prediction import build_location_features

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


def compare_detectors(
    flight_features: FlightFeatures,
    model: FireMLP,
    scaler: StandardScaler,
    threshold: float = 0.5,
) -> None:
    """Compare ML detector vs detect_fire_simple for every flight.

    For each flight, prints a row showing:
    - Total grid-cell locations
    - Threshold fire detections (from detect_fire_simple labels)
    - ML fire detections (from trained model)
    - Agreement breakdown (both, ML-only, threshold-only)
    - Confusion matrix (TP, FP, FN, TN)

    The threshold detector labels (y) come from detect_fire_simple() which
    uses T4 > 325K (day) / 310K (night) and dT > 10K.

    Args:
        flight_features (FlightFeatures): Dict from load_flight_features().
        model (FireMLP): Trained model (on CPU).
        scaler (StandardScaler): Fitted scaler for feature normalization.
        threshold (float): ML classification threshold. Default 0.5.
    """
    print('\n' + '=' * 80)
    print(f'Per-Flight Detector Comparison: Threshold vs ML (threshold={threshold})')
    print('=' * 80)

    header = (f'  {"Flight":<14s} {"Comment":<22s} {"Locations":>9s}  '
              f'{"Thresh":>6s}  {"ML":>6s}  '
              f'{"Both":>6s}  {"ML only":>7s}  {"Thr only":>8s}  '
              f'{"TP":>5s}  {"FP":>5s}  {"FN":>5s}  {"TN":>6s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    model.eval()
    totals: dict[str, int] = {
        'locs': 0, 'thresh': 0, 'ml': 0, 'both': 0,
        'ml_only': 0, 'thresh_only': 0,
        'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0,
    }

    for fnum in sorted(flight_features.keys()):
        d = flight_features[fnum]
        X, y = d['X'], d['y']
        comment = d.get('comment', '')[:22]

        # ML predictions via inference module
        ml_fire, probs = predict(model, scaler, X, threshold=threshold)
        thresh_fire = y > 0.5

        n_locs = len(y)
        n_thresh = int(thresh_fire.sum())
        n_ml = int(ml_fire.sum())
        both = int((ml_fire & thresh_fire).sum())
        ml_only = int((ml_fire & ~thresh_fire).sum())
        thresh_only = int((~ml_fire & thresh_fire).sum())

        # Confusion matrix (threshold labels = ground truth)
        TP = both
        FP = ml_only
        FN = thresh_only
        TN = int((~ml_fire & ~thresh_fire).sum())

        print(f'  {fnum:<14s} {comment:<22s} {n_locs:>9,}  '
              f'{n_thresh:>6,}  {n_ml:>6,}  '
              f'{both:>6,}  {ml_only:>7,}  {thresh_only:>8,}  '
              f'{TP:>5,}  {FP:>5,}  {FN:>5,}  {TN:>6,}')

        totals['locs'] += n_locs
        totals['thresh'] += n_thresh
        totals['ml'] += n_ml
        totals['both'] += both
        totals['ml_only'] += ml_only
        totals['thresh_only'] += thresh_only
        totals['TP'] += TP
        totals['FP'] += FP
        totals['FN'] += FN
        totals['TN'] += TN

    print('  ' + '-' * (len(header) - 2))
    print(f'  {"TOTAL":<14s} {"":<22s} {totals["locs"]:>9,}  '
          f'{totals["thresh"]:>6,}  {totals["ml"]:>6,}  '
          f'{totals["both"]:>6,}  {totals["ml_only"]:>7,}  {totals["thresh_only"]:>8,}  '
          f'{totals["TP"]:>5,}  {totals["FP"]:>5,}  {totals["FN"]:>5,}  {totals["TN"]:>6,}')

    # Summary stats
    total_P = totals['TP'] + totals['FN']
    if total_P > 0:
        error_rate = (totals['FN'] + totals['FP']) / total_P
        print(f'\n  Error rate (FN+FP)/P: {error_rate:.4f}  '
              f'(P = {total_P:,} threshold fire locations)')

    precision = totals['TP'] / max(totals['TP'] + totals['FP'], 1)
    recall = totals['TP'] / max(totals['TP'] + totals['FN'], 1)
    print(f'  Overall Precision: {precision:.4f}')
    print(f'  Overall Recall:    {recall:.4f}')


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    """Load model, build features, and compare detectors."""
    parser = argparse.ArgumentParser(
        description='Compare ML vs threshold fire detector per flight.')
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='ML classification threshold (default: 0.5)')
    parser.add_argument(
        '--model', type=str, default='checkpoint/fire_detector.pt',
        help='Path to model checkpoint')
    args = parser.parse_args()

    print('Loading model...')
    model, scaler = load_model(args.model)
    print(f'  Model: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'  Features: {FEATURE_NAMES}')

    print('\nBuilding per-location features...')
    flight_features = load_flight_features()

    compare_detectors(flight_features, model, scaler, threshold=args.threshold)


if __name__ == '__main__':
    main()

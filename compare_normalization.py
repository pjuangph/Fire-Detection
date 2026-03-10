"""compare_normalization.py - Compare hybrid vs StandardScaler-only normalization.

Takes the best model's hyperparameters and trains the same architecture twice:
  1. Hybrid normalization (thermal / T_ignition + StandardScaler on non-thermal)
  2. StandardScaler-only normalization (fit on all 12 features of GT flight)

Prints a side-by-side comparison of test metrics.

Usage:
    python compare_normalization.py
    python compare_normalization.py --config configs/best_model_mlp.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

from lib import group_files_by_flight
from lib.constants import THERMAL_FEATURE_INDICES, NON_THERMAL_FEATURE_INDICES
from lib.evaluation import evaluate
from lib.training import (
    load_all_data, extract_train_test, oversample_minority,
    compute_error_rate,
    FlightFeatures,
)
from train_mlp import train_model


def _normalize_hybrid(
    X: np.ndarray, scaler: StandardScaler, T_ign: float,
) -> np.ndarray:
    """Thermal / T_ign + StandardScaler on non-thermal."""
    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    X_norm = np.empty_like(X_clean)
    X_norm[:, THERMAL_FEATURE_INDICES] = X_clean[:, THERMAL_FEATURE_INDICES] / T_ign
    X_norm[:, NON_THERMAL_FEATURE_INDICES] = scaler.transform(
        X_clean[:, NON_THERMAL_FEATURE_INDICES]).astype(np.float32)
    return X_norm


def _normalize_standard(
    X: np.ndarray, scaler: StandardScaler,
) -> np.ndarray:
    """Full StandardScaler on all 12 features."""
    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return scaler.transform(X_clean).astype(np.float32)


def run_comparison(
    flight_features: FlightFeatures,
    layers: list[int],
    lr: float,
    loss_fn: str,
    epochs: int,
    T_ign: float,
    importance_weights: dict[str, float],
) -> None:
    """Train the same architecture with both normalization methods and compare."""
    wc = importance_weights

    # Extract train/test data (shared between both methods)
    (X_train, y_train, w_train, flight_src_train,
     X_test, y_test, w_test, flight_src_test) = extract_train_test(
        flight_features,
        train_flights=['24-801-04', '24-801-05'],
        test_flights=['24-801-06'],
        ground_truth_flight='24-801-03',
        gt_test_ratio=0.2,
        importance_gt=float(wc.get('gt', 10)),
        importance_fire=float(wc.get('fire', 5)),
        importance_other=float(wc.get('other', 1)),
    )

    # Oversample (shared)
    gt_mask_train = (flight_src_train == '24-801-03').astype(np.float32)
    gt_col = gt_mask_train.reshape(-1, 1)
    X_aug = np.concatenate([X_train, gt_col], axis=1)
    X_aug_ready, y_ready, w_ready = oversample_minority(X_aug, y_train, w_train)
    X_ready = X_aug_ready[:, :-1]
    gt_mask_ready = X_aug_ready[:, -1]
    P_total = float(y_ready.sum())

    gt_X = flight_features['24-801-03']['X']
    gt_X_clean = np.where(np.isfinite(gt_X), gt_X, 0.0).astype(np.float32)

    results = {}

    # ── Method 1: Hybrid normalization ──────────────────────────
    print('\n' + '=' * 60)
    print('  Method 1: HYBRID normalization (thermal / T_ignition)')
    print('=' * 60)

    scaler_hybrid = StandardScaler()
    scaler_hybrid.fit(gt_X_clean[:, NON_THERMAL_FEATURE_INDICES])

    X_norm = _normalize_hybrid(X_ready, scaler_hybrid, T_ign)
    X_test_norm = _normalize_hybrid(X_test, scaler_hybrid, T_ign)

    result_h = train_model(
        X_norm, y_ready, w_ready,
        n_epochs=epochs, lr=lr, batch_size=4096,
        loss_fn=loss_fn, hidden_layers=layers, P_total=P_total,
        gt_mask=gt_mask_ready,
        X_test=X_test_norm, y_test=y_test)

    test_metrics_h, _ = evaluate(result_h['model'], X_test_norm, y_test)
    results['hybrid'] = test_metrics_h

    # ── Method 2: StandardScaler-only normalization ────────────
    print('\n' + '=' * 60)
    print('  Method 2: STANDARD SCALER normalization (all 12 features)')
    print('=' * 60)

    scaler_full = StandardScaler()
    scaler_full.fit(gt_X_clean)  # fit on ALL 12 features

    X_norm_s = _normalize_standard(X_ready, scaler_full)
    X_test_norm_s = _normalize_standard(X_test, scaler_full)

    result_s = train_model(
        X_norm_s, y_ready, w_ready,
        n_epochs=epochs, lr=lr, batch_size=4096,
        loss_fn=loss_fn, hidden_layers=layers, P_total=P_total,
        gt_mask=gt_mask_ready,
        X_test=X_test_norm_s, y_test=y_test)

    test_metrics_s, _ = evaluate(result_s['model'], X_test_norm_s, y_test)
    results['standard'] = test_metrics_s

    # ── Comparison table ───────────────────────────────────────
    err_h = compute_error_rate(test_metrics_h)
    err_s = compute_error_rate(test_metrics_s)

    print('\n' + '=' * 70)
    print('  NORMALIZATION COMPARISON')
    print('=' * 70)
    print(f'  Architecture: 12 -> {" -> ".join(str(h) for h in layers)} -> 1')
    print(f'  Loss: {loss_fn}  |  LR: {lr}  |  Epochs: {epochs}')
    print(f'  T_ignition: {T_ign:.2f} K ({T_ign - 273.15:.1f} °C)')
    print()
    print(f'  {"Metric":<20s}  {"Hybrid":>10s}  {"StandardScaler":>14s}  {"Diff":>10s}')
    print(f'  {"-" * 58}')
    print(f'  {"TP":<20s}  {test_metrics_h["TP"]:>10,.0f}  {test_metrics_s["TP"]:>14,.0f}  '
          f'{test_metrics_h["TP"] - test_metrics_s["TP"]:>+10,.0f}')
    print(f'  {"FP":<20s}  {test_metrics_h["FP"]:>10,.0f}  {test_metrics_s["FP"]:>14,.0f}  '
          f'{test_metrics_h["FP"] - test_metrics_s["FP"]:>+10,.0f}')
    print(f'  {"FN":<20s}  {test_metrics_h["FN"]:>10,.0f}  {test_metrics_s["FN"]:>14,.0f}  '
          f'{test_metrics_h["FN"] - test_metrics_s["FN"]:>+10,.0f}')
    print(f'  {"TN":<20s}  {test_metrics_h["TN"]:>10,.0f}  {test_metrics_s["TN"]:>14,.0f}  '
          f'{test_metrics_h["TN"] - test_metrics_s["TN"]:>+10,.0f}')
    print(f'  {"Precision":<20s}  {test_metrics_h["precision"]:>10.4f}  '
          f'{test_metrics_s["precision"]:>14.4f}  '
          f'{test_metrics_h["precision"] - test_metrics_s["precision"]:>+10.4f}')
    print(f'  {"Recall":<20s}  {test_metrics_h["recall"]:>10.4f}  '
          f'{test_metrics_s["recall"]:>14.4f}  '
          f'{test_metrics_h["recall"] - test_metrics_s["recall"]:>+10.4f}')
    print(f'  {"Error Rate":<20s}  {err_h:>10.4f}  {err_s:>14.4f}  '
          f'{err_h - err_s:>+10.4f}')
    print(f'  {"FP + FN":<20s}  '
          f'{test_metrics_h["FP"] + test_metrics_h["FN"]:>10,.0f}  '
          f'{test_metrics_s["FP"] + test_metrics_s["FN"]:>14,.0f}  '
          f'{(test_metrics_h["FP"] + test_metrics_h["FN"]) - (test_metrics_s["FP"] + test_metrics_s["FN"]):>+10,.0f}')
    print()

    if err_h < err_s:
        print('  --> Hybrid normalization has LOWER error rate (better)')
    elif err_s < err_h:
        print('  --> StandardScaler normalization has LOWER error rate (better)')
    else:
        print('  --> Both methods have the same error rate')
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare hybrid vs StandardScaler normalization')
    parser.add_argument(
        '--config', type=str, default='configs/best_model_mlp.yaml',
        help='Best model YAML config (for architecture/hyperparameters)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count (default: from config or 120)')
    args = parser.parse_args()

    print('=' * 60)
    print('  Normalization Comparison: Hybrid vs StandardScaler')
    print('=' * 60)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    training = cfg.get('training', {})
    layers = training.get('layers', [64, 32])
    lr = training.get('learning_rate', 0.001)
    loss_fn = training.get('loss', 'error-rate')
    epochs = args.epochs or training.get('epochs', 120)
    T_ign = cfg.get('T_ignition', 300.0) + 273.15
    wc = training.get('importance_weights', {'gt': 10, 'fire': 5, 'other': 1})

    print(f'  Config: {args.config}')
    print(f'  Layers: {layers}, LR: {lr}, Loss: {loss_fn}, Epochs: {epochs}')

    data_dir = cfg.get('data_dir', 'ignite_fire_data')
    print(f'\n--- Loading flight data ---')
    flights = group_files_by_flight(data_dir)
    flight_features = load_all_data(flights)

    run_comparison(
        flight_features,
        layers=layers, lr=lr, loss_fn=loss_fn, epochs=epochs,
        T_ign=T_ign, importance_weights=wc,
    )
    print('Done.')


if __name__ == '__main__':
    main()

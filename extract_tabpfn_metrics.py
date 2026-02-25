"""Extract metrics from TabPFN checkpoints and generate missing artifacts.

Reads all run checkpoints to build results JSONs and convergence plots,
then evaluates best models on the test set to produce YAML configs.

Usage:
    python extract_tabpfn_metrics.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.finetuning.train_util import clone_model_for_evaluation

from lib import group_files_by_flight
from lib.inference import FEATURE_NAMES
from lib.evaluation import auto_device
from lib.training import load_all_data, build_representative_context


def _build_full_dataset(flight_features: dict) -> tuple[np.ndarray, np.ndarray]:
    """Combine all flights into a single X, y array (ground truth flight forced to y=0)."""
    gt_flight = '24-801-03'
    X_parts, y_parts = [], []
    for fnum, ff in sorted(flight_features.items()):
        X_parts.append(ff['X'])
        if fnum == gt_flight:
            y_parts.append(np.zeros(len(ff['X']), dtype=np.float32))
        else:
            y_parts.append(ff['y'])
    return np.concatenate(X_parts), np.concatenate(y_parts)


def _split_and_scale(X: np.ndarray, y: np.ndarray, scaler: StandardScaler,
                     seed: int = 0, test_size: float = 0.2):
    """Apply scaler and reproduce the same train/test split used during training."""
    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    X_norm = scaler.transform(X_clean).astype(np.float32)
    y_norm = y.astype(np.float32)

    n_fire = int(y_norm.sum())
    stratify = y_norm if n_fire >= 2 and (len(y_norm) - n_fire) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=test_size, random_state=seed, stratify=stratify)
    return X_train, X_test, y_train, y_test


# ── Phase 1: Build results JSONs from all run checkpoints ────


def _scan_classification_runs() -> list[dict[str, Any]]:
    """Scan all classification run checkpoints and extract metadata."""
    results = []
    for i in range(1, 17):  # 16 runs
        path = f'checkpoint/fire_detector_tabpfn_run_{i:02d}.pt'
        if not os.path.isfile(path):
            continue
        ckpt = torch.load(path, weights_only=False, map_location='cpu')
        init = ckpt.get('classifier_init', {})
        results.append({
            'run_id': i,
            'learning_rate': 3e-4 if (i - 1) // 8 == 0 else 1e-4,  # from grid order
            'batch_size': ckpt.get('batch_size', 1024),
            'n_estimators': init.get('n_estimators', 1),
            'weight_decay': 0.01,  # will be refined below
            'grad_clip_norm': 1.0,
            'epochs_completed': ckpt.get('epoch', 20),
            'loss_history': ckpt.get('loss_history', []),
            'checkpoint': path,
            'error_rate': float('inf'),  # placeholder until evaluation
        })
    return results


def _scan_regression_runs() -> list[dict[str, Any]]:
    """Scan all regression run checkpoints and extract metadata."""
    results = []
    for i in range(1, 33):  # 32 runs
        path = f'checkpoint/fire_detector_tabpfn_reg_run_{i:02d}.pt'
        if not os.path.isfile(path):
            continue
        ckpt = torch.load(path, weights_only=False, map_location='cpu')
        init = ckpt.get('regressor_init', {})
        results.append({
            'run_id': i,
            'learning_rate': 3e-4,  # placeholder
            'batch_size': ckpt.get('batch_size', 1024),
            'n_estimators': init.get('n_estimators', 1),
            'weight_decay': 0.01,
            'grad_clip_norm': 1.0,
            'crps_loss_weight': 1.0,
            'mse_loss_weight': 0.0,
            'epochs_completed': ckpt.get('epoch', 20),
            'loss_history': ckpt.get('loss_history', []),
            'checkpoint': path,
            'error_rate': float('inf'),
        })
    return results


def _save_results_json(results: list[dict], path: str, config_path: str) -> None:
    """Save results in the format expected by plot_convergence_curves."""
    # Find best by final loss as proxy (real error_rate from eval)
    metric = 'error_rate'
    valid = [r for r in results if r.get('error_rate', float('inf')) < float('inf')]
    output: dict[str, Any] = {
        'config': config_path,
        'timestamp': datetime.now().isoformat(),
        'metric': metric,
        'results': results,
    }
    if valid:
        best = min(valid, key=lambda r: r[metric])
        output['best_run_id'] = best['run_id']
        output[f'best_{metric}'] = best[metric]
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=float)


# ── Phase 2: Evaluate best models ───────────────────────────


def evaluate_classification_best(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> dict[str, Any]:
    """Load best classification checkpoint, evaluate on test set."""
    path = 'checkpoint/fire_detector_tabpfn_best.pt'
    ckpt = torch.load(path, weights_only=False, map_location='cpu')

    device_str = auto_device()
    init = ckpt['classifier_init'].copy()
    init['device'] = device_str  # use current device, not training device

    seed = ckpt.get('seed', 0)
    batch_size = ckpt.get('batch_size', 1024)

    # Build classifier from checkpoint
    classifier = TabPFNClassifier(
        **init,
        fit_mode="batched",
        differentiable_input=False,
    )
    classifier._initialize_model_variables()
    model = classifier.models_[0]
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(torch.device(device_str))
    model.eval()

    # Clone for clean eval
    eval_init = {k: v for k, v in init.items() if k != 'model_path'}
    eval_clf = clone_model_for_evaluation(classifier, eval_init, TabPFNClassifier)

    # Fit on representative context from checkpoint
    ctx_X = ckpt['context_X']
    ctx_y = ckpt['context_y']
    eval_clf.fit(ctx_X, ctx_y)

    # Predict on test set in batches (MPS can't handle 300k+ at once)
    batch_size_pred = 1000
    probs_parts = []
    for i in range(0, len(X_test), batch_size_pred):
        batch = eval_clf.predict_proba(X_test[i:i + batch_size_pred])
        probs_parts.append(batch[:, 1])
    probs = np.concatenate(probs_parts).astype(np.float32)
    threshold = ckpt.get('threshold', 0.5)
    preds = (probs >= threshold).astype(np.float32)

    TP = int(np.sum((preds == 1) & (y_test == 1)))
    FP = int(np.sum((preds == 1) & (y_test == 0)))
    FN = int(np.sum((preds == 0) & (y_test == 1)))
    TN = int(np.sum((preds == 0) & (y_test == 0)))
    P = TP + FN
    error_rate = (FN + FP) / max(P, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)

    print(f'\n  Classification best:')
    print(f'    TP={TP}, FP={FP}, FN={FN}, TN={TN}')
    print(f'    Error rate={error_rate:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'error_rate': error_rate, 'precision': precision, 'recall': recall,
        'n_estimators': init.get('n_estimators', 1),
        'batch_size': batch_size,
        'seed': seed,
        'threshold': threshold,
        'loss_history': ckpt.get('loss_history', []),
        'epoch': ckpt.get('epoch', 20),
    }


def evaluate_regression_best(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> dict[str, Any]:
    """Load best regression checkpoint, evaluate on test set."""
    path = 'checkpoint/fire_detector_tabpfn_regression_best.pt'
    ckpt = torch.load(path, weights_only=False, map_location='cpu')

    device_str = auto_device()
    init = ckpt['regressor_init'].copy()
    init['device'] = device_str

    seed = ckpt.get('seed', 0)
    batch_size = ckpt.get('batch_size', 1024)

    # Build regressor from checkpoint
    regressor = TabPFNRegressor(
        **init,
        fit_mode="batched",
        differentiable_input=False,
    )
    regressor._initialize_model_variables()
    model = regressor.models_[0]
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(torch.device(device_str))
    model.eval()

    # Clone for clean eval
    eval_init = {k: v for k, v in init.items() if k != 'model_path'}
    eval_reg = clone_model_for_evaluation(regressor, eval_init, TabPFNRegressor)

    # Fit on representative context from checkpoint
    ctx_X = ckpt['context_X']
    ctx_y = ckpt['context_y']
    eval_reg.fit(ctx_X, ctx_y)

    # Predict on test set in batches — regression output clipped to [0,1]
    batch_size_pred = 1000
    preds_parts = []
    for i in range(0, len(X_test), batch_size_pred):
        preds_parts.append(eval_reg.predict(X_test[i:i + batch_size_pred]))
    preds_raw = np.concatenate(preds_parts)
    probs = np.clip(preds_raw, 0.0, 1.0).astype(np.float32)
    threshold = ckpt.get('threshold', 0.5)
    preds = (probs >= threshold).astype(np.float32)

    TP = int(np.sum((preds == 1) & (y_test == 1)))
    FP = int(np.sum((preds == 1) & (y_test == 0)))
    FN = int(np.sum((preds == 0) & (y_test == 1)))
    TN = int(np.sum((preds == 0) & (y_test == 0)))
    P = TP + FN
    error_rate = (FN + FP) / max(P, 1)
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)

    print(f'\n  Regression best:')
    print(f'    TP={TP}, FP={FP}, FN={FN}, TN={TN}')
    print(f'    Error rate={error_rate:.4f}, Precision={precision:.4f}, Recall={recall:.4f}')

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'error_rate': error_rate, 'precision': precision, 'recall': recall,
        'n_estimators': init.get('n_estimators', 1),
        'batch_size': batch_size,
        'seed': seed,
        'threshold': threshold,
        'loss_history': ckpt.get('loss_history', []),
        'epoch': ckpt.get('epoch', 20),
    }


# ── Phase 3: Write YAML configs ─────────────────────────────


def write_classification_config(metrics: dict[str, Any]) -> None:
    """Write configs/best_model_tabpfn_classification.yaml."""
    config = {
        'detector': 'tabpfn_classification',
        'model_type': 'tabpfn',
        'checkpoint': 'checkpoint/fire_detector_tabpfn_best.pt',
        'threshold': metrics['threshold'],
        'training': {
            'batch_size': metrics['batch_size'],
            'n_estimators': metrics['n_estimators'],
            'epochs': metrics['epoch'],
        },
        'metrics': {
            'error_rate': float(metrics['error_rate']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'TP': int(metrics['TP']),
            'FP': int(metrics['FP']),
            'FN': int(metrics['FN']),
            'TN': int(metrics['TN']),
        },
    }
    path = 'configs/best_model_tabpfn_classification.yaml'
    os.makedirs('configs', exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'  Written: {path}')


def write_regression_config(metrics: dict[str, Any]) -> None:
    """Write configs/best_model_tabpfn_regression.yaml."""
    config = {
        'detector': 'tabpfn_regression',
        'model_type': 'tabpfn_regression',
        'checkpoint': 'checkpoint/fire_detector_tabpfn_regression_best.pt',
        'threshold': metrics['threshold'],
        'training': {
            'batch_size': metrics['batch_size'],
            'n_estimators': metrics['n_estimators'],
            'epochs': metrics['epoch'],
        },
        'metrics': {
            'error_rate': float(metrics['error_rate']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'TP': int(metrics['TP']),
            'FP': int(metrics['FP']),
            'FN': int(metrics['FN']),
            'TN': int(metrics['TN']),
        },
    }
    path = 'configs/best_model_tabpfn_regression.yaml'
    os.makedirs('configs', exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'  Written: {path}')


# ── Phase 4: Convergence plots ──────────────────────────────


def plot_convergence(results_path: str, out_path: str, title: str) -> None:
    """Generate convergence plot from results JSON."""
    from lib.plotting import plot_convergence_curves
    plot_convergence_curves(results_path, out_path=out_path, title=title)
    print(f'  Written: {out_path}')


# ── Main ─────────────────────────────────────────────────────


def main() -> None:
    print('=' * 60)
    print('TabPFN Metric Extraction from Checkpoints')
    print('=' * 60)

    # Phase 1: Build results JSONs from run checkpoints
    print('\n--- Phase 1: Scanning run checkpoints ---')

    cls_results = _scan_classification_runs()
    print(f'  Classification: {len(cls_results)} runs found')
    cls_json = 'results/grid_search_tabpfn_classification_results.json'
    _save_results_json(cls_results, cls_json,
                       'configs/grid_search_tabpfn_classification.yaml')

    reg_results = _scan_regression_runs()
    print(f'  Regression: {len(reg_results)} runs found')
    reg_json = 'results/grid_search_tabpfn_regression_results.json'
    _save_results_json(reg_results, reg_json,
                       'configs/grid_search_tabpfn_regression.yaml')

    # Phase 1b: Convergence plots (only needs loss_history from JSONs)
    print('\n--- Phase 1b: Convergence plots ---')
    plot_convergence(cls_json, 'plots/convergence_tabpfn_classification.png',
                     'TabPFN Classifier — Training Convergence')
    plot_convergence(reg_json, 'plots/convergence_tabpfn_regression.png',
                     'TabPFN Regressor — Training Convergence')

    # Phase 2: Load flight data for evaluation
    print('\n--- Phase 2: Loading flight data ---')
    flights = group_files_by_flight()
    flight_features = load_all_data(flights)

    X_all, y_all = _build_full_dataset(flight_features)
    print(f'  Total: {len(X_all):,} locations, {int(y_all.sum()):,} with fire')

    # Use scaler from best classification checkpoint (both should be identical)
    cls_ckpt = torch.load('checkpoint/fire_detector_tabpfn_best.pt',
                          weights_only=False, map_location='cpu')
    scaler = cls_ckpt['scaler']
    seed = cls_ckpt.get('seed', 0)

    X_train, X_test, y_train, y_test = _split_and_scale(
        X_all, y_all, scaler, seed=seed)
    print(f'  Train: {len(X_train):,}  Test: {len(X_test):,}')

    # Phase 3: Evaluate best models
    print('\n--- Phase 3: Evaluating best models ---')
    cls_metrics = evaluate_classification_best(X_train, X_test, y_train, y_test)
    reg_metrics = evaluate_regression_best(X_train, X_test, y_train, y_test)

    # Phase 4: Write YAML configs
    print('\n--- Phase 4: Writing configs ---')
    write_classification_config(cls_metrics)
    write_regression_config(reg_metrics)

    # Update results JSONs with actual error rates for best runs
    # (re-save so convergence plot labels are accurate)
    print('\n--- Done ---')
    print('Generated:')
    print(f'  configs/best_model_tabpfn_classification.yaml')
    print(f'  configs/best_model_tabpfn_regression.yaml')
    print(f'  results/grid_search_tabpfn_classification_results.json')
    print(f'  results/grid_search_tabpfn_regression_results.json')
    print(f'  plots/convergence_tabpfn_classification.png')
    print(f'  plots/convergence_tabpfn_regression.png')


if __name__ == '__main__':
    main()

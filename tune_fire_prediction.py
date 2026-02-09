"""tune_fire_prediction.py - YAML-driven grid search for fire detection MLP.

Iterates over combinations of loss function, architecture, learning rate,
and importance weights. Results are saved to JSON. The best model is saved
to checkpoint/fire_detector_best.pt for automatic discovery by inference.

Usage:
    python tune_fire_prediction.py --config configs/grid_search.yaml
    python tune_fire_prediction.py --loss bce --layers 64 64 64 32   # single run
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
from datetime import datetime
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from lib import group_files_by_flight
from lib.inference import FEATURE_NAMES
from lib.evaluation import evaluate
from lib.training import (
    load_all_data, extract_train_test, oversample_minority, train_model,
    FlightFeatures,
)


# ── Grid Search ──────────────────────────────────────────────


def build_combos(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all hyperparameter combinations from YAML config."""
    space = cfg['search_space']
    losses = space['loss']
    layers_list = space['layers']
    lrs = space['learning_rate']
    weight_configs = space.get('importance_weights', [{'gt': 10, 'fire': 5, 'other': 1}])

    combos: list[dict[str, Any]] = []
    for loss, layers, lr in itertools.product(losses, layers_list, lrs):
        if loss == 'error-rate':
            # Error-rate uses uniform weights; only 1 weight combo needed
            combos.append({
                'loss': loss,
                'layers': list(layers),
                'learning_rate': lr,
                'importance_weights': {'gt': 10, 'fire': 5, 'other': 1},
            })
        else:
            for wc in weight_configs:
                combos.append({
                    'loss': loss,
                    'layers': list(layers),
                    'learning_rate': lr,
                    'importance_weights': dict(wc),
                })
    return combos


def _combo_key(combo: dict[str, Any]) -> str:
    """Build a unique string key for a hyperparameter combo."""
    wc = combo['importance_weights']
    return (f"{combo['loss']}|{combo['layers']}|{combo['learning_rate']}"
            f"|{wc['gt']}/{wc['fire']}/{wc['other']}")


def _load_existing_results(results_path: str) -> list[dict[str, Any]]:
    """Load previously completed results from JSON (for restart)."""
    if not os.path.isfile(results_path):
        return []
    with open(results_path) as f:
        data = json.load(f)
    return data.get('results', [])


def _save_incremental(results: list[dict[str, Any]], cfg: dict[str, Any],
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


def run_grid_search(cfg: dict[str, Any], flight_features: FlightFeatures,
                    config_path: str = '',
                    results_path: str = 'results/grid_search_results.json',
                    ) -> list[dict[str, Any]]:
    """Run all grid search combinations and return results.

    Supports restart: loads existing results from results_path, skips
    fully-trained combos, and resumes under-trained ones.
    """
    combos = build_combos(cfg)
    epochs = cfg.get('epochs', 100)
    batch_size = cfg.get('batch_size', 4096)

    # Load previously completed results (restart support)
    existing_results = _load_existing_results(results_path)
    # Index existing results by combo key for fast lookup
    existing_by_key: dict[str, dict[str, Any]] = {}
    for r in existing_results:
        key = _combo_key({
            'loss': r['loss'], 'layers': r['layers'],
            'learning_rate': r['learning_rate'],
            'importance_weights': r['importance_weights'],
        })
        existing_by_key[key] = r

    n_skip = sum(1 for r in existing_by_key.values()
                 if r.get('epochs_completed', 0) >= epochs)
    n_resume = sum(1 for r in existing_by_key.values()
                   if 0 < r.get('epochs_completed', 0) < epochs)
    if existing_results:
        print(f'\n  Restart: {n_skip} fully trained, '
              f'{n_resume} under-trained (will resume), '
              f'{len(combos) - len(existing_by_key)} new')

    # Fit scaler once on all data
    all_X = np.concatenate([ff['X'] for ff in flight_features.values()])
    all_X = np.where(np.isfinite(all_X), all_X, 0.0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(all_X)

    os.makedirs('checkpoint', exist_ok=True)
    # Start with existing results (fully trained ones kept, under-trained replaced)
    results: list[dict[str, Any]] = [
        r for r in existing_results
        if r.get('epochs_completed', 0) >= epochs
    ]
    n_combos = len(combos)

    for i, combo in enumerate(combos):
        run_id = i + 1
        loss = combo['loss']
        layers = combo['layers']
        lr = combo['learning_rate']
        wc = combo['importance_weights']
        use_error_rate = loss == 'error-rate'
        key = _combo_key(combo)

        # Skip fully-trained runs
        existing = existing_by_key.get(key)
        if existing and existing.get('epochs_completed', 0) >= epochs:
            print(f'\n  Run {run_id}/{n_combos}: SKIPPED (already completed)')
            continue

        arch_str = ' -> '.join(['12'] + [str(h) for h in layers] + ['1'])
        wt_str = f"gt={wc['gt']}, fire={wc['fire']}, other={wc['other']}"
        print(f'\n{"=" * 60}')
        print(f'Run {run_id}/{n_combos}: {loss}, {arch_str}, lr={lr}, weights=[{wt_str}]')
        print('=' * 60)

        # Resume from under-trained checkpoint if available
        resume_ckpt = None
        if existing and existing.get('epochs_completed', 0) < epochs:
            resume_ckpt = existing.get('checkpoint')
            print(f'  Resuming under-trained run '
                  f'({existing["epochs_completed"]}/{epochs} epochs)')

        # Build train/test with this combo's importance weights
        X_train, y_train, w_train, X_test, y_test, w_test = extract_train_test(
            flight_features,
            train_flights=['24-801-04', '24-801-05'],
            test_flights=['24-801-06'],
            ground_truth_flight='24-801-03',
            gt_test_ratio=0.2,
            importance_gt=float(wc['gt']),
            importance_fire=float(wc['fire']),
            importance_other=float(wc['other']),
        )

        # Capture P_total before oversampling
        P_original = float(y_train.sum())

        # Oversample to 50/50 balance
        X_ready, y_ready, w_ready = oversample_minority(X_train, y_train, w_train)

        # For error-rate: uniform weights
        if use_error_rate:
            w_ready = np.ones_like(w_ready)

        # Normalize
        X_clean = np.where(np.isfinite(X_ready), X_ready, 0.0).astype(np.float32)
        X_norm = scaler.transform(X_clean).astype(np.float32)

        X_test_clean = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)
        X_test_norm = scaler.transform(X_test_clean).astype(np.float32)

        # Train
        train_result = train_model(
            X_norm, y_ready, w_ready,
            n_epochs=epochs, lr=lr, batch_size=batch_size,
            loss_fn=loss, hidden_layers=layers, P_total=P_original,
            quiet=False, resume_from=resume_ckpt)

        model = train_result['model']
        loss_history = train_result['loss_history']
        epochs_completed = train_result['epochs_completed']

        # Evaluate on original (non-oversampled) train data
        X_train_eval = np.where(np.isfinite(X_train), X_train, 0.0).astype(np.float32)
        X_train_norm = scaler.transform(X_train_eval).astype(np.float32)
        train_metrics, _ = evaluate(model, X_train_norm, y_train)

        # Evaluate on test
        test_metrics, _ = evaluate(model, X_test_norm, y_test)

        # Error rate: (FN+FP)/P
        test_P = test_metrics['TP'] + test_metrics['FN']
        error_rate = (test_metrics['FN'] + test_metrics['FP']) / max(test_P, 1)

        # Save checkpoint with full training state
        ckpt_path = f'checkpoint/fire_detector_run_{run_id:02d}.pt'
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': train_result['optimizer_state'],
            'scheduler_state': train_result['scheduler_state'],
            'epochs_completed': epochs_completed,
            'mean': scaler.mean_,
            'std': scaler.scale_,
            'scaler': scaler,
            'n_features': 12,
            'hidden_layers': model.hidden_layers,
            'threshold': 0.5,
            'feature_names': FEATURE_NAMES,
            'loss_fn': loss,
        }, ckpt_path)

        final_loss = float(loss_history[-1, 0]) if len(loss_history) > 0 else 0.0

        result = {
            'run_id': run_id,
            'loss': loss,
            'layers': layers,
            'learning_rate': lr,
            'importance_weights': wc,
            'epochs_completed': epochs_completed,
            'train': {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                      for k, v in train_metrics.items()},
            'test': {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                     for k, v in test_metrics.items()},
            'error_rate': float(error_rate),
            'final_loss': final_loss,
            'checkpoint': ckpt_path,
        }
        results.append(result)

        # Save after every run (crash-safe)
        _save_incremental(results, cfg, config_path, results_path)

        # Print result summary
        print(f'  Train: TP={train_metrics["TP"]:,} FP={train_metrics["FP"]:,} '
              f'FN={train_metrics["FN"]:,} TN={train_metrics["TN"]:,}')
        print(f'  Test:  TP={test_metrics["TP"]:,} FP={test_metrics["FP"]:,} '
              f'FN={test_metrics["FN"]:,} TN={test_metrics["TN"]:,}')
        print(f'  Precision={test_metrics["precision"]:.4f} '
              f'Recall={test_metrics["recall"]:.4f} '
              f'Error rate={error_rate:.4f}')

    return results


def print_results_table(results: list[dict[str, Any]], metric: str) -> None:
    """Print sorted summary table of grid search results."""
    sorted_results = sorted(results, key=lambda r: r.get(metric, float('inf')))

    print(f'\n{"=" * 100}')
    print(f'Grid Search Results (sorted by {metric}, best first)')
    print('=' * 100)

    header = (f'  {"Run":>3s}  {"Loss":<10s}  {"Layers":<18s}  {"LR":>8s}  '
              f'{"Weights":<16s}  {"TP":>6s}  {"FP":>5s}  {"FN":>5s}  '
              f'{"Prec":>6s}  {"Rec":>6s}  {"ErrRate":>7s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for r in sorted_results:
        wc = r['importance_weights']
        wt_str = f"{wc['gt']}/{wc['fire']}/{wc['other']}"
        layers_str = 'x'.join(str(h) for h in r['layers'])
        t = r['test']
        print(f'  {r["run_id"]:>3d}  {r["loss"]:<10s}  {layers_str:<18s}  '
              f'{r["learning_rate"]:>8.4f}  {wt_str:<16s}  '
              f'{t["TP"]:>6.0f}  {t["FP"]:>5.0f}  {t["FN"]:>5.0f}  '
              f'{t["precision"]:>6.4f}  {t["recall"]:>6.4f}  '
              f'{r["error_rate"]:>7.4f}')


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    """Run grid search or single training run."""
    parser = argparse.ArgumentParser(
        description='Tune MLP fire detector via YAML grid search')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML grid search config (default: configs/grid_search.yaml)')

    # Single-run mode (e.g. --loss bce --layers 64 64 64 32)
    parser.add_argument('--loss', choices=['bce', 'error-rate'], default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    # Determine mode: config file vs single run
    single_mode = args.loss is not None or args.layers is not None or args.lr is not None
    if single_mode and args.config:
        parser.error('Cannot use --config with --loss/--layers/--lr')

    # Load flight data (shared across all runs)
    print('=' * 60)
    print('ML Fire Detection — Hyperparameter Tuning')
    print('=' * 60)
    print('\n--- Loading flight data ---')
    flights = group_files_by_flight()
    flight_features = load_all_data(flights)

    if single_mode:
        loss = args.loss or 'bce'
        layers = args.layers or [64, 32]
        lr = args.lr or 1e-3

        cfg = {
            'epochs': args.epochs,
            'batch_size': 4096,
            'metric': 'error_rate',
            'search_space': {
                'loss': [loss],
                'layers': [layers],
                'learning_rate': [lr],
                'importance_weights': [{'gt': 10, 'fire': 5, 'other': 1}],
            },
        }
        config_path = 'single-run'
    else:
        config_path = args.config or 'configs/grid_search.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    combos = build_combos(cfg)
    metric = cfg.get('metric', 'error_rate')
    print(f'\nConfig: {config_path}')
    print(f'Runs: {len(combos)} | Epochs: {cfg.get("epochs", 100)} | Metric: {metric}')

    # Run grid search (with restart support)
    results_path = 'results/grid_search_results.json'
    results = run_grid_search(cfg, flight_features,
                              config_path=config_path,
                              results_path=results_path)

    # Print summary table
    print_results_table(results, metric)

    # Find best run and copy to best.pt
    best = min(results, key=lambda r: r.get(metric, float('inf')))
    best_ckpt = best['checkpoint']
    best_path = 'checkpoint/fire_detector_best.pt'
    shutil.copy2(best_ckpt, best_path)

    # Summary
    wc = best['importance_weights']
    arch = ' -> '.join(['12'] + [str(h) for h in best['layers']] + ['1'])
    print(f'\n{"=" * 60}')
    print(f'Best run: #{best["run_id"]}')
    print(f'  Loss:         {best["loss"]}')
    print(f'  Architecture: {arch}')
    print(f'  LR:           {best["learning_rate"]}')
    print(f'  Weights:      gt={wc["gt"]}, fire={wc["fire"]}, other={wc["other"]}')
    print(f'  Error rate:   {best["error_rate"]:.4f}')
    print(f'  Precision:    {best["test"]["precision"]:.4f}')
    print(f'  Recall:       {best["test"]["recall"]:.4f}')
    print(f'\n  Best model:  {best_path}')
    print(f'  Results:     {results_path}')
    print(f'\n  To compare per flight:')
    print(f'    python compare_fire_detectors.py --model {best_path}')
    print(f'\n  To use in realtime:')
    print(f'    python realtime_fire.py --detector ml')
    print('Done.')


if __name__ == '__main__':
    main()

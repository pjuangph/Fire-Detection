"""train_mlp.py - YAML-driven grid search for fire detection MLP.

Iterates over combinations of loss function, architecture, learning rate,
and importance weights. Results are saved to JSON. The best model is saved
to checkpoint/fire_detector_best.pt for automatic discovery by inference.

Usage:
    python train_mlp.py --config configs/grid_search_mlp.yaml
    python train_mlp.py --loss bce --layers 64 64 64 32   # single run
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
import numpy.typing as npt
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import trange

from models.firemlp import FireMLP
from lib import group_files_by_flight
from lib.inference import FEATURE_NAMES
from lib.evaluation import get_device, evaluate
from lib.losses import SoftErrorRateLoss
from lib.training import (
    load_all_data, extract_train_test, oversample_minority,
    FlightFeatures,
)

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
TrainResult = dict[str, Any]


# ── MLP Training ──────────────────────────────────────────────


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
    save_path: str | None = None,
    save_every: int = 25,
) -> TrainResult:
    """Train FireMLP with the selected loss function.

    Args:
        quiet: If True, suppress per-epoch progress bar.
        resume_from: Checkpoint path to resume training from.
        save_path: If set, save checkpoint to this path periodically.
        save_every: Save checkpoint every N epochs (default 25).

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

        # Periodic checkpoint save for crash recovery
        epochs_done = start_epoch + i + 1
        if save_path and save_every > 0 and epochs_done % save_every == 0:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epochs_completed': epochs_done,
                'hidden_layers': hidden_layers or [64, 32],
            }, save_path)

    model = model.cpu()
    return {
        'model': model,
        'loss_history': loss_history,
        'epochs_completed': n_epochs,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }


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
                    results_path: str = 'results/grid_search_mlp_results.json',
                    ) -> list[dict[str, Any]]:
    """Run all grid search combinations and return results.

    Supports restart: loads existing results from results_path, skips
    fully-trained combos, and resumes under-trained ones.
    """
    combos = build_combos(cfg)
    epochs = cfg.get('epochs', 100)
    batch_size = cfg.get('batch_size', 4096)
    save_every = cfg.get('save_every', 25)

    # Load previously completed results (restart support)
    existing_results = _load_existing_results(results_path)
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

        ckpt_path = f'checkpoint/fire_detector_mlp_run_{run_id:02d}.pt'

        # Skip fully-trained runs
        existing = existing_by_key.get(key)
        if existing and existing.get('epochs_completed', 0) >= epochs:
            print(f'\n  Run {run_id}/{n_combos}: SKIPPED (already completed)')
            continue

        # Check for resume checkpoint on disk
        resume_ckpt = None
        resume_epochs = 0
        if os.path.isfile(ckpt_path):
            try:
                ckpt_info = torch.load(ckpt_path, weights_only=False,
                                       map_location='cpu')
                resume_epochs = ckpt_info.get('epochs_completed', 0)
            except Exception:
                resume_epochs = 0
            if resume_epochs >= epochs:
                pass
            if resume_epochs > 0:
                resume_ckpt = ckpt_path
        elif existing and existing.get('checkpoint'):
            ckpt_file = existing['checkpoint']
            if os.path.isfile(ckpt_file):
                resume_ckpt = ckpt_file
                resume_epochs = existing.get('epochs_completed', 0)

        arch_str = ' -> '.join(['12'] + [str(h) for h in layers] + ['1'])
        wt_str = f"gt={wc['gt']}, fire={wc['fire']}, other={wc['other']}"
        print(f'\n{"=" * 60}')
        print(f'Run {run_id}/{n_combos}: {loss}, {arch_str}, lr={lr}, weights=[{wt_str}]')
        print('=' * 60)
        if resume_ckpt:
            print(f'  Resuming from checkpoint ({resume_epochs}/{epochs} epochs)')

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

        P_original = float(y_train.sum())
        X_ready, y_ready, w_ready = oversample_minority(X_train, y_train, w_train)

        if use_error_rate:
            w_ready = np.ones_like(w_ready)

        X_clean = np.where(np.isfinite(X_ready), X_ready, 0.0).astype(np.float32)
        X_norm = scaler.transform(X_clean).astype(np.float32)

        X_test_clean = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)
        X_test_norm = scaler.transform(X_test_clean).astype(np.float32)

        train_result = train_model(
            X_norm, y_ready, w_ready,
            n_epochs=epochs, lr=lr, batch_size=batch_size,
            loss_fn=loss, hidden_layers=layers, P_total=P_original,
            quiet=False, resume_from=resume_ckpt,
            save_path=ckpt_path, save_every=save_every)

        model = train_result['model']
        loss_history = train_result['loss_history']
        epochs_completed = train_result['epochs_completed']

        X_train_eval = np.where(np.isfinite(X_train), X_train, 0.0).astype(np.float32)
        X_train_norm = scaler.transform(X_train_eval).astype(np.float32)
        train_metrics, _ = evaluate(model, X_train_norm, y_train)
        test_metrics, _ = evaluate(model, X_test_norm, y_test)

        test_P = test_metrics['TP'] + test_metrics['FN']
        error_rate = (test_metrics['FN'] + test_metrics['FP']) / max(test_P, 1)

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

        _save_incremental(results, cfg, config_path, results_path)

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


def write_best_model_config(best: dict[str, Any], checkpoint_path: str,
                            config_path: str = 'configs/best_model.yaml',
                            ) -> str:
    """Write YAML config for the best model (consumed by realtime scripts)."""
    wc = best['importance_weights']
    t = best['test']
    config = {
        'detector': 'ml',
        'model_type': 'firemlp',
        'checkpoint': checkpoint_path,
        'threshold': 0.5,
        'training': {
            'loss': best['loss'],
            'layers': best['layers'],
            'learning_rate': best['learning_rate'],
            'epochs': best['epochs_completed'],
            'importance_weights': {
                'gt': wc['gt'],
                'fire': wc['fire'],
                'other': wc['other'],
            },
        },
        'metrics': {
            'error_rate': float(best['error_rate']),
            'precision': float(t['precision']),
            'recall': float(t['recall']),
            'TP': int(t['TP']),
            'FP': int(t['FP']),
            'FN': int(t['FN']),
            'TN': int(t['TN']),
        },
    }
    os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    """Run MLP grid search or single training run."""
    parser = argparse.ArgumentParser(
        description='Train MLP fire detector via YAML grid search')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML grid search config (default: configs/grid_search_mlp.yaml)')
    parser.add_argument('--loss', choices=['bce', 'error-rate'], default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    single_mode = args.loss is not None or args.layers is not None or args.lr is not None
    if single_mode and args.config:
        parser.error('Cannot use --config with --loss/--layers/--lr')

    print('=' * 60)
    print('ML Fire Detection \u2014 MLP Hyperparameter Tuning')
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
        config_path = args.config or 'configs/grid_search_mlp.yaml'
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    combos = build_combos(cfg)
    metric = cfg.get('metric', 'error_rate')
    print(f'\nConfig: {config_path}')
    print(f'Runs: {len(combos)} | Epochs: {cfg.get("epochs", 100)} | Metric: {metric}')

    results_path = 'results/grid_search_mlp_results.json'
    results = run_grid_search(cfg, flight_features,
                              config_path=config_path,
                              results_path=results_path)

    print_results_table(results, metric)

    best = min(results, key=lambda r: r.get(metric, float('inf')))
    best_ckpt = best['checkpoint']
    best_path = 'checkpoint/fire_detector_best.pt'
    shutil.copy2(best_ckpt, best_path)

    model_config_path = 'configs/best_model.yaml'
    write_best_model_config(best, best_path, model_config_path)

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
    print(f'  Config:      {model_config_path}')
    print(f'  Results:     {results_path}')
    print(f'\n  To compare per flight:')
    print(f'    python compare_fire_detectors.py --model {best_path}')
    print(f'\n  To use in realtime:')
    print(f'    python realtime_mlp.py --config {model_config_path}')
    print('Done.')


if __name__ == '__main__':
    main()

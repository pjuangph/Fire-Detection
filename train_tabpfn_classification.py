"""train_tabpfn_classification.py - From-scratch TabPFN classification training.

Initializes TabPFN with random weights (model_path="random:<seed>") so no
pretrained checkpoint download is needed. Trains the transformer with gradient
descent using cross-entropy loss on binary fire/no-fire classification.

Iterates over combinations of learning_rate, batch_size, n_estimators,
weight_decay, and grad_clip_norm. Results are saved to JSON. The best model
checkpoint includes a representative training context for inference.

Usage:
    python train_tabpfn_classification.py --config configs/grid_search_tabpfn_classification.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import shutil
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.train_util import clone_model_for_evaluation
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks, meta_dataset_collator,
)

from lib import group_files_by_flight
from lib.inference import FEATURE_NAMES
from lib.evaluation import evaluate
from lib.training import (
    load_all_data,
    FlightFeatures,
)

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
TrainResult = dict[str, Any]


# ── Device Detection ──────────────────────────────────────────


def _auto_device() -> str:
    """Return best available device string for TabPFN."""
    if platform.system() == "Darwin":
        mps_ok = (hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available())
        return "mps" if mps_ok else "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ── TabPFN Training ──────────────────────────────────────────


def _build_representative_context(
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


def train_tabpfn_model(
    X: NDArrayFloat,
    y: NDArrayFloat,
    *,
    n_epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    n_estimators: int = 1,
    grad_clip_norm: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
    quiet: bool = False,
    resume_from: str | None = None,
    save_path: str | None = None,
    save_every: int = 5,
) -> TrainResult:
    """Train TabPFN classifier from random weights with gradient descent.

    Splits X/y internally into train/test, then trains the transformer
    on shuffled chunks of batch_size rows per epoch.

    Args:
        X: Scaled feature array (N, 12) — full dataset (train+test).
        y: Binary labels (N,).
        n_epochs: Number of training epochs.
        lr: Learning rate for AdamW.
        weight_decay: AdamW weight decay.
        batch_size: Rows per chunk fed to TabPFN attention (power of 2).
        n_estimators: Number of internal ensemble models.
        grad_clip_norm: Max gradient norm for clipping.
        test_size: Fraction held out for internal test split.
        seed: Random seed.
        quiet: Suppress progress output.
        resume_from: Checkpoint path for resume.
        save_path: Periodic checkpoint save path.
        save_every: Save every N epochs.

    Returns:
        Dict with model, classifier, loss_history, epochs_completed,
        optimizer_state, classifier_init, X_train, y_train, X_test, y_test.
    """
    device_str = _auto_device()
    if not quiet:
        print(f'  Device: {device_str}')

    # Internal train/test split (stratified)
    n_fire = int(y.sum())
    stratify = y if n_fire >= 2 and (len(y) - n_fire) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify,
    )
    if not quiet:
        print(f'  Split: {len(X_train)} train, {len(X_test)} test '
              f'({int(y_train.sum())} fire / {int((y_train == 0).sum())} no-fire)')

    classifier_init = {
        "device": device_str,
        "random_state": seed,
        "n_estimators": n_estimators,
        "ignore_pretraining_limits": True,
        "inference_precision": torch.float32,
        "model_path": f"random:{seed}",
    }
    classifier = TabPFNClassifier(
        **classifier_init,
        fit_mode="batched",
        differentiable_input=False,
    )

    classifier._initialize_model_variables()
    if len(classifier.models_) != 1:
        raise ValueError(
            f"Expected 1 internal model, got {len(classifier.models_)}.")
    model = classifier.models_[0]
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Resume from checkpoint
    start_epoch = 0
    if resume_from and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, weights_only=False,
                          map_location=device_str)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        if not quiet:
            print(f'  Resuming from epoch {start_epoch}/{n_epochs}')

    early_result = {
        'model': model,
        'classifier': classifier,
        'loss_history': np.zeros((0, 1)),
        'epochs_completed': n_epochs,
        'optimizer_state': optimizer.state_dict(),
        'classifier_init': classifier_init,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }
    if start_epoch >= n_epochs:
        return early_result

    rng = np.random.default_rng(seed)

    def _safe_split(*args, stratify=None, **kwargs):
        """Split that falls back to non-stratified when a class has < 2 members."""
        try:
            return train_test_split(*args, stratify=stratify, **kwargs)
        except ValueError:
            return train_test_split(*args, stratify=None, **kwargs)

    splitter = partial(_safe_split, test_size=0.2, random_state=seed)

    remaining = n_epochs - start_epoch
    loss_history = np.zeros((remaining, 1))

    for ep_i in range(remaining):
        epoch = start_epoch + ep_i + 1
        model.train()

        # Shuffle and partition training data into batch_size chunks
        perm = rng.permutation(len(X_train))
        X_contexts: list[NDArrayFloat] = []
        y_contexts: list[NDArrayFloat] = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            if len(idx) < 2:
                continue
            X_contexts.append(X_train[idx])
            y_contexts.append(y_train[idx])

        # Preprocess into TabPFN dataset format
        training_datasets = get_preprocessed_dataset_chunks(
            calling_instance=classifier,
            X_raw=X_contexts,
            y_raw=y_contexts,
            split_fn=splitter,
            max_data_size=None,
            model_type="classifier",
            equal_split_size=True,
            seed=seed + epoch,
        )

        dataloader = DataLoader(
            training_datasets, batch_size=1,
            collate_fn=meta_dataset_collator,
        )

        epoch_losses: list[float] = []
        desc = f'Epoch {epoch}/{n_epochs}'
        progress = tqdm(dataloader, desc=desc, disable=quiet, unit='batch')

        for batch in progress:
            optimizer.zero_grad(set_to_none=True)

            # Fit from preprocessed context
            classifier.fit_from_preprocessed(
                batch.X_context, batch.y_context,
                batch.cat_indices, batch.configs,
            )

            # Forward with raw logits: shape (Q, B=1, E, L)
            logits_QBEL = classifier.forward(
                batch.X_query, return_raw_logits=True,
            )
            Q, B, E, L = logits_QBEL.shape

            # Reshape to (B*E, L, Q) for cross-entropy
            logits_BLQ = logits_QBEL.permute(1, 2, 3, 0).reshape(B * E, L, Q)

            # Expand targets to (B*E, Q)
            targets_BQ = batch.y_query.to(logits_BLQ.device)
            targets_BQ = targets_BQ.expand(B * E, -1).long()

            loss = F.cross_entropy(logits_BLQ, targets_BQ)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))
            if not quiet:
                progress.set_postfix(loss=f'{epoch_losses[-1]:.4e}')

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_history[ep_i] = avg_loss

        if not quiet:
            print(f'  Epoch {epoch} avg loss: {avg_loss:.4e}')

        # Periodic checkpoint
        if save_path and save_every > 0 and epoch % save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'classifier_init': classifier_init,
            }, save_path)

    return {
        'model': model,
        'classifier': classifier,
        'loss_history': loss_history,
        'epochs_completed': n_epochs,
        'optimizer_state': optimizer.state_dict(),
        'classifier_init': classifier_init,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }


def _evaluate_tabpfn(
    classifier: TabPFNClassifier,
    classifier_init: dict[str, Any],
    X_train: NDArrayFloat, y_train: NDArrayFloat,
    X_eval: NDArrayFloat, y_eval: NDArrayFloat,
    batch_size: int, seed: int,
) -> tuple[dict[str, Any], NDArrayFloat]:
    """Evaluate a trained TabPFN classifier on a dataset.

    Clones the model for clean eval, fits on a representative context,
    then uses the existing evaluate() function.
    """
    eval_init = {k: v for k, v in classifier_init.items() if k != 'model_path'}
    eval_clf = clone_model_for_evaluation(
        classifier, eval_init, TabPFNClassifier)

    # Fit on a representative context sample
    rng = np.random.default_rng(seed)
    n = min(batch_size, len(X_train))
    replace = len(X_train) < batch_size
    idx = rng.choice(len(X_train), size=n, replace=replace)
    eval_clf.fit(X_train[idx], y_train[idx])

    return evaluate(eval_clf, X_eval, y_eval)


# ── Grid Search ──────────────────────────────────────────────


def build_combos(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all hyperparameter combinations from YAML config."""
    space = cfg['search_space']
    combos: list[dict[str, Any]] = []
    for lr, bs, n_est, wd, gc in itertools.product(
        space['learning_rate'],
        space['batch_size'],
        space['n_estimators'],
        space['weight_decay'],
        space['grad_clip_norm'],
    ):
        combos.append({
            'learning_rate': lr,
            'batch_size': bs,
            'n_estimators': n_est,
            'weight_decay': wd,
            'grad_clip_norm': gc,
        })
    return combos


def _combo_key(combo: dict[str, Any]) -> str:
    """Build a unique string key for a hyperparameter combo."""
    return (f"{combo['learning_rate']}|{combo['batch_size']}"
            f"|{combo['n_estimators']}|{combo['weight_decay']}"
            f"|{combo['grad_clip_norm']}")


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
                    results_path: str = 'results/grid_search_tabpfn_results.json',
                    ) -> list[dict[str, Any]]:
    """Run all TabPFN grid search combinations and return results.

    Supports restart: loads existing results, skips completed combos,
    resumes under-trained ones from checkpoint.
    """
    combos = build_combos(cfg)
    epochs = cfg.get('epochs', 10)
    save_every = cfg.get('save_every', 5)
    seed = cfg.get('seed', 0)

    # Load previously completed results
    existing_results = _load_existing_results(results_path)
    existing_by_key: dict[str, dict[str, Any]] = {}
    for r in existing_results:
        key = _combo_key({
            'learning_rate': r['learning_rate'],
            'batch_size': r['batch_size'],
            'n_estimators': r['n_estimators'],
            'weight_decay': r['weight_decay'],
            'grad_clip_norm': r['grad_clip_norm'],
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

    # Build full dataset — ground truth flight (pre-burn) forced to y=0
    gt_flight = '24-801-03'
    X_parts, y_parts = [], []
    for fnum, ff in flight_features.items():
        X_parts.append(ff['X'])
        if fnum == gt_flight:
            y_parts.append(np.zeros(len(ff['X']), dtype=np.float32))
        else:
            y_parts.append(ff['y'])
    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(y_parts)

    # Fit scaler once on all features, then normalize
    X_clean = np.where(np.isfinite(X_all), X_all, 0.0).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_clean)
    X_norm = scaler.transform(X_clean).astype(np.float32)
    y_norm = y_all.astype(np.float32)

    os.makedirs('checkpoint', exist_ok=True)
    results: list[dict[str, Any]] = [
        r for r in existing_results
        if r.get('epochs_completed', 0) >= epochs
    ]
    n_combos = len(combos)

    for i, combo in enumerate(combos):
        run_id = i + 1
        lr_val = combo['learning_rate']
        bs = combo['batch_size']
        n_est = combo['n_estimators']
        wd = combo['weight_decay']
        gc = combo['grad_clip_norm']
        key = _combo_key(combo)

        ckpt_path = f'checkpoint/fire_detector_tabpfn_run_{run_id:02d}.pt'

        # Skip fully-trained runs
        existing = existing_by_key.get(key)
        if existing and existing.get('epochs_completed', 0) >= epochs:
            print(f'\n  Run {run_id}/{n_combos}: SKIPPED (already completed)')
            continue

        # Check for resume checkpoint on disk
        resume_ckpt = None
        if os.path.isfile(ckpt_path):
            try:
                ckpt_info = torch.load(ckpt_path, weights_only=False,
                                       map_location='cpu')
                resume_epochs = ckpt_info.get('epoch', 0)
            except Exception:
                resume_epochs = 0
            if 0 < resume_epochs < epochs:
                resume_ckpt = ckpt_path

        print(f'\n{"=" * 60}')
        print(f'Run {run_id}/{n_combos}: lr={lr_val}, bs={bs}, '
              f'n_est={n_est}, wd={wd}, gc={gc}')
        print('=' * 60)
        if resume_ckpt:
            print(f'  Resuming from checkpoint')

        train_result = train_tabpfn_model(
            X_norm, y_norm,
            n_epochs=epochs, lr=lr_val, weight_decay=wd,
            batch_size=bs, n_estimators=n_est,
            grad_clip_norm=gc, seed=seed,
            quiet=False, resume_from=resume_ckpt,
            save_path=ckpt_path, save_every=save_every,
        )

        classifier = train_result['classifier']
        classifier_init = train_result['classifier_init']
        epochs_completed = train_result['epochs_completed']
        X_tr = train_result['X_train']
        y_tr = train_result['y_train']
        X_te = train_result['X_test']
        y_te = train_result['y_test']

        # Evaluate on internal train/test splits
        train_metrics, _ = _evaluate_tabpfn(
            classifier, classifier_init,
            X_tr, y_tr, X_tr, y_tr,
            batch_size=bs, seed=seed)
        test_metrics, _ = _evaluate_tabpfn(
            classifier, classifier_init,
            X_tr, y_tr, X_te, y_te,
            batch_size=bs, seed=seed)

        test_P = test_metrics['TP'] + test_metrics['FN']
        error_rate = (test_metrics['FN'] + test_metrics['FP']) / max(test_P, 1)

        # Build representative context for inference checkpoint
        ctx_X, ctx_y = _build_representative_context(
            X_tr, y_tr, batch_size=bs, seed=seed)

        # Save full checkpoint
        torch.save({
            'model_state_dict': train_result['model'].state_dict(),
            'optimizer_state_dict': train_result['optimizer_state'],
            'epoch': epochs_completed,
            'classifier_init': classifier_init,
            'context_X': ctx_X,
            'context_y': ctx_y,
            'scaler': scaler,
            'threshold': 0.5,
            'feature_names': FEATURE_NAMES,
            'n_estimators': n_est,
            'batch_size': bs,
            'seed': seed,
        }, ckpt_path)

        loss_history = train_result['loss_history']
        final_loss = float(loss_history[-1, 0]) if len(loss_history) > 0 else 0.0

        result = {
            'run_id': run_id,
            'learning_rate': lr_val,
            'batch_size': bs,
            'n_estimators': n_est,
            'weight_decay': wd,
            'grad_clip_norm': gc,
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

    print(f'\n{"=" * 110}')
    print(f'TabPFN Grid Search Results (sorted by {metric}, best first)')
    print('=' * 110)

    header = (f'  {"Run":>3s}  {"LR":>8s}  {"BS":>4s}  {"Est":>3s}  '
              f'{"WD":>6s}  {"GC":>4s}  '
              f'{"TP":>6s}  {"FP":>5s}  {"FN":>5s}  '
              f'{"Prec":>6s}  {"Rec":>6s}  {"ErrRate":>7s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for r in sorted_results:
        t = r['test']
        print(f'  {r["run_id"]:>3d}  {r["learning_rate"]:>8.1e}  '
              f'{r["batch_size"]:>4d}  {r["n_estimators"]:>3d}  '
              f'{r["weight_decay"]:>6.3f}  {r["grad_clip_norm"]:>4.1f}  '
              f'{t["TP"]:>6.0f}  {t["FP"]:>5.0f}  {t["FN"]:>5.0f}  '
              f'{t["precision"]:>6.4f}  {t["recall"]:>6.4f}  '
              f'{r["error_rate"]:>7.4f}')


def write_best_model_config(best: dict[str, Any], checkpoint_path: str,
                            config_path: str = 'configs/best_model.yaml',
                            ) -> str:
    """Write YAML config for the best model (consumed by realtime scripts)."""
    t = best['test']
    config = {
        'detector': 'ml',
        'model_type': 'tabpfn',
        'checkpoint': checkpoint_path,
        'threshold': 0.5,
        'training': {
            'learning_rate': best['learning_rate'],
            'batch_size': best['batch_size'],
            'n_estimators': best['n_estimators'],
            'weight_decay': best['weight_decay'],
            'grad_clip_norm': best['grad_clip_norm'],
            'epochs': best['epochs_completed'],
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
    """Run TabPFN from-scratch grid search."""
    parser = argparse.ArgumentParser(
        description='Train TabPFN fire detector from scratch via grid search')
    parser.add_argument(
        '--config', type=str, default='configs/grid_search_tabpfn_classification.yaml',
        help='Path to YAML grid search config')
    args = parser.parse_args()

    print('=' * 60)
    print('ML Fire Detection \u2014 TabPFN Classification Training')
    print('=' * 60)
    print('\n--- Loading flight data ---')
    flights = group_files_by_flight()
    flight_features = load_all_data(flights)

    config_path = args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    combos = build_combos(cfg)
    metric = cfg.get('metric', 'error_rate')
    print(f'\nConfig: {config_path}')
    print(f'Runs: {len(combos)} | Epochs: {cfg.get("epochs", 10)} | Metric: {metric}')

    results_path = 'results/grid_search_tabpfn_classification_results.json'
    results = run_grid_search(cfg, flight_features,
                              config_path=config_path,
                              results_path=results_path)

    print_results_table(results, metric)

    best = min(results, key=lambda r: r.get(metric, float('inf')))
    best_ckpt = best['checkpoint']
    best_path = 'checkpoint/fire_detector_tabpfn_best.pt'
    shutil.copy2(best_ckpt, best_path)

    model_config_path = 'configs/best_model.yaml'
    write_best_model_config(best, best_path, model_config_path)

    print(f'\n{"=" * 60}')
    print(f'Best run: #{best["run_id"]}')
    print(f'  LR:              {best["learning_rate"]}')
    print(f'  Batch size:      {best["batch_size"]}')
    print(f'  n_estimators:    {best["n_estimators"]}')
    print(f'  Weight decay:    {best["weight_decay"]}')
    print(f'  Grad clip:       {best["grad_clip_norm"]}')
    print(f'  Error rate:      {best["error_rate"]:.4f}')
    print(f'  Precision:       {best["test"]["precision"]:.4f}')
    print(f'  Recall:          {best["test"]["recall"]:.4f}')
    print(f'\n  Best model:  {best_path}')
    print(f'  Config:      {model_config_path}')
    print(f'  Results:     {results_path}')
    print(f'\n  To use in realtime:')
    print(f'    python realtime_tabpfn.py --config {model_config_path}')
    print('Done.')


if __name__ == '__main__':
    main()

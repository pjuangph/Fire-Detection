"""train_tabpfn_regression.py - From-scratch TabPFN regression training.

Initializes TabPFN with random weights (model_path="random:<seed>") so no
pretrained checkpoint download is needed. Trains the transformer with gradient
descent using bar distribution regression loss (CRPS + optional MSE auxiliary)
on binary fire/no-fire targets treated as continuous regression.

Iterates over combinations of learning_rate, batch_size, n_estimators,
weight_decay, grad_clip_norm, and regression loss weights. Results are saved
to JSON. The best model checkpoint includes a representative training context
for inference.

Usage:
    python train_tabpfn_regression.py --config configs/grid_search_tabpfn_regression.yaml
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetuning.train_util import clone_model_for_evaluation
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss
from tabpfn.finetuning.data_util import (
    get_preprocessed_dataset_chunks, meta_dataset_collator,
)

from lib import group_files_by_flight
from lib.inference import FEATURE_NAMES
from lib.evaluation import auto_device
from lib.training import (
    load_all_data, load_existing_results, save_incremental,
    build_representative_context, make_splitter,
    FlightFeatures,
)

# Type aliases
NDArrayFloat = npt.NDArray[np.floating[Any]]
TrainResult = dict[str, Any]


# ── TabPFN Regression Training ───────────────────────────────


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
    crps_loss_weight: float = 1.0,
    mse_loss_weight: float = 1.0,
    test_size: float = 0.2,
    seed: int = 0,
    quiet: bool = False,
    resume_from: str | None = None,
    save_path: str | None = None,
    save_every: int = 5,
) -> TrainResult:
    """Train TabPFN regressor from random weights with gradient descent.

    Splits X/y internally into train/test, then trains the transformer
    on shuffled chunks of batch_size rows per epoch using bar distribution
    regression loss.

    Args:
        X: Scaled feature array (N, 12) — full dataset (train+test).
        y: Binary labels (N,) treated as continuous regression targets.
        n_epochs: Number of training epochs.
        lr: Learning rate for AdamW.
        weight_decay: AdamW weight decay.
        batch_size: Rows per chunk fed to TabPFN attention (power of 2).
        n_estimators: Number of internal ensemble models.
        grad_clip_norm: Max gradient norm for clipping.
        crps_loss_weight: Weight for CRPS (continuous ranked probability score) loss.
        mse_loss_weight: Weight for auxiliary MSE loss on mean prediction.
        test_size: Fraction held out for internal test split.
        seed: Random seed.
        quiet: Suppress progress output.
        resume_from: Checkpoint path for resume.
        save_path: Periodic checkpoint save path.
        save_every: Save every N epochs.

    Returns:
        Dict with model, regressor, loss_history, epochs_completed,
        optimizer_state, regressor_init, X_train, y_train, X_test, y_test.
    """
    device_str = auto_device()
    if not quiet:
        print(f'  Device: {device_str}')

    # Internal train/test split (no stratification for regression)
    n_fire = int(y.sum())
    stratify = y if n_fire >= 2 and (len(y) - n_fire) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify,
    )
    if not quiet:
        print(f'  Split: {len(X_train)} train, {len(X_test)} test '
              f'({int(y_train.sum())} fire / {int((y_train == 0).sum())} no-fire)')

    regressor_init = {
        "device": device_str,
        "random_state": seed,
        "n_estimators": n_estimators,
        "ignore_pretraining_limits": True,
        "inference_precision": torch.float32,
        "model_path": f"random:{seed}",
    }
    regressor = TabPFNRegressor(
        **regressor_init,
        fit_mode="batched",
        differentiable_input=False,
    )

    regressor._initialize_model_variables()
    if len(regressor.models_) != 1:
        raise ValueError(
            f"Expected 1 internal model, got {len(regressor.models_)}.")
    model = regressor.models_[0]
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
        'regressor': regressor,
        'loss_history': np.zeros((0, 1)),
        'epochs_completed': n_epochs,
        'optimizer_state': optimizer.state_dict(),
        'regressor_init': regressor_init,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }
    if start_epoch >= n_epochs:
        return early_result

    rng = np.random.default_rng(seed)
    splitter = make_splitter(test_size=0.2, seed=seed)

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

        # Preprocess into TabPFN dataset format (regression mode)
        training_datasets = get_preprocessed_dataset_chunks(
            calling_instance=regressor,
            X_raw=X_contexts,
            y_raw=y_contexts,
            split_fn=splitter,
            max_data_size=None,
            model_type="regressor",
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

            # Set bar distribution for this batch (regression-specific)
            regressor.raw_space_bardist_ = batch.raw_space_bardist
            regressor.bardist_ = batch.znorm_space_bardist

            # Fit from preprocessed context
            regressor.fit_from_preprocessed(
                batch.X_context, batch.y_context,
                batch.cat_indices, batch.configs,
            )

            # Forward: returns (averaged_logits, per_estim_logits, borders)
            _, per_estim_logits, _ = regressor.forward(batch.X_query)

            # per_estim_logits: list of [Q, B(=1), L] tensors
            logits_QBEL = torch.stack(per_estim_logits, dim=2)
            Q, B, E, L = logits_QBEL.shape

            # Reshape to (B*E, Q, L) for bar distribution loss
            logits_BQL = logits_QBEL.permute(1, 2, 0, 3).reshape(B * E, Q, L)

            # Expand targets to (B*E, Q)
            targets_BQ = batch.y_query.to(logits_BQL.device)
            targets_BQ = targets_BQ.repeat(B * E, 1)

            loss = _compute_regression_loss(
                logits_BQL=logits_BQL,
                targets_BQ=targets_BQ,
                bardist_loss_fn=batch.znorm_space_bardist,
                ce_loss_weight=0.0,
                crps_loss_weight=crps_loss_weight,
                mse_loss_weight=mse_loss_weight,
                mae_loss_weight=0.0,
            )
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
                'regressor_init': regressor_init,
            }, save_path)

    return {
        'model': model,
        'regressor': regressor,
        'loss_history': loss_history,
        'epochs_completed': n_epochs,
        'optimizer_state': optimizer.state_dict(),
        'regressor_init': regressor_init,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }


def _evaluate_tabpfn_regressor(
    regressor: TabPFNRegressor,
    regressor_init: dict[str, Any],
    X_train: NDArrayFloat, y_train: NDArrayFloat,
    X_eval: NDArrayFloat, y_eval: NDArrayFloat,
    batch_size: int, seed: int, threshold: float = 0.5,
) -> tuple[dict[str, Any], NDArrayFloat]:
    """Evaluate a trained TabPFN regressor on a dataset.

    Clones the model for clean eval, fits on a representative context,
    predicts continuous values, clips to [0,1], and thresholds for
    binary fire detection metrics.
    """
    eval_init = {k: v for k, v in regressor_init.items() if k != 'model_path'}
    eval_reg = clone_model_for_evaluation(
        regressor, eval_init, TabPFNRegressor)

    # Fit on a representative context sample
    rng = np.random.default_rng(seed)
    n = min(batch_size, len(X_train))
    replace = len(X_train) < batch_size
    idx = rng.choice(len(X_train), size=n, replace=replace)
    eval_reg.fit(X_train[idx], y_train[idx])

    # Regression output -> continuous values -> clip to [0,1] -> threshold
    preds_raw = eval_reg.predict(X_eval)
    probs = np.clip(preds_raw, 0.0, 1.0).astype(np.float32)
    preds = (probs >= threshold).astype(np.float32)

    TP = int(np.sum((preds == 1) & (y_eval == 1)))
    FP = int(np.sum((preds == 1) & (y_eval == 0)))
    FN = int(np.sum((preds == 0) & (y_eval == 1)))
    TN = int(np.sum((preds == 0) & (y_eval == 0)))
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision, 'recall': recall,
    }, probs


# ── Grid Search ──────────────────────────────────────────────


def build_combos(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all hyperparameter combinations from YAML config."""
    space = cfg['search_space']
    combos: list[dict[str, Any]] = []
    for lr, bs, n_est, wd, gc, crps_w, mse_w in itertools.product(
        space['learning_rate'],
        space['batch_size'],
        space['n_estimators'],
        space['weight_decay'],
        space['grad_clip_norm'],
        space.get('crps_loss_weight', [1.0]),
        space.get('mse_loss_weight', [1.0]),
    ):
        combos.append({
            'learning_rate': lr,
            'batch_size': bs,
            'n_estimators': n_est,
            'weight_decay': wd,
            'grad_clip_norm': gc,
            'crps_loss_weight': crps_w,
            'mse_loss_weight': mse_w,
        })
    return combos


def _combo_key(combo: dict[str, Any]) -> str:
    """Build a unique string key for a hyperparameter combo."""
    return (f"{combo['learning_rate']}|{combo['batch_size']}"
            f"|{combo['n_estimators']}|{combo['weight_decay']}"
            f"|{combo['grad_clip_norm']}"
            f"|{combo['crps_loss_weight']}|{combo['mse_loss_weight']}")


def run_grid_search(cfg: dict[str, Any], flight_features: FlightFeatures,
                    config_path: str = '',
                    results_path: str = 'results/grid_search_tabpfn_regression_results.json',
                    ) -> list[dict[str, Any]]:
    """Run all TabPFN regression grid search combinations and return results.

    Supports restart: loads existing results, skips completed combos,
    resumes under-trained ones from checkpoint.
    """
    combos = build_combos(cfg)
    epochs = cfg.get('epochs', 10)
    save_every = cfg.get('save_every', 5)
    seed = cfg.get('seed', 0)

    # Load previously completed results
    existing_results = load_existing_results(results_path)
    existing_by_key: dict[str, dict[str, Any]] = {}
    for r in existing_results:
        key = _combo_key({
            'learning_rate': r['learning_rate'],
            'batch_size': r['batch_size'],
            'n_estimators': r['n_estimators'],
            'weight_decay': r['weight_decay'],
            'grad_clip_norm': r['grad_clip_norm'],
            'crps_loss_weight': r['crps_loss_weight'],
            'mse_loss_weight': r['mse_loss_weight'],
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
        crps_w = combo['crps_loss_weight']
        mse_w = combo['mse_loss_weight']
        key = _combo_key(combo)

        ckpt_path = f'checkpoint/fire_detector_tabpfn_reg_run_{run_id:02d}.pt'

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
              f'n_est={n_est}, wd={wd}, gc={gc}, '
              f'crps={crps_w}, mse={mse_w}')
        print('=' * 60)
        if resume_ckpt:
            print(f'  Resuming from checkpoint')

        train_result = train_tabpfn_model(
            X_norm, y_norm,
            n_epochs=epochs, lr=lr_val, weight_decay=wd,
            batch_size=bs, n_estimators=n_est,
            grad_clip_norm=gc,
            crps_loss_weight=crps_w, mse_loss_weight=mse_w,
            seed=seed,
            quiet=False, resume_from=resume_ckpt,
            save_path=ckpt_path, save_every=save_every,
        )

        regressor = train_result['regressor']
        regressor_init = train_result['regressor_init']
        epochs_completed = train_result['epochs_completed']
        X_tr = train_result['X_train']
        y_tr = train_result['y_train']
        X_te = train_result['X_test']
        y_te = train_result['y_test']

        # Evaluate on internal train/test splits
        train_metrics, _ = _evaluate_tabpfn_regressor(
            regressor, regressor_init,
            X_tr, y_tr, X_tr, y_tr,
            batch_size=bs, seed=seed)
        test_metrics, _ = _evaluate_tabpfn_regressor(
            regressor, regressor_init,
            X_tr, y_tr, X_te, y_te,
            batch_size=bs, seed=seed)

        test_P = test_metrics['TP'] + test_metrics['FN']
        error_rate = (test_metrics['FN'] + test_metrics['FP']) / max(test_P, 1)

        # Build representative context for inference checkpoint
        ctx_X, ctx_y = build_representative_context(
            X_tr, y_tr, batch_size=bs, seed=seed)

        # Save full checkpoint
        torch.save({
            'model_state_dict': train_result['model'].state_dict(),
            'optimizer_state_dict': train_result['optimizer_state'],
            'epoch': epochs_completed,
            'regressor_init': regressor_init,
            'context_X': ctx_X,
            'context_y': ctx_y,
            'scaler': scaler,
            'threshold': 0.5,
            'feature_names': FEATURE_NAMES,
            'n_estimators': n_est,
            'batch_size': bs,
            'seed': seed,
            'loss_history': train_result['loss_history'][:, 0].tolist(),
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
            'crps_loss_weight': crps_w,
            'mse_loss_weight': mse_w,
            'epochs_completed': epochs_completed,
            'train': {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                      for k, v in train_metrics.items()},
            'test': {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                     for k, v in test_metrics.items()},
            'error_rate': float(error_rate),
            'final_loss': final_loss,
            'loss_history': loss_history[:, 0].tolist(),
            'checkpoint': ckpt_path,
        }
        results.append(result)

        save_incremental(results, cfg, config_path, results_path)

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

    print(f'\n{"=" * 130}')
    print(f'TabPFN Regression Grid Search Results (sorted by {metric}, best first)')
    print('=' * 130)

    header = (f'  {"Run":>3s}  {"LR":>8s}  {"BS":>4s}  {"Est":>3s}  '
              f'{"WD":>6s}  {"GC":>4s}  {"CRPS":>4s}  {"MSE":>4s}  '
              f'{"TP":>6s}  {"FP":>5s}  {"FN":>5s}  '
              f'{"Prec":>6s}  {"Rec":>6s}  {"ErrRate":>7s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for r in sorted_results:
        t = r['test']
        print(f'  {r["run_id"]:>3d}  {r["learning_rate"]:>8.1e}  '
              f'{r["batch_size"]:>4d}  {r["n_estimators"]:>3d}  '
              f'{r["weight_decay"]:>6.3f}  {r["grad_clip_norm"]:>4.1f}  '
              f'{r["crps_loss_weight"]:>4.1f}  {r["mse_loss_weight"]:>4.1f}  '
              f'{t["TP"]:>6.0f}  {t["FP"]:>5.0f}  {t["FN"]:>5.0f}  '
              f'{t["precision"]:>6.4f}  {t["recall"]:>6.4f}  '
              f'{r["error_rate"]:>7.4f}')


def write_best_model_config(best: dict[str, Any], checkpoint_path: str,
                            config_path: str = 'configs/best_model_tabpfn_regression.yaml',
                            ) -> str:
    """Write YAML config for the best model (consumed by realtime scripts)."""
    t = best['test']
    config = {
        'detector': 'ml',
        'model_type': 'tabpfn_regression',
        'checkpoint': checkpoint_path,
        'threshold': 0.5,
        'training': {
            'learning_rate': best['learning_rate'],
            'batch_size': best['batch_size'],
            'n_estimators': best['n_estimators'],
            'weight_decay': best['weight_decay'],
            'grad_clip_norm': best['grad_clip_norm'],
            'crps_loss_weight': best['crps_loss_weight'],
            'mse_loss_weight': best['mse_loss_weight'],
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
    """Run TabPFN regression grid search."""
    parser = argparse.ArgumentParser(
        description='Train TabPFN regressor for fire detection via grid search')
    parser.add_argument(
        '--config', type=str, default='configs/grid_search_tabpfn_regression.yaml',
        help='Path to YAML grid search config')
    args = parser.parse_args()

    print('=' * 60)
    print('ML Fire Detection \u2014 TabPFN Regression Training')
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

    results_path = 'results/grid_search_tabpfn_regression_results.json'
    results = run_grid_search(cfg, flight_features,
                              config_path=config_path,
                              results_path=results_path)

    print_results_table(results, metric)

    best = min(results, key=lambda r: r.get(metric, float('inf')))
    best_ckpt = best['checkpoint']
    best_path = 'checkpoint/fire_detector_tabpfn_regression_best.pt'
    shutil.copy2(best_ckpt, best_path)

    model_config_path = 'configs/best_model_tabpfn_regression.yaml'
    write_best_model_config(best, best_path, model_config_path)

    print(f'\n{"=" * 60}')
    print(f'Best run: #{best["run_id"]}')
    print(f'  LR:              {best["learning_rate"]}')
    print(f'  Batch size:      {best["batch_size"]}')
    print(f'  n_estimators:    {best["n_estimators"]}')
    print(f'  Weight decay:    {best["weight_decay"]}')
    print(f'  Grad clip:       {best["grad_clip_norm"]}')
    print(f'  CRPS weight:     {best["crps_loss_weight"]}')
    print(f'  MSE weight:      {best["mse_loss_weight"]}')
    print(f'  Error rate:      {best["error_rate"]:.4f}')
    print(f'  Precision:       {best["test"]["precision"]:.4f}')
    print(f'  Recall:          {best["test"]["recall"]:.4f}')
    print(f'\n  Best model:  {best_path}')
    print(f'  Config:      {model_config_path}')
    print(f'  Results:     {results_path}')
    print(f'\n  To use in realtime:')
    print(f'    python realtime_tabpfn.py --config {model_config_path}')

    from lib.plotting import plot_convergence_curves
    plot_convergence_curves(
        results_path, out_path='plots/convergence_tabpfn_regression.png',
        title='TabPFN Regressor — Training Convergence')
    print('Done.')


if __name__ == '__main__':
    main()

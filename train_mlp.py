"""train_mlp.py - YAML-driven grid search for fire detection MLP.

Iterates over combinations of loss function, architecture, learning rate,
and importance weights. Results are saved to JSON. The best model is saved
to checkpoint/fire_detector_mlp_best.pt for automatic discovery by inference.

Usage:
    python train_mlp.py --config configs/grid_search_mlp.yaml
    python train_mlp.py --loss bce --layers 64 64 64 32   # single run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
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
from lib.losses import (
    SoftErrorRateLoss, TverskyLoss, FocalErrorRateLoss, CombinedLoss,
)
from torch.nn.utils import clip_grad_norm_
from lib.training import (
    load_all_data, extract_train_test, oversample_minority,
    load_existing_results, save_incremental,
    compute_error_rate, coerce_metrics,
    FlightFeatures,
)
from lib.constants import THERMAL_FEATURE_INDICES, NON_THERMAL_FEATURE_INDICES

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
    gt_mask: NDArrayFloat | None = None,
    gt_fp_penalty: float = 5.0,
    loss_kwargs: dict[str, Any] | None = None,
    grad_clip_norm: float = 1.0,
    dropout: float = 0.0,
    quiet: bool = False,
    resume_from: str | None = None,
    save_path: str | None = None,
    save_every: int = 25,
    X_test: NDArrayFloat | None = None,
    y_test: NDArrayFloat | None = None,
    force_cpu: bool = False,
) -> TrainResult:
    """Train FireMLP with the selected loss function.

    Args:
        loss_kwargs: Extra hyperparameters for the loss (e.g. alpha/beta
            for Tversky, gamma for focal, lam for combined).
        grad_clip_norm: Max gradient norm for clipping. Default 1.0.
        dropout: Dropout probability for FireMLP hidden layers. Default 0.0.
        quiet: If True, suppress per-epoch progress bar.
        resume_from: Checkpoint path to resume training from.
        save_path: If set, save checkpoint to this path periodically.
        save_every: Save checkpoint every N epochs (default 25).
        X_test: Optional normalized test features for per-epoch evaluation.
        y_test: Optional test labels for per-epoch evaluation.
        force_cpu: If True, train on CPU only (useful for parallel workers).

    Returns:
        Dict with keys: model, loss_history, epochs_completed,
        optimizer_state, scheduler_state.
    """
    device = get_device(force_cpu=force_cpu)
    if not quiet:
        print(f'  Device: {device}')

    model = FireMLP(hidden_layers=hidden_layers, dropout=dropout).to(device)
    lk = loss_kwargs or {}

    # Build loss criterion based on loss_fn string.
    # All losses except BCE use a unified forward(logits, y, w, gt_mask).
    uses_unified_forward = True
    if loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_label = 'Pixel Weighted BCE'
        uses_unified_forward = False
    elif loss_fn == 'error-rate':
        if P_total is None:
            P_total = float(y_train.sum())
        criterion = SoftErrorRateLoss(P_total, gt_fp_penalty=gt_fp_penalty).to(device)
        loss_label = f'Soft (FN+FP)/P, P={P_total:.0f}, gt_pen={gt_fp_penalty}'
    elif loss_fn == 'tversky':
        if P_total is None:
            P_total = float(y_train.sum())
        alpha = lk.get('alpha', 0.3)
        beta = lk.get('beta', 0.7)
        criterion = TverskyLoss(
            alpha=alpha, beta=beta, P_total=P_total,
            gt_fp_penalty=gt_fp_penalty).to(device)
        loss_label = f'Tversky(a={alpha},b={beta}), gt_pen={gt_fp_penalty}'
    elif loss_fn == 'focal-error-rate':
        if P_total is None:
            P_total = float(y_train.sum())
        gamma = lk.get('gamma', 2.0)
        criterion = FocalErrorRateLoss(
            P_total, gamma=gamma, gt_fp_penalty=gt_fp_penalty).to(device)
        loss_label = f'FocalER(g={gamma}), P={P_total:.0f}'
    elif loss_fn == 'combined':
        if P_total is None:
            P_total = float(y_train.sum())
        lam = lk.get('lam', 0.5)
        criterion = CombinedLoss(
            P_total, lam=lam, gt_fp_penalty=gt_fp_penalty).to(device)
        loss_label = f'Combined(lam={lam}), P={P_total:.0f}'
    else:
        raise ValueError(f'Unknown loss_fn: {loss_fn!r}')
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
            'loss_history': np.zeros((0, 7)),
            'epochs_completed': n_epochs,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
        }

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    w_t = torch.tensor(w_train, dtype=torch.float32)
    if gt_mask is not None:
        gt_t = torch.tensor(gt_mask, dtype=torch.float32)
    else:
        gt_t = torch.zeros_like(y_t)

    dataset = torch.utils.data.TensorDataset(X_t, y_t, w_t, gt_t)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # Prepare test tensors for per-epoch evaluation
    has_test = X_test is not None and y_test is not None
    X_test_t: torch.Tensor | None = None
    y_test_t: torch.Tensor | None = None
    if has_test:
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    remaining = n_epochs - start_epoch
    # Columns: [loss, train_FP, train_FN, train_TP, test_FP, test_FN, test_TP]
    loss_history = np.zeros((remaining, 7))
    pbar = trange(remaining, desc=loss_label, disable=quiet)
    for i in pbar:
        epoch = start_epoch + i
        model.train()
        total_loss = 0.0
        train_FP = 0
        train_FN = 0
        train_TP = 0
        for X_batch, y_batch, w_batch, gt_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            gt_batch = gt_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            if uses_unified_forward:
                loss = criterion(logits, y_batch, w_batch, gt_mask=gt_batch)
            else:
                loss_per_sample = criterion(logits, y_batch)
                loss = (loss_per_sample * w_batch).mean()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

            # Accumulate train FP/FN/TP (reuses logits, zero overhead)
            with torch.no_grad():
                preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
                train_FP += int(((preds == 1) & (y_batch == 0)).sum().item())
                train_FN += int(((preds == 0) & (y_batch == 1)).sum().item())
                train_TP += int(((preds == 1) & (y_batch == 1)).sum().item())

        scheduler.step()
        avg_loss = total_loss / len(X_t)

        # Per-epoch test evaluation (cheap forward pass, no gradients)
        test_FP, test_FN, test_TP = 0, 0, 0
        if has_test:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test_t)
                test_preds = (torch.sigmoid(test_logits) >= 0.5).float()
                test_FP = int(((test_preds == 1) & (y_test_t == 0)).sum().item())
                test_FN = int(((test_preds == 0) & (y_test_t == 1)).sum().item())
                test_TP = int(((test_preds == 1) & (y_test_t == 1)).sum().item())

        loss_history[i] = [avg_loss, train_FP, train_FN, train_TP,
                           test_FP, test_FN, test_TP]
        cur_lr = scheduler.get_last_lr()[0]
        if not quiet:
            pbar.set_postfix({'loss': f'{avg_loss:.4e}', 'lr': f'{cur_lr:.1e}',
                              'ep': f'{epoch + 1}/{n_epochs}',
                              'tFP': train_FP, 'tFN': train_FN})

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
    dropouts = space.get('dropout', [0.0])
    grad_clips = space.get('grad_clip_norm', [1.0])
    norms = space.get('normalization', ['hybrid'])

    # Loss-specific hyperparameters
    tversky_params = space.get('tversky_params', [{'alpha': 0.3, 'beta': 0.7}])
    focal_params = space.get('focal_params', [{'gamma': 2.0}])
    combined_params = space.get('combined_params', [{'lam': 0.5}])

    # Per-loss overrides for layers / learning_rate
    loss_overrides = space.get('loss_overrides', {})
    DEFAULT_WEIGHTS = {'gt': 10, 'fire': 5, 'other': 1}

    combos: list[dict[str, Any]] = []
    for loss in losses:
        override = loss_overrides.get(loss, {})
        loss_layers = override.get('layers', layers_list)
        loss_lrs = override.get('learning_rate', lrs)

        for layers, lr, dp, gc, norm in itertools.product(
                loss_layers, loss_lrs, dropouts, grad_clips, norms):
            base = {
                'loss': loss,
                'layers': list(layers),
                'learning_rate': lr,
                'dropout': dp,
                'grad_clip_norm': gc,
                'normalization': norm,
            }

            if loss == 'bce':
                for wc in weight_configs:
                    combos.append({**base, 'importance_weights': dict(wc),
                                   'loss_kwargs': {}})
            elif loss == 'error-rate':
                combos.append({**base, 'importance_weights': DEFAULT_WEIGHTS,
                               'loss_kwargs': {}})
            elif loss == 'tversky':
                for tp in tversky_params:
                    combos.append({**base, 'importance_weights': DEFAULT_WEIGHTS,
                                   'loss_kwargs': dict(tp)})
            elif loss == 'focal-error-rate':
                for fp in focal_params:
                    combos.append({**base, 'importance_weights': DEFAULT_WEIGHTS,
                                   'loss_kwargs': dict(fp)})
            elif loss == 'combined':
                for cp in combined_params:
                    combos.append({**base, 'importance_weights': DEFAULT_WEIGHTS,
                                   'loss_kwargs': dict(cp)})
    return combos


def _combo_key(combo: dict[str, Any]) -> str:
    """Build a unique string key for a hyperparameter combo."""
    wc = combo['importance_weights']
    lk = combo.get('loss_kwargs', {})
    lk_str = '|'.join(f'{k}={v}' for k, v in sorted(lk.items())) if lk else ''
    dp = combo.get('dropout', 0.0)
    gc = combo.get('grad_clip_norm', 1.0)
    norm = combo.get('normalization', 'hybrid')
    return (f"{combo['loss']}|{combo['layers']}|{combo['learning_rate']}"
            f"|{wc['gt']}/{wc['fire']}/{wc['other']}|{lk_str}|dp={dp}|gc={gc}"
            f"|norm={norm}")


def run_grid_search(cfg: dict[str, Any], flight_features: FlightFeatures,
                    config_path: str = '',
                    results_path: str = 'results/grid_search_mlp_results.json',
                    worker_id: int = 0,
                    num_workers: int = 1,
                    force_cpu: bool = False,
                    ) -> list[dict[str, Any]]:
    """Run all grid search combinations and return results.

    Supports restart: loads existing results from results_path, skips
    fully-trained combos, and resumes under-trained ones.

    Args:
        worker_id: This worker's ID (0-based). Default 0.
        num_workers: Total parallel workers. Default 1 (sequential).
        force_cpu: If True, train on CPU only.
    """
    all_combos = build_combos(cfg)
    # Tag each combo with its original run_id (1-based) so checkpoint
    # filenames stay stable across workers.
    indexed_combos = [(i + 1, c) for i, c in enumerate(all_combos)]
    if num_workers > 1:
        indexed_combos = [(rid, c) for rid, c in indexed_combos
                          if (rid - 1) % num_workers == worker_id]
        print(f'\n  Worker {worker_id}/{num_workers}: handling '
              f'{len(indexed_combos)}/{len(all_combos)} combos')
    n_combos_total = len(all_combos)
    epochs = cfg.get('epochs', 100)
    batch_size = cfg.get('batch_size', 4096)
    save_every = cfg.get('save_every', 25)

    # Load previously completed results (restart support)
    existing_results = load_existing_results(results_path)
    existing_by_key: dict[str, dict[str, Any]] = {}
    for r in existing_results:
        key = _combo_key({
            'loss': r['loss'], 'layers': r['layers'],
            'learning_rate': r['learning_rate'],
            'importance_weights': r['importance_weights'],
            'loss_kwargs': r.get('loss_kwargs', {}),
            'dropout': r.get('dropout', 0.0),
            'grad_clip_norm': r.get('grad_clip_norm', 1.0),
            'normalization': r.get('normalization', 'hybrid'),
        })
        existing_by_key[key] = r

    n_skip = sum(1 for r in existing_by_key.values()
                 if r.get('epochs_completed', 0) >= epochs)
    n_resume = sum(1 for r in existing_by_key.values()
                   if 0 < r.get('epochs_completed', 0) < epochs)
    if existing_results:
        print(f'\n  Restart: {n_skip} fully trained, '
              f'{n_resume} under-trained (will resume), '
              f'{len(indexed_combos) - len(existing_by_key)} new')

    # Normalization setup — prepare both scalers upfront.
    #   hybrid:   thermal features / T_ignition, non-thermal via StandardScaler
    #   standard: all 12 features via StandardScaler (fit on GT flight)
    T_ign = cfg.get('T_ignition', 300.0) + 273.15  # YAML is °C, internal is K
    therm_idx = THERMAL_FEATURE_INDICES      # [0, 1, 2, 3]
    non_therm_idx = NON_THERMAL_FEATURE_INDICES  # [4, 5, 6, 7, 8, 9, 10, 11]

    gt_X = flight_features['24-801-03']['X']
    gt_X_clean = np.where(np.isfinite(gt_X), gt_X, 0.0).astype(np.float32)

    # Hybrid scaler: fit only on non-thermal columns
    hybrid_scaler = StandardScaler()
    hybrid_scaler.fit(gt_X_clean[:, non_therm_idx])

    # Standard scaler: fit on all 12 columns
    standard_scaler = StandardScaler()
    standard_scaler.fit(gt_X_clean)

    os.makedirs('checkpoint', exist_ok=True)
    results: list[dict[str, Any]] = [
        r for r in existing_results
        if r.get('epochs_completed', 0) >= epochs
    ]

    for run_id, combo in indexed_combos:
        loss = combo['loss']
        layers = combo['layers']
        lr = combo['learning_rate']
        wc = combo['importance_weights']
        loss_kwargs = combo.get('loss_kwargs', {})
        dp = combo.get('dropout', 0.0)
        gc = combo.get('grad_clip_norm', 1.0)
        norm = combo.get('normalization', 'hybrid')
        key = _combo_key(combo)

        # Select scaler based on normalization strategy
        if norm == 'standard':
            scaler = standard_scaler
        else:
            scaler = hybrid_scaler

        ckpt_path = f'checkpoint/fire_detector_mlp_run_{run_id:02d}.pt'

        # Skip fully-trained runs
        existing = existing_by_key.get(key)
        if existing and existing.get('epochs_completed', 0) >= epochs:
            print(f'\n  Run {run_id}/{n_combos_total}: SKIPPED (already completed)')
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
        extra = ''
        if loss_kwargs:
            extra += f', {loss_kwargs}'
        if dp > 0:
            extra += f', dropout={dp}'
        print(f'\n{"=" * 60}')
        print(f'Run {run_id}/{n_combos_total}: {loss}, {arch_str}, lr={lr}, '
              f'norm={norm}, weights=[{wt_str}]{extra}')
        print('=' * 60)
        if resume_ckpt:
            print(f'  Resuming from checkpoint ({resume_epochs}/{epochs} epochs)')

        (X_train, y_train, w_train, flight_src_train,
         X_test, y_test, w_test, flight_src_test) = extract_train_test(
            flight_features,
            train_flights=['24-801-04', '24-801-05'],
            test_flights=['24-801-06'],
            ground_truth_flight='24-801-03',
            gt_test_ratio=0.2,
            importance_gt=float(wc['gt']),
            importance_fire=float(wc['fire']),
            importance_other=float(wc['other']),
        )

        # Build GT mask before oversampling, carry through as extra column
        gt_mask_train = (flight_src_train == '24-801-03').astype(np.float32)
        gt_col = gt_mask_train.reshape(-1, 1)
        X_aug = np.concatenate([X_train, gt_col], axis=1)
        X_aug_ready, y_ready, w_ready = oversample_minority(X_aug, y_train, w_train)
        X_ready = X_aug_ready[:, :-1]       # features
        gt_mask_ready = X_aug_ready[:, -1]   # GT mask (survived same shuffle)
        P_total = float(y_ready.sum())  # After oversampling

        X_clean = np.where(np.isfinite(X_ready), X_ready, 0.0).astype(np.float32)
        X_test_clean = np.where(np.isfinite(X_test), X_test, 0.0).astype(np.float32)

        if norm == 'standard':
            X_norm = scaler.transform(X_clean).astype(np.float32)
            X_test_norm = scaler.transform(X_test_clean).astype(np.float32)
        else:  # hybrid
            X_norm = np.empty_like(X_clean)
            X_norm[:, therm_idx] = X_clean[:, therm_idx] / T_ign
            X_norm[:, non_therm_idx] = scaler.transform(
                X_clean[:, non_therm_idx]).astype(np.float32)
            X_test_norm = np.empty_like(X_test_clean)
            X_test_norm[:, therm_idx] = X_test_clean[:, therm_idx] / T_ign
            X_test_norm[:, non_therm_idx] = scaler.transform(
                X_test_clean[:, non_therm_idx]).astype(np.float32)

        train_result = train_model(
            X_norm, y_ready, w_ready,
            n_epochs=epochs, lr=lr, batch_size=batch_size,
            loss_fn=loss, hidden_layers=layers, P_total=P_total,
            gt_mask=gt_mask_ready, loss_kwargs=loss_kwargs,
            grad_clip_norm=gc, dropout=dp,
            quiet=False, resume_from=resume_ckpt,
            save_path=ckpt_path, save_every=save_every,
            X_test=X_test_norm, y_test=y_test,
            force_cpu=force_cpu)

        model = train_result['model']
        loss_history = train_result['loss_history']
        epochs_completed = train_result['epochs_completed']

        X_train_eval = np.where(np.isfinite(X_train), X_train, 0.0).astype(np.float32)
        if norm == 'standard':
            X_train_norm = scaler.transform(X_train_eval).astype(np.float32)
        else:  # hybrid
            X_train_norm = np.empty_like(X_train_eval)
            X_train_norm[:, therm_idx] = X_train_eval[:, therm_idx] / T_ign
            X_train_norm[:, non_therm_idx] = scaler.transform(
                X_train_eval[:, non_therm_idx]).astype(np.float32)
        train_metrics, _ = evaluate(model, X_train_norm, y_train)
        test_metrics, _ = evaluate(model, X_test_norm, y_test)

        # Per-flight evaluation: Flight 03 FP
        gt_mask_test = flight_src_test == '24-801-03'
        gt03_metrics, _ = evaluate(model, X_test_norm[gt_mask_test],
                                   y_test[gt_mask_test])

        error_rate = compute_error_rate(test_metrics)

        # Build epoch_metrics dict from expanded loss_history columns
        epoch_metrics = {
            'train_FP': loss_history[:, 1].astype(int).tolist(),
            'train_FN': loss_history[:, 2].astype(int).tolist(),
            'train_TP': loss_history[:, 3].astype(int).tolist(),
            'test_FP': loss_history[:, 4].astype(int).tolist(),
            'test_FN': loss_history[:, 5].astype(int).tolist(),
            'test_TP': loss_history[:, 6].astype(int).tolist(),
        }

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': train_result['optimizer_state'],
            'scheduler_state': train_result['scheduler_state'],
            'epochs_completed': epochs_completed,
            'mean': scaler.mean_,
            'std': scaler.scale_,
            'scaler': scaler,
            'T_ignition': T_ign,
            'normalization': norm,
            'n_features': 12,
            'hidden_layers': model.hidden_layers,
            'dropout': dp,
            'threshold': 0.5,
            'feature_names': FEATURE_NAMES,
            'loss_fn': loss,
            'loss_kwargs': loss_kwargs,
            'loss_history': loss_history[:, 0].tolist(),
            'epoch_metrics': epoch_metrics,
        }, ckpt_path)

        final_loss = float(loss_history[-1, 0]) if len(loss_history) > 0 else 0.0

        result = {
            'run_id': run_id,
            'loss': loss,
            'layers': layers,
            'learning_rate': lr,
            'importance_weights': wc,
            'loss_kwargs': loss_kwargs,
            'dropout': dp,
            'grad_clip_norm': gc,
            'normalization': norm,
            'epochs_completed': epochs_completed,
            'train': coerce_metrics(train_metrics),
            'test': coerce_metrics(test_metrics),
            'error_rate': float(error_rate),
            'flight03_FP': int(gt03_metrics['FP']),
            'final_loss': final_loss,
            'loss_history': loss_history[:, 0].tolist(),
            'epoch_metrics': epoch_metrics,
            'checkpoint': ckpt_path,
        }
        results.append(result)

        save_incremental(results, cfg, config_path, results_path)

        # Checkpoint cleanup: keep only best + currently running
        metric = cfg.get('metric', 'error_rate')
        THRESHOLD_FP = 70
        eligible = [r for r in results
                    if r.get('flight03_FP', float('inf')) <= THRESHOLD_FP]
        if eligible:
            current_best = min(eligible,
                               key=lambda r: r['test']['FP'] + r['test']['FN'])
        else:
            current_best = min(results,
                               key=lambda r: r.get('flight03_FP', float('inf')))
        best_so_far_path = 'checkpoint/fire_detector_mlp_best.pt'

        if result['run_id'] == current_best['run_id']:
            shutil.copy2(ckpt_path, best_so_far_path)


        print(f'  Train: TP={train_metrics["TP"]:,} FP={train_metrics["FP"]:,} '
              f'FN={train_metrics["FN"]:,} TN={train_metrics["TN"]:,}')
        print(f'  Test:  TP={test_metrics["TP"]:,} FP={test_metrics["FP"]:,} '
              f'FN={test_metrics["FN"]:,} TN={test_metrics["TN"]:,}')
        print(f'  Flight 03 FP: {gt03_metrics["FP"]:,} '
              f'(threshold baseline: 70)')
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

    header = (f'  {"Run":>3s}  {"Loss":<16s}  {"Layers":<18s}  {"LR":>8s}  '
              f'{"Norm":<8s}  {"Weights":<12s}  {"Extra":<16s}  {"TP":>6s}  '
              f'{"FP":>5s}  {"FN":>5s}  {"Prec":>6s}  {"Rec":>6s}  '
              f'{"ErrRate":>7s}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for r in sorted_results:
        wc = r['importance_weights']
        wt_str = f"{wc['gt']}/{wc['fire']}/{wc['other']}"
        layers_str = 'x'.join(str(h) for h in r['layers'])
        norm_str = r.get('normalization', 'hybrid')
        lk = r.get('loss_kwargs', {})
        dp = r.get('dropout', 0.0)
        extra_parts = [f'{k}={v}' for k, v in sorted(lk.items())]
        if dp > 0:
            extra_parts.append(f'dp={dp}')
        extra_str = ','.join(extra_parts) if extra_parts else '-'
        t = r['test']
        print(f'  {r["run_id"]:>3d}  {r["loss"]:<16s}  {layers_str:<18s}  '
              f'{r["learning_rate"]:>8.4f}  {norm_str:<8s}  {wt_str:<12s}  '
              f'{extra_str:<16s}  {t["TP"]:>6.0f}  {t["FP"]:>5.0f}  '
              f'{t["FN"]:>5.0f}  {t["precision"]:>6.4f}  {t["recall"]:>6.4f}  '
              f'{r["error_rate"]:>7.4f}')


def write_best_model_config(best: dict[str, Any], checkpoint_path: str,
                            config_path: str = 'configs/best_model_mlp.yaml',
                            ) -> str:
    """Write YAML config for the best model (consumed by realtime scripts)."""
    wc = best['importance_weights']
    t = best['test']
    config = {
        'detector': 'ml',
        'model_type': 'firemlp',
        'checkpoint': checkpoint_path,
        'threshold': 0.5,
        'T_ignition': 300,  # [°C]
        'normalization': best.get('normalization', 'hybrid'),
        'training': {
            'loss': best['loss'],
            'layers': best['layers'],
            'learning_rate': best['learning_rate'],
            'epochs': best['epochs_completed'],
            'dropout': best.get('dropout', 0.0),
            'grad_clip_norm': best.get('grad_clip_norm', 1.0),
            'loss_kwargs': best.get('loss_kwargs', {}),
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


def write_results_csv(results: list[dict[str, Any]],
                      csv_path: str = 'results/grid_search_mlp_summary.csv',
                      ) -> str:
    """Write a CSV summary of all grid search runs for easy sorting/comparison."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    fieldnames = [
        'run_id', 'loss', 'layers', 'learning_rate', 'normalization',
        'importance_gt', 'importance_fire', 'importance_other',
        'dropout', 'grad_clip_norm', 'epochs_completed',
        'train_TP', 'train_FP', 'train_FN',
        'test_TP', 'test_FP', 'test_FN',
        'flight03_FP', 'error_rate', 'precision', 'recall',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x['run_id']):
            wc = r['importance_weights']
            tr = r['train']
            te = r['test']
            writer.writerow({
                'run_id': r['run_id'],
                'loss': r['loss'],
                'layers': 'x'.join(str(h) for h in r['layers']),
                'learning_rate': r['learning_rate'],
                'normalization': r.get('normalization', 'hybrid'),
                'importance_gt': wc['gt'],
                'importance_fire': wc['fire'],
                'importance_other': wc['other'],
                'dropout': r.get('dropout', 0.0),
                'grad_clip_norm': r.get('grad_clip_norm', 1.0),
                'epochs_completed': r['epochs_completed'],
                'train_TP': int(tr['TP']),
                'train_FP': int(tr['FP']),
                'train_FN': int(tr['FN']),
                'test_TP': int(te['TP']),
                'test_FP': int(te['FP']),
                'test_FN': int(te['FN']),
                'flight03_FP': r.get('flight03_FP', ''),
                'error_rate': f'{r["error_rate"]:.6f}',
                'precision': f'{te["precision"]:.6f}',
                'recall': f'{te["recall"]:.6f}',
            })
    return csv_path


# ── Worker Merge ─────────────────────────────────────────────


def merge_worker_results(
    cfg: dict[str, Any],
    config_path: str,
    results_dir: str = 'results',
    merged_path: str = 'results/grid_search_mlp_results.json',
) -> list[dict[str, Any]]:
    """Merge per-worker result files into a single combined JSON.

    Globs results/grid_search_mlp_worker_*.json, deduplicates by combo key,
    and writes the merged result to merged_path.
    """
    import glob as globmod
    from datetime import datetime

    pattern = os.path.join(results_dir, 'grid_search_mlp_worker_*.json')
    worker_files = sorted(globmod.glob(pattern))
    if not worker_files:
        print(f'  No worker result files found matching {pattern}')
        return []

    print(f'\n  Merging {len(worker_files)} worker result files...')
    seen_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    for wf in worker_files:
        print(f'    {wf}')
        with open(wf) as f:
            data = json.load(f)
        for r in data.get('results', []):
            key = _combo_key({
                'loss': r['loss'], 'layers': r['layers'],
                'learning_rate': r['learning_rate'],
                'importance_weights': r['importance_weights'],
                'loss_kwargs': r.get('loss_kwargs', {}),
                'dropout': r.get('dropout', 0.0),
                'grad_clip_norm': r.get('grad_clip_norm', 1.0),
                'normalization': r.get('normalization', 'hybrid'),
            })
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(r)

    # Write merged JSON
    metric = cfg.get('metric', 'error_rate')
    output = {
        'config': config_path,
        'timestamp': datetime.now().isoformat(),
        'metric': metric,
        'results': merged,
    }
    if merged:
        best = min(merged, key=lambda r: r.get(metric, float('inf')))
        output['best_run_id'] = best['run_id']
        output[f'best_{metric}'] = best[metric]
    os.makedirs(os.path.dirname(merged_path) or '.', exist_ok=True)
    with open(merged_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'  Merged {len(merged)} results -> {merged_path}')
    return merged


# ── Main ──────────────────────────────────────────────────────


def _finalize_results(results: list[dict[str, Any]], metric: str,
                      results_path: str) -> None:
    """Select best model, write configs/CSV/plots from completed results."""
    print_results_table(results, metric)

    csv_path = 'results/grid_search_mlp_summary.csv'
    write_results_csv(results, csv_path)
    print(f'\n  CSV summary: {csv_path}')

    # Select best model: FP must be <= baseline threshold, then minimize FP+FN
    THRESHOLD_FP = 70
    eligible = [r for r in results
                if r.get('flight03_FP', float('inf')) <= THRESHOLD_FP]
    if eligible:
        best = min(eligible,
                   key=lambda r: r['test']['FP'] + r['test']['FN'])
        print(f'\n  {len(eligible)}/{len(results)} models have Flight 03 FP <= {THRESHOLD_FP}')
    else:
        best = min(results, key=lambda r: r.get('flight03_FP', float('inf')))
        print(f'\n  WARNING: No model has Flight 03 FP <= {THRESHOLD_FP}. '
              f'Selected lowest FP: {best.get("flight03_FP")}')
    best_ckpt = best['checkpoint']
    best_path = 'checkpoint/fire_detector_mlp_best.pt'
    if os.path.isfile(best_ckpt):
        shutil.copy2(best_ckpt, best_path)

    model_config_path = 'configs/best_model_mlp.yaml'
    write_best_model_config(best, best_path, model_config_path)

    wc = best['importance_weights']
    te = best['test']
    arch = ' -> '.join(['12'] + [str(h) for h in best['layers']] + ['1'])
    print(f'\n{"=" * 60}')
    print(f'Best run: #{best["run_id"]}  (lowest FP+FN = {te["FP"] + te["FN"]:.0f})')
    print(f'  Loss:         {best["loss"]}')
    lk = best.get('loss_kwargs', {})
    if lk:
        print(f'  Loss params:  {lk}')
    print(f'  Architecture: {arch}')
    print(f'  LR:           {best["learning_rate"]}')
    print(f'  Norm:         {best.get("normalization", "hybrid")}')
    print(f'  Weights:      gt={wc["gt"]}, fire={wc["fire"]}, other={wc["other"]}')
    dp = best.get('dropout', 0.0)
    if dp > 0:
        print(f'  Dropout:      {dp}')
    print(f'  Grad clip:    {best.get("grad_clip_norm", 1.0)}')
    print(f'  Test TP={te["TP"]:.0f}  FP={te["FP"]:.0f}  FN={te["FN"]:.0f}')
    print(f'  Error rate:   {best["error_rate"]:.4f}')
    print(f'  Precision:    {te["precision"]:.4f}')
    print(f'  Recall:       {te["recall"]:.4f}')
    print(f'\n  Best model:  {best_path}')
    print(f'  Config:      {model_config_path}')
    print(f'  Results:     {results_path}')
    print(f'\n  To compare per flight:')
    print(f'    python compare_fire_detectors.py --model {best_path}')
    print(f'\n  To use in realtime:')
    print(f'    python realtime_mlp.py --config {model_config_path}')

    from lib.plotting import plot_convergence_curves, plot_best_model_metrics
    plot_convergence_curves(results_path, out_path='plots/convergence_mlp.png',
                            title='MLP Fire Detector — Best Model Convergence',
                            top_n=1)
    if best.get('epoch_metrics'):
        plot_best_model_metrics(best, out_path='plots/best_model_metrics_mlp.png')
    print('Done.')


def main() -> None:
    """Run MLP grid search or single training run."""
    parser = argparse.ArgumentParser(
        description='Train MLP fire detector via YAML grid search')
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML grid search config (default: configs/grid_search_mlp.yaml)')
    parser.add_argument('--loss', choices=[
        'bce', 'error-rate', 'tversky', 'focal-error-rate', 'combined',
    ], default=None)
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    # Parallel worker support
    parser.add_argument('--worker-id', type=int, default=0,
                        help='Worker ID for parallel grid search (0-based)')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Total number of parallel workers (default: 1 = sequential)')
    parser.add_argument('--merge-results', action='store_true',
                        help='Merge per-worker result files and finalize')
    args = parser.parse_args()

    single_mode = args.loss is not None or args.layers is not None or args.lr is not None
    if single_mode and args.config:
        parser.error('Cannot use --config with --loss/--layers/--lr')
    if single_mode and (args.num_workers > 1 or args.merge_results):
        parser.error('Cannot use --num-workers/--merge-results in single-run mode')

    # Load config
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

    metric = cfg.get('metric', 'error_rate')
    results_path = 'results/grid_search_mlp_results.json'

    # ── Merge mode: combine worker results and finalize ──
    if args.merge_results:
        print('=' * 60)
        print('ML Fire Detection — Merging Worker Results')
        print('=' * 60)
        results = merge_worker_results(cfg, config_path,
                                       merged_path=results_path)
        if results:
            _finalize_results(results, metric, results_path)
        return

    # ── Training mode (sequential or worker) ──
    parallel = args.num_workers > 1
    force_cpu = parallel

    print('=' * 60)
    print('ML Fire Detection — MLP Hyperparameter Tuning')
    if parallel:
        print(f'  Worker {args.worker_id} of {args.num_workers} (CPU only)')
    print('=' * 60)

    data_dir = cfg.get('data_dir', 'ignite_fire_data')
    print('\n--- Loading flight data ---')
    print(f'  Data directory: {data_dir}')
    flights = group_files_by_flight(data_dir)
    flight_features = load_all_data(flights)

    combos = build_combos(cfg)
    print(f'\nConfig: {config_path}')
    print(f'Total combos: {len(combos)} | Epochs: {cfg.get("epochs", 100)} | Metric: {metric}')

    # Per-worker results path to avoid file conflicts
    if parallel:
        results_path = f'results/grid_search_mlp_worker_{args.worker_id}.json'

    results = run_grid_search(cfg, flight_features,
                              config_path=config_path,
                              results_path=results_path,
                              worker_id=args.worker_id,
                              num_workers=args.num_workers,
                              force_cpu=force_cpu)

    if not parallel:
        # Sequential mode: finalize immediately
        _finalize_results(results, metric, results_path)
    else:
        print(f'\n  Worker {args.worker_id} finished. '
              f'Results: {results_path}')
        print(f'  Run --merge-results after all workers complete.')


if __name__ == '__main__':
    main()

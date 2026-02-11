"""Diagnostic plots for ML fire detection training."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from models.firemlp import FireMLP

NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]


def plot_training_loss(loss_history: NDArrayFloat) -> None:
    """Plot training loss curve.

    Saves plot to plots/tune_training_loss.png.

    Args:
        loss_history: Array of per-epoch average loss values.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', linewidth=1.5, label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/tune_training_loss.png', dpi=150, bbox_inches='tight')
    print('  Saved plots/tune_training_loss.png')
    plt.close()


def plot_probability_hist(
    probs_fire: NDArrayFloat,
    probs_nofire: NDArrayFloat,
) -> None:
    """Plot P(fire) histogram for fire vs non-fire locations.

    Visualizes model calibration by showing probability distributions
    for each class. Well-calibrated model should show separation.

    Saves plot to plots/tune_probability_hist.png.

    Args:
        probs_fire: P(fire) for true fire locations.
        probs_nofire: P(fire) for true non-fire locations.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs_nofire, bins=50, alpha=0.6, label='No fire', color='gray',
            density=True)
    ax.hist(probs_fire, bins=50, alpha=0.6, label='Fire', color='red',
            density=True)
    ax.axvline(0.5, color='black', linestyle='--', linewidth=1.5,
               label='Threshold (0.5)')
    ax.set_xlabel('P(fire)')
    ax.set_ylabel('Density')
    ax.set_title('Fire Probability Distribution (Test Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/tune_probability_hist.png', dpi=150, bbox_inches='tight')
    print('  Saved plots/tune_probability_hist.png')
    plt.close()


def plot_prediction_map(
    flight_features: FlightFeatures,
    model: FireMLP,
    scaler: StandardScaler,
    flight_num: str,
) -> None:
    """Plot spatial fire predictions for one flight.

    Creates 2x2 subplot comparing ML predictions vs threshold detector:
        - Top-left: ML predictions (red = fire)
        - Top-right: Threshold detector labels
        - Bottom-left: Agreement (green=both, blue=ML only, orange=thresh only)
        - Bottom-right: P(fire) heatmap

    Saves plot to plots/tune_prediction_map_{flight_num}.png.

    Args:
        flight_features: Dict from load_all_data().
        model: Trained model.
        scaler: Fitted scaler for feature normalization.
        flight_num: Flight ID to plot (e.g., '24-801-06').
    """
    d = flight_features[flight_num]
    X, y = d['X'], d['y']
    lats, lons = d['lats'], d['lons']

    X_clean = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    X_norm = scaler.transform(X_clean).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_norm))
        probs = torch.sigmoid(logits).numpy()

    ml_fire = probs >= 0.5
    thresh_fire = y > 0.5

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'ML vs Threshold \u2014 Flight {flight_num} ({d["comment"]})',
        fontsize=14, fontweight='bold')

    # Top-left: ML predictions
    ax = axes[0, 0]
    ax.scatter(lons[~ml_fire], lats[~ml_fire], s=0.1, c='gray', alpha=0.2)
    if ml_fire.any():
        ax.scatter(lons[ml_fire], lats[ml_fire], s=1, c='red', alpha=0.7)
    ax.set_title(f'ML Predictions ({int(ml_fire.sum()):,} fire locations)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Top-right: Threshold labels
    ax = axes[0, 1]
    ax.scatter(lons[~thresh_fire], lats[~thresh_fire], s=0.1, c='gray',
               alpha=0.2)
    if thresh_fire.any():
        ax.scatter(lons[thresh_fire], lats[thresh_fire], s=1, c='red',
                   alpha=0.7)
    ax.set_title(f'Threshold Labels ({int(thresh_fire.sum()):,} fire locations)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Bottom-left: Agreement
    ax = axes[1, 0]
    both = ml_fire & thresh_fire
    ml_only = ml_fire & ~thresh_fire
    thresh_only = ~ml_fire & thresh_fire
    neither = ~ml_fire & ~thresh_fire
    ax.scatter(lons[neither], lats[neither], s=0.1, c='gray', alpha=0.1)
    if both.any():
        ax.scatter(lons[both], lats[both], s=1.5, c='green', alpha=0.7,
                   label=f'Both ({int(both.sum()):,})')
    if ml_only.any():
        ax.scatter(lons[ml_only], lats[ml_only], s=1.5, c='blue', alpha=0.7,
                   label=f'ML only ({int(ml_only.sum()):,})')
    if thresh_only.any():
        ax.scatter(lons[thresh_only], lats[thresh_only], s=1.5, c='orange',
                   alpha=0.7, label=f'Thresh only ({int(thresh_only.sum()):,})')
    ax.set_title('Agreement')
    ax.legend(fontsize=9, markerscale=5)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # Bottom-right: P(fire) heatmap
    ax = axes[1, 1]
    sc = ax.scatter(lons, lats, s=0.5, c=probs, cmap='hot', vmin=0, vmax=1,
                    alpha=0.7)
    plt.colorbar(sc, ax=ax, label='P(fire)')
    ax.set_title('ML Fire Probability')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    plt.tight_layout()
    outname = f'plots/tune_prediction_map_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()

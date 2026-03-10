"""present_models.py - Compare two selected ML models per flight.

Creates one PNG per flight showing spatial fire predictions from both models
overlaid on an NDVI background (daytime) or T4 background (nighttime):
  Red circles    = Conservative model (Run #17, lowest daytime FP)
  Purple circles = Best Overall model (Run #156, lowest error rate)
  Light blue     = Both models predict fire

Usage:
    python present_models.py
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib import (
    group_files_by_flight, compute_grid_extent, build_pixel_table,
    build_mosaic,
)
from lib.inference import load_model, predict
from lib.features import build_location_features

NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]

# Model checkpoints
CONSERVATIVE_PATH = 'checkpoint/fire_detector_mlp_run_17.pt'
BEST_OVERALL_PATH = 'checkpoint/fire_detector_mlp_run_156.pt'

GROUND_TRUTH_FLIGHT = '24-801-03'

# Colors
COLOR_CONSERVATIVE = 'red'
COLOR_BEST = 'purple'
COLOR_BOTH = 'lightskyblue'


def load_flight_data() -> tuple[FlightFeatures, dict[str, dict]]:
    """Load HDF data, compute per-location features AND build mosaics."""
    flights = group_files_by_flight()
    flight_features: FlightFeatures = {}
    flight_mosaics: dict[str, dict] = {}

    for fnum, info in sorted(flights.items()):
        files = info['files']
        day_night = info['day_night']
        print(f'  Flight {fnum} ({len(files)} files, {day_night})...')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)

        # Build mosaic for background imagery
        mosaic = build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night)
        flight_mosaics[fnum] = mosaic

        # Build per-location features for ML predictions
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

    return flight_features, flight_mosaics


def compute_flight_metrics(
    preds: npt.NDArray[np.bool_],
    gt: npt.NDArray[np.bool_],
) -> dict[str, int]:
    """Compute TP, FP, FN, TN for one flight."""
    return {
        'TP': int((preds & gt).sum()),
        'FP': int((preds & ~gt).sum()),
        'FN': int((~preds & gt).sum()),
        'TN': int((~preds & ~gt).sum()),
    }


def plot_flight_comparison(
    fnum: str,
    comment: str,
    day_night: str,
    lats: NDArrayFloat,
    lons: NDArrayFloat,
    gt: npt.NDArray[np.bool_],
    cons_preds: npt.NDArray[np.bool_],
    best_preds: npt.NDArray[np.bool_],
    mosaic: dict[str, Any],
) -> None:
    """Create a single comparison plot for one flight."""
    cons_metrics = compute_flight_metrics(cons_preds, gt)
    best_metrics = compute_flight_metrics(best_preds, gt)

    # Classification masks
    both = cons_preds & best_preds
    cons_only = cons_preds & ~best_preds
    best_only = best_preds & ~cons_preds

    dn_label = 'Night' if day_night == 'N' else 'Day'
    is_preburn = fnum == GROUND_TRUTH_FLIGHT

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(
        f'Flight {fnum} — {comment}\n{dn_label}',
        fontsize=20, fontweight='bold', y=0.98)

    # Background: NDVI for daytime, T4 for nighttime, at 50% alpha
    lat_axis = mosaic['lat_axis']
    lon_axis = mosaic['lon_axis']
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    if day_night == 'D':
        bg_data = mosaic['NDVI']
        im_bg = ax.imshow(bg_data, extent=extent, aspect='equal',
                          cmap='RdYlGn', vmin=-0.1, vmax=0.5, alpha=0.5)
        cb = plt.colorbar(im_bg, ax=ax, fraction=0.03, pad=0.02, shrink=0.7)
        cb.set_label('NDVI', fontsize=12)
        cb.ax.tick_params(labelsize=10)
    else:
        bg_data = mosaic['T4']
        valid = bg_data[np.isfinite(bg_data)]
        vmin, vmax = (np.percentile(valid, [2, 98]) if len(valid) > 0
                      else (280, 320))
        im_bg = ax.imshow(bg_data, extent=extent, aspect='equal',
                          cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)
        cb = plt.colorbar(im_bg, ax=ax, fraction=0.03, pad=0.02, shrink=0.7)
        cb.set_label('T4 [K]', fontsize=12)
        cb.ax.tick_params(labelsize=10)

    # Light blue: both models agree on fire
    if both.any():
        ax.scatter(lons[both], lats[both], s=6, c=COLOR_BOTH,
                   edgecolors='steelblue', linewidths=0.3,
                   alpha=0.9, zorder=3,
                   label=f'Both ({int(both.sum()):,})')

    # Red: conservative only
    if cons_only.any():
        ax.scatter(lons[cons_only], lats[cons_only], s=6, c=COLOR_CONSERVATIVE,
                   edgecolors='darkred', linewidths=0.3,
                   alpha=0.9, zorder=4,
                   label=f'Conservative only ({int(cons_only.sum()):,})')

    # Purple: best overall only
    if best_only.any():
        ax.scatter(lons[best_only], lats[best_only], s=6, c=COLOR_BEST,
                   edgecolors='indigo', linewidths=0.3,
                   alpha=0.9, zorder=4,
                   label=f'Best Overall only ({int(best_only.sum()):,})')

    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    ax.tick_params(labelsize=12)

    # Legend at bottom
    ax.legend(
        fontsize=13, loc='lower center', bbox_to_anchor=(0.5, -0.12),
        ncol=3, markerscale=3, frameon=True, fancybox=True)

    # Metric summary boxes
    if is_preburn:
        cons_text = (
            f'Conservative (Run #17)\n'
            f'FP: {cons_metrics["FP"]:,}\n'
            f'(all detections are FP)')
        best_text = (
            f'Best Overall (Run #156)\n'
            f'FP: {best_metrics["FP"]:,}\n'
            f'(all detections are FP)')
    else:
        cons_text = (
            f'Conservative (Run #17)\n'
            f'TP: {cons_metrics["TP"]:,}  FP: {cons_metrics["FP"]:,}  FN: {cons_metrics["FN"]:,}')
        best_text = (
            f'Best Overall (Run #156)\n'
            f'TP: {best_metrics["TP"]:,}  FP: {best_metrics["FP"]:,}  FN: {best_metrics["FN"]:,}')

    # Red-outlined box (top-left)
    ax.text(0.02, 0.98, cons_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_CONSERVATIVE, linewidth=2.5, alpha=0.9))

    # Purple-outlined box (top-right)
    ax.text(0.98, 0.98, best_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_BEST, linewidth=2.5, alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    os.makedirs('plots', exist_ok=True)
    outname = f'plots/model_compare_{fnum.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()


def main() -> None:
    """Load both models, build features, and generate comparison plots."""
    print('Loading models...')
    cons_model, cons_scaler, cons_T_ign, cons_norm = load_model(CONSERVATIVE_PATH)
    best_model, best_scaler, best_T_ign, best_norm = load_model(BEST_OVERALL_PATH)
    print(f'  Conservative: {sum(p.numel() for p in cons_model.parameters()):,} params')
    print(f'  Best Overall: {sum(p.numel() for p in best_model.parameters()):,} params')

    print('\nBuilding per-location features and mosaics...')
    flight_features, flight_mosaics = load_flight_data()

    print('\nGenerating comparison plots...')
    for fnum in sorted(flight_features.keys()):
        d = flight_features[fnum]
        X = d['X']
        raw_y = d['y']
        lats, lons = d['lats'], d['lons']

        # Ground truth
        if fnum == GROUND_TRUTH_FLIGHT:
            gt = np.zeros_like(raw_y, dtype=bool)
        else:
            gt = raw_y > 0.5

        # Predictions from both models
        cons_preds, _ = predict(cons_model, cons_scaler, X,
                                T_ignition=cons_T_ign, normalization=cons_norm)
        best_preds, _ = predict(best_model, best_scaler, X,
                                T_ignition=best_T_ign, normalization=best_norm)

        print(f'\nFlight {fnum}:')
        print(f'  Conservative: {int(cons_preds.sum()):,} detections')
        print(f'  Best Overall: {int(best_preds.sum()):,} detections')

        plot_flight_comparison(
            fnum, d['comment'], d['day_night'],
            lats, lons, gt, cons_preds, best_preds,
            flight_mosaics[fnum])

    print('\nDone.')


if __name__ == '__main__':
    main()

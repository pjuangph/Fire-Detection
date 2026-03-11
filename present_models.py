"""present_models.py - Compare two ML models + baseline per flight.

Creates animated GIFs (one per flight) showing sweep-by-sweep buildup of
fire predictions overlaid on an NDVI background (daytime) or T4 (nighttime):
  Red circles    = Conservative model (Run #17, lowest daytime FP)
  Purple circles = Best Overall model (Run #156, lowest error rate)
  Green circles  = Both ML models predict fire
  Orange squares = Baseline threshold detector (detect_fire_simple)

Also generates final-frame PNGs.

Usage:
    python present_models.py              # GIFs + PNGs
    python present_models.py --png-only   # PNGs only (faster)
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v3 as iio

from lib import (
    group_files_by_flight, compute_grid_extent, build_pixel_table,
    build_mosaic,
)
from lib.io import process_file
from lib.fire import detect_fire_simple
from lib.inference import load_model, predict
from lib.features import build_location_features
from lib.constants import GRID_RES

NDArrayFloat = npt.NDArray[np.floating[Any]]
FlightFeatures = dict[str, dict[str, Any]]

# Model checkpoints
CONSERVATIVE_PATH = 'checkpoint/fire_detector_mlp_run_17.pt'
BEST_OVERALL_PATH = 'checkpoint/fire_detector_mlp_run_156.pt'

GROUND_TRUTH_FLIGHT = '24-801-03'

# Colors — chosen for maximum distinguishability
COLOR_CONSERVATIVE = '#e41a1c'   # bright red (circles)
COLOR_BEST = '#984ea3'           # purple (circles)
COLOR_BOTH = '#4daf4a'           # green (both ML agree)
COLOR_BASELINE = '#ff7f00'       # orange (squares)

FRAME_DIR = 'plots/model_compare_frames'
GIF_DIR = 'plots/gifs'


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


def render_frame(
    fnum: str,
    comment: str,
    day_night: str,
    lats: NDArrayFloat,
    lons: NDArrayFloat,
    baseline: npt.NDArray[np.bool_],
    cons_preds: npt.NDArray[np.bool_],
    best_preds: npt.NDArray[np.bool_],
    mosaic: dict[str, Any],
    sweep_num: int,
    n_sweeps: int,
    outpath: str,
    ndvi_mosaic: dict[str, Any] | None = None,
) -> None:
    """Render a single comparison frame (used for both GIFs and final PNGs).

    For nighttime flights, ndvi_mosaic supplies NDVI from a prior daytime
    flight to use as background instead of grayscale T4.
    """
    # Metrics: ML models compared against baseline threshold detector
    cons_metrics = compute_flight_metrics(cons_preds, baseline)
    best_metrics = compute_flight_metrics(best_preds, baseline)

    # Classification masks for ML models
    both_ml = cons_preds & best_preds
    cons_only = cons_preds & ~best_preds
    best_only = best_preds & ~cons_preds

    # Baseline-only: detected by baseline but neither ML model
    baseline_only = baseline & ~cons_preds & ~best_preds

    dn_label = 'Night' if day_night == 'N' else 'Day'
    is_preburn = fnum == GROUND_TRUTH_FLIGHT
    n_baseline = int(baseline.sum())

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle(
        f'Flight {fnum} — {comment}\n{dn_label}',
        fontsize=20, fontweight='bold', y=0.98)

    # Background: NDVI for daytime, T4 for nighttime, at 50% alpha
    lat_axis = mosaic['lat_axis']
    lon_axis = mosaic['lon_axis']
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    # Use NDVI background for all flights; nighttime uses prior daytime NDVI
    if day_night == 'D':
        bg_ndvi = mosaic['NDVI']
        bg_extent = extent
    elif ndvi_mosaic is not None:
        bg_ndvi = ndvi_mosaic['NDVI']
        ndvi_lat = ndvi_mosaic['lat_axis']
        ndvi_lon = ndvi_mosaic['lon_axis']
        bg_extent = [ndvi_lon[0], ndvi_lon[-1], ndvi_lat[-1], ndvi_lat[0]]
    else:
        bg_ndvi = None

    if bg_ndvi is not None:
        im_bg = ax.imshow(bg_ndvi, extent=bg_extent, aspect='equal',
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

    # Baseline-only: orange squares (lowest z-order of markers)
    if baseline_only.any():
        ax.scatter(lons[baseline_only], lats[baseline_only],
                   s=10, c=COLOR_BASELINE, marker='s',
                   edgecolors='#cc6600', linewidths=0.3,
                   alpha=0.8, zorder=2,
                   label=f'Baseline only ({int(baseline_only.sum()):,})')

    # Green: both ML models agree on fire
    if both_ml.any():
        ax.scatter(lons[both_ml], lats[both_ml], s=6, c=COLOR_BOTH,
                   edgecolors='darkgreen', linewidths=0.3,
                   alpha=0.9, zorder=3,
                   label=f'Both ML ({int(both_ml.sum()):,})')

    # Red circles: conservative only
    if cons_only.any():
        ax.scatter(lons[cons_only], lats[cons_only], s=6, c=COLOR_CONSERVATIVE,
                   edgecolors='darkred', linewidths=0.3,
                   alpha=0.9, zorder=4,
                   label=f'Conservative only ({int(cons_only.sum()):,})')

    # Purple circles: best overall only
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
        fontsize=12, loc='lower center', bbox_to_anchor=(0.5, -0.12),
        ncol=4, markerscale=3, frameon=True, fancybox=True)

    # Metric summary boxes — top row
    n_cons = int(cons_preds.sum())
    n_best = int(best_preds.sum())
    if is_preburn:
        cons_text = (
            f'Conservative (Run #17)\n'
            f'FP: {n_cons:,}  (no real fire)')
        baseline_text = (
            f'Baseline (Threshold)\n'
            f'FP: {n_baseline:,}  (no real fire)')
        best_text = (
            f'Best Overall (Run #156)\n'
            f'FP: {n_best:,}  (no real fire)')
    else:
        cons_text = (
            f'Conservative (Run #17)\n'
            f'TP: {cons_metrics["TP"]:,}  FP: {cons_metrics["FP"]:,}\n'
            f'FN: {cons_metrics["FN"]:,}')
        baseline_text = (
            f'Baseline (Threshold)\n'
            f'Detections: {n_baseline:,}')
        best_text = (
            f'Best Overall (Run #156)\n'
            f'TP: {best_metrics["TP"]:,}  FP: {best_metrics["FP"]:,}\n'
            f'FN: {best_metrics["FN"]:,}')

    # Red-outlined box (top-left) — Conservative
    ax.text(0.02, 0.98, cons_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_CONSERVATIVE, linewidth=2.5, alpha=0.9))

    # Orange-outlined box (top-center) — Baseline
    ax.text(0.50, 0.98, baseline_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_BASELINE, linewidth=2.5, alpha=0.9))

    # Purple-outlined box (top-right) — Best Overall
    ax.text(0.98, 0.98, best_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=COLOR_BEST, linewidth=2.5, alpha=0.9))

    # Sweep counter (bottom-left)
    ax.text(0.02, 0.02, f'Sweep {sweep_num}/{n_sweeps}',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='bottom', horizontalalignment='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', linewidth=1.5, alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


def build_incremental_pixel_table(
    filepath: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    day_night: str,
    flight_num: str,
) -> pd.DataFrame:
    """Build pixel table from a single HDF file."""
    nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
    ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))
    T4_thresh = 310.0 if day_night == 'N' else 325.0

    pf = process_file(filepath)
    T4, T11, SWIR = pf['T4'], pf['T11'], pf['SWIR']
    lat, lon = pf['lat'], pf['lon']
    NDVI = pf['NDVI'] if day_night != 'N' else np.full_like(T4, np.nan)

    fire = detect_fire_simple(T4, T11, T4_thresh=T4_thresh)
    dT = T4 - T11

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(T4)
    row_idx = ((lat_max - lat) / GRID_RES).astype(np.int32)
    col_idx = ((lon - lon_min) / GRID_RES).astype(np.int32)
    in_bounds = (
        valid &
        (row_idx >= 0) & (row_idx < nrows) &
        (col_idx >= 0) & (col_idx < ncols)
    )

    r = row_idx[in_bounds]
    c = col_idx[in_bounds]
    grid_lat = lat_max - r * GRID_RES
    grid_lon = lon_min + c * GRID_RES

    return pd.DataFrame({
        'flight': flight_num,
        'file': os.path.basename(filepath),
        'lat': grid_lat,
        'lon': grid_lon,
        'T4': T4[in_bounds],
        'T11': T11[in_bounds],
        'dT': dT[in_bounds],
        'SWIR': SWIR[in_bounds],
        'Red': pf['Red'][in_bounds],
        'NIR': pf['NIR'][in_bounds],
        'NDVI': NDVI[in_bounds],
        'fire': fire[in_bounds],
    })


def main() -> None:
    """Load both models, build features sweep-by-sweep, generate GIFs + PNGs."""
    parser = argparse.ArgumentParser(description='Compare ML models + baseline')
    parser.add_argument('--png-only', action='store_true',
                        help='Only generate final PNGs (no GIFs)')
    parser.add_argument('--fps', type=int, default=3,
                        help='GIF frames per second (default: 3)')
    args = parser.parse_args()

    print('Loading models...')
    cons_model, cons_scaler, cons_T_ign, cons_norm = load_model(CONSERVATIVE_PATH)
    best_model, best_scaler, best_T_ign, best_norm = load_model(BEST_OVERALL_PATH)
    print(f'  Conservative: {sum(p.numel() for p in cons_model.parameters()):,} params')
    print(f'  Best Overall: {sum(p.numel() for p in best_model.parameters()):,} params')

    flights = group_files_by_flight()

    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(GIF_DIR, exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Track most recent daytime mosaic for nighttime NDVI background
    prior_day_mosaic: dict[str, Any] | None = None

    for fnum, info in sorted(flights.items()):
        files = info['files']
        day_night = info['day_night']
        comment = info['comment']
        n_sweeps = len(files)
        flight_clean = fnum.replace('-', '')

        print(f'\nFlight {fnum} ({n_sweeps} files, {day_night}) — {comment}')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)

        # Build full mosaic for background (computed once)
        print('  Building mosaic...')
        mosaic = build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night)

        # Remember daytime mosaics for nighttime NDVI backgrounds
        if day_night == 'D':
            prior_day_mosaic = mosaic

        is_preburn = fnum == GROUND_TRUTH_FLIGHT

        # Accumulate pixel data sweep by sweep
        cumulative_dfs: list[pd.DataFrame] = []
        frame_paths: list[str] = []

        for si, filepath in enumerate(files):
            sweep_num = si + 1
            fname = os.path.basename(filepath)

            # Add this sweep's pixels to cumulative table
            sweep_df = build_incremental_pixel_table(
                filepath, lat_min, lat_max, lon_min, lon_max,
                day_night, fnum)
            cumulative_dfs.append(sweep_df)

            # Build features from all sweeps so far
            cumulative_df = pd.concat(cumulative_dfs, ignore_index=True)
            X, y, lats, lons = build_location_features(cumulative_df)

            # Baseline threshold detections (includes FP on pre-burn)
            baseline = y > 0.5

            # ML predictions
            cons_preds, _ = predict(cons_model, cons_scaler, X,
                                    T_ignition=cons_T_ign, normalization=cons_norm)
            best_preds, _ = predict(best_model, best_scaler, X,
                                    T_ignition=best_T_ign, normalization=best_norm)

            n_bl = int(baseline.sum())
            n_cons = int(cons_preds.sum())
            n_best = int(best_preds.sum())

            frame_path = os.path.join(
                FRAME_DIR, f'compare-{flight_clean}-{sweep_num:03d}.png')
            frame_paths.append(frame_path)

            if not args.png_only or sweep_num == n_sweeps:
                render_frame(
                    fnum, comment, day_night,
                    lats, lons, baseline, cons_preds, best_preds,
                    mosaic, sweep_num, n_sweeps, frame_path,
                    ndvi_mosaic=prior_day_mosaic if day_night == 'N' else None)

            print(f'  [{sweep_num:2d}/{n_sweeps}] {fname} — '
                  f'BL:{n_bl:,} Cons:{n_cons:,} Best:{n_best:,}')

        # Copy final frame as the static PNG
        final_png = f'plots/model_compare_{flight_clean}.png'
        if frame_paths:
            import shutil
            shutil.copy2(frame_paths[-1], final_png)
            print(f'  Final PNG: {final_png}')

        # Assemble GIF
        if not args.png_only and len(frame_paths) > 1:
            gif_path = os.path.join(GIF_DIR, f'model_compare_{flight_clean}.gif')
            images = [iio.imread(f) for f in frame_paths]

            # Pad to consistent size
            max_h = max(img.shape[0] for img in images)
            max_w = max(img.shape[1] for img in images)
            channels = images[0].shape[2] if images[0].ndim == 3 else 1

            padded = []
            for img in images:
                if img.shape[0] == max_h and img.shape[1] == max_w:
                    padded.append(img)
                else:
                    p = np.full((max_h, max_w, channels), 255, dtype=np.uint8)
                    p[:img.shape[0], :img.shape[1]] = img
                    padded.append(p)

            duration_ms = 1000 // args.fps
            durations = [duration_ms] * len(padded)
            durations[-1] = duration_ms * 4  # hold last frame

            iio.imwrite(gif_path, padded, duration=durations, loop=0)
            size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            print(f'  GIF: {gif_path} ({size_mb:.1f} MB)')

    print('\nDone.')


if __name__ == '__main__':
    main()

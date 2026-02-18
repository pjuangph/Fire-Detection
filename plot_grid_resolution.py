#!/usr/bin/env python3
"""Generate intuitive before/after grid resolution comparison plots.

Shows real vegetation (NDVI) and fire (T4) imagery at native ~8 m resolution
vs the 28 m gridded resolution, so the viewer can SEE:
  - How vegetation detail gets blockier (like a lower-resolution photo)
  - How the fire signal (T4 max) still carries through the grid

Usage:
    python plot_grid_resolution.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lib.constants import GRID_RES
from lib.io import process_file, group_files_by_flight


def grid_channel(values, lat, lon, grid_res, lat_min, lat_max, lon_min, lon_max,
                 agg='last'):
    """Grid raw pixels onto a regular lat/lon grid.

    agg: 'last' (last-write), 'max' (running maximum), 'mean'
    """
    nrows = int(np.ceil((lat_max - lat_min) / grid_res))
    ncols = int(np.ceil((lon_max - lon_min) / grid_res))

    grid = np.full((nrows, ncols), np.nan, dtype=np.float64)
    count = np.zeros((nrows, ncols), dtype=np.int32)
    acc = np.zeros((nrows, ncols), dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    row_idx = ((lat_max - lat[valid]) / grid_res).astype(np.int32)
    col_idx = ((lon[valid] - lon_min) / grid_res).astype(np.int32)
    in_bounds = ((row_idx >= 0) & (row_idx < nrows) &
                 (col_idx >= 0) & (col_idx < ncols))
    r = row_idx[in_bounds]
    c = col_idx[in_bounds]
    v = values[valid][in_bounds]

    if agg == 'last':
        grid[r, c] = v
    elif agg == 'max':
        grid[r, c] = -np.inf
        np.maximum.at(grid, (r, c), v)
        grid[grid == -np.inf] = np.nan
    elif agg == 'mean':
        np.add.at(acc, (r, c), v)
        np.add.at(count, (r, c), 1)
        mask = count > 0
        grid[mask] = acc[mask] / count[mask]

    lat_axis = np.linspace(lat_max, lat_min, nrows)
    lon_axis = np.linspace(lon_min, lon_max, ncols)
    return grid, lat_axis, lon_axis


def plot_all(filepath: str):
    """Create before/after plots using real T4, NDVI, and fire data."""
    print(f'Loading {os.path.basename(filepath)}...')
    d = process_file(filepath)
    T4 = d['T4']
    ndvi = d['NDVI']
    lat = d['lat']
    lon = d['lon']

    valid = np.isfinite(lat) & np.isfinite(lon)
    lat_min, lat_max = np.nanmin(lat[valid]), np.nanmax(lat[valid])
    lon_min, lon_max = np.nanmin(lon[valid]), np.nanmax(lon[valid])

    # Grid all channels
    ndvi_grid, lat_ax, lon_ax = grid_channel(
        ndvi, lat, lon, GRID_RES, lat_min, lat_max, lon_min, lon_max, agg='mean')
    t4_grid_max, _, _ = grid_channel(
        T4, lat, lon, GRID_RES, lat_min, lat_max, lon_min, lon_max, agg='max')
    t4_grid_last, _, _ = grid_channel(
        T4, lat, lon, GRID_RES, lat_min, lat_max, lon_min, lon_max, agg='last')

    extent = [lon_min, lon_max, lat_min, lat_max]
    os.makedirs('plots', exist_ok=True)

    # Get T4 range for consistent colorbars
    t4_valid = T4[np.isfinite(T4)]
    t4_lo = np.percentile(t4_valid, 1)
    t4_hi = max(np.percentile(t4_valid, 99), 350)  # ensure fire shows up

    # ── Figure 1: 2×2 full-scene NDVI + T4 ────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 12))
    fig.suptitle('What Happens When We Grid: Native 8 m Pixels  \u2192  28 m Grid Cells',
                 fontsize=28, fontweight='bold', y=0.98)

    # Top-left: Raw NDVI
    ax = axes[0, 0]
    v = np.isfinite(ndvi) & valid
    sc = ax.scatter(lon[v], lat[v], c=ndvi[v], s=0.15,
                    cmap='RdYlGn', vmin=-0.2, vmax=0.8, alpha=0.9,
                    rasterized=True)
    ax.set_title('Raw Vegetation (NDVI)  —  ~8 m pixels',
                 fontsize=20, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=18)
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('NDVI  (green = healthy vegetation)', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    ax.text(0.02, 0.02, 'Sharp detail — individual trees, rocks, bare soil',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Top-right: Gridded NDVI
    ax = axes[0, 1]
    im = ax.imshow(ndvi_grid, extent=extent, aspect='auto',
                   cmap='RdYlGn', vmin=-0.2, vmax=0.8,
                   interpolation='nearest', origin='upper')
    ax.set_title('Gridded Vegetation (NDVI)  —  28 m cells  (3× coarser)',
                 fontsize=20, fontweight='bold')
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('NDVI  (mean per cell)', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    ax.text(0.02, 0.02, 'Same scene, slightly blockier — like a lower-res photo\n'
            'But now we can stack multiple flight passes!',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Bottom-left: Raw T4
    ax = axes[1, 0]
    v = np.isfinite(T4) & valid
    sc = ax.scatter(lon[v], lat[v], c=T4[v], s=0.15,
                    cmap='hot', vmin=t4_lo, vmax=t4_hi, alpha=0.9,
                    rasterized=True)
    ax.set_title('Raw Fire Channel (T4, 3.9 μm)  —  ~8 m pixels',
                 fontsize=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('T4 Brightness Temperature (K)', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    n_fire_raw = int((T4[v] > 340).sum())
    ax.text(0.02, 0.02, f'Bright spots = hot surfaces  ({n_fire_raw:,} pixels > 340 K)',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Bottom-right: Gridded T4 MAX
    ax = axes[1, 1]
    im = ax.imshow(t4_grid_max, extent=extent, aspect='auto',
                   cmap='hot', vmin=t4_lo, vmax=t4_hi,
                   interpolation='nearest', origin='upper')
    ax.set_title('Gridded T4 (max per cell)  —  28 m cells',
                 fontsize=20, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('T4 max per cell (K)  —  hottest pixel wins!', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    ax.text(0.02, 0.02, 'Fire signal PRESERVED  →  T4_max keeps\n'
            'the hottest reading, fire never gets averaged away',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = 'plots/grid_resolution_comparison.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()

    # ── Figure 2: Zoomed into a region with fire or interesting veg ───
    # Find the hottest area
    fire_mask = np.isfinite(T4) & valid & (T4 > 330)
    if fire_mask.sum() > 50:
        center_lat = np.median(lat[fire_mask])
        center_lon = np.median(lon[fire_mask])
    else:
        center_lat = np.nanmean(lat)
        center_lon = np.nanmean(lon)

    zoom = 0.012  # ~1.3 km
    zoom_lat = (center_lat - zoom, center_lat + zoom)
    zoom_lon = (center_lon - zoom, center_lon + zoom)

    fig2, axes2 = plt.subplots(1, 3, figsize=(26, 9))
    fig2.suptitle('Zoomed In: How Gridding Affects What You See',
                  fontsize=28, fontweight='bold', y=0.98)

    # Left: Raw NDVI zoomed
    ax = axes2[0]
    v = (np.isfinite(ndvi) & valid &
         (lat > zoom_lat[0]) & (lat < zoom_lat[1]) &
         (lon > zoom_lon[0]) & (lon < zoom_lon[1]))
    sc = ax.scatter(lon[v], lat[v], c=ndvi[v], s=4,
                    cmap='RdYlGn', vmin=-0.2, vmax=0.8, alpha=0.9)
    ax.set_xlim(zoom_lon)
    ax.set_ylim(zoom_lat)
    ax.set_title('Raw NDVI  —  see individual trees and soil',
                 fontsize=22, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('NDVI', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    n_raw = int(v.sum())
    ax.text(0.02, 0.02, f'{n_raw:,} raw pixels in view',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Middle: Gridded NDVI zoomed
    ax = axes2[1]
    r0 = max(0, int((lat_ax[0] - zoom_lat[1]) / GRID_RES))
    r1 = min(ndvi_grid.shape[0], int((lat_ax[0] - zoom_lat[0]) / GRID_RES))
    c0 = max(0, int((zoom_lon[0] - lon_ax[0]) / GRID_RES))
    c1 = min(ndvi_grid.shape[1], int((zoom_lon[1] - lon_ax[0]) / GRID_RES))
    zoom_extent = [zoom_lon[0], zoom_lon[1], zoom_lat[0], zoom_lat[1]]

    im = ax.imshow(ndvi_grid[r0:r1, c0:c1], extent=zoom_extent, aspect='auto',
                   cmap='RdYlGn', vmin=-0.2, vmax=0.8,
                   interpolation='nearest', origin='upper')
    ax.set_title('Gridded NDVI  —  blockier, like a low-res photo',
                 fontsize=22, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('NDVI (mean)', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    n_cells = int(np.isfinite(ndvi_grid[r0:r1, c0:c1]).sum())
    ax.text(0.02, 0.02, f'{n_cells:,} grid cells\n'
            f'Each cell averages ~{max(1, n_raw // max(1, n_cells))} raw pixels',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Right: Gridded T4_max zoomed
    ax = axes2[2]
    im = ax.imshow(t4_grid_max[r0:r1, c0:c1], extent=zoom_extent, aspect='auto',
                   cmap='hot', vmin=t4_lo, vmax=t4_hi,
                   interpolation='nearest', origin='upper')
    ax.set_title('Gridded T4 max  —  fire signal carries through!',
                 fontsize=22, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.tick_params(labelsize=14)
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('T4 max (K)  —  hottest pixel per cell', fontsize=18)
    cb.ax.tick_params(labelsize=14)
    ax.text(0.02, 0.02, 'T4_max = keep the hottest reading\n'
            '→ fire is never averaged away by cooler neighbors',
            transform=ax.transAxes, fontsize=18, color='white', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = 'plots/grid_resolution_zoom.png'
    plt.savefig(out2, dpi=200, bbox_inches='tight')
    print(f'Saved {out2}')
    plt.close()


def main():
    flights = group_files_by_flight()

    # Use a later file from flight 06 (daytime smoldering) — has both NDVI and fire
    target = None
    for flight_num, info in sorted(flights.items()):
        if '06' in flight_num:
            files = info['files']
            # Pick a file near the middle so fire has developed
            target = files[len(files) // 2]
            break
    if target is None:
        for flight_num, info in sorted(flights.items()):
            target = info['files'][0]
            break
    if target is None:
        print('No HDF files found.')
        sys.exit(1)

    print(f'Using: {target}')
    plot_all(target)


if __name__ == '__main__':
    main()

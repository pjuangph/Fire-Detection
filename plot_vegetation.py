"""plot_vegetation.py - Vegetation (NDVI) mapping with fire overlay.

Creates one PNG per daytime flight with a 2x2 layout:
    Top-left:     NDVI map (green-brown colormap)
    Top-right:    NDVI with fire overlay (red scatter on NDVI background)
    Bottom-left:  Red band radiance (0.654 μm)
    Bottom-right: NIR band radiance (0.866 μm)

Nighttime flights are skipped (NDVI requires solar illumination).

Usage:
    python plot_vegetation.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lib import (
    group_files_by_flight, compute_grid_extent, build_mosaic,
)


def plot_vegetation_flight(mosaic, flight_num, comment, n_files):
    """Create a 2x2 vegetation analysis figure for one daytime flight.

    Args:
        mosaic: dict from build_mosaic().
        flight_num: flight identifier string.
        comment: flight comment from HDF metadata.
        n_files: number of flight line files.
    """
    if mosaic['day_night'] == 'N':
        print(f'  Skipping {flight_num} (nighttime, no NDVI)')
        return

    grid_NDVI = mosaic['NDVI']
    grid_Red = mosaic['Red']
    grid_NIR = mosaic['NIR']
    grid_fire = mosaic['fire']
    lat_axis = mosaic['lat_axis']
    lon_axis = mosaic['lon_axis']

    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Vegetation Analysis — Flight {flight_num}\n{comment}, '
                 f'{n_files} flight lines',
                 fontsize=14, fontweight='bold')

    # --- Top-left: NDVI map ---
    ax = axes[0, 0]
    im = ax.imshow(grid_NDVI, extent=extent, aspect='equal',
                   cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    ax.set_title('NDVI (Vegetation Index)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='NDVI')

    # NDVI stats
    valid_ndvi = grid_NDVI[np.isfinite(grid_NDVI)]
    if len(valid_ndvi) > 0:
        veg_count = np.sum(valid_ndvi > 0.3)
        bare_count = np.sum(valid_ndvi <= 0.2)
        ax.text(0.02, 0.02,
                f'Vegetation (>0.3): {veg_count:,}\n'
                f'Bare soil (≤0.2): {bare_count:,}\n'
                f'Mean NDVI: {np.nanmean(valid_ndvi):.3f}',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
                family='monospace')

    # --- Top-right: NDVI + fire overlay ---
    ax = axes[0, 1]
    ax.imshow(grid_NDVI, extent=extent, aspect='equal',
              cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    fire_count = np.sum(grid_fire)
    if fire_count > 0:
        fire_rows, fire_cols = np.where(grid_fire)
        fire_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * fire_rows / (len(lat_axis) - 1)
        fire_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * fire_cols / (len(lon_axis) - 1)
        ax.scatter(fire_lons, fire_lats, s=0.3, c='red', alpha=0.7,
                   label=f'Fire ({fire_count:,} cells)')
        ax.legend(loc='upper right', markerscale=15, fontsize=9)

        # NDVI at fire locations
        fire_ndvi = grid_NDVI[grid_fire]
        fire_ndvi_valid = fire_ndvi[np.isfinite(fire_ndvi)]
        if len(fire_ndvi_valid) > 0:
            ax.text(0.02, 0.02,
                    f'Fire pixel NDVI:\n'
                    f'  Mean: {np.nanmean(fire_ndvi_valid):.3f}\n'
                    f'  Min:  {np.nanmin(fire_ndvi_valid):.3f}\n'
                    f'  Max:  {np.nanmax(fire_ndvi_valid):.3f}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
                    family='monospace')
    ax.set_title('Fire on Vegetation Map')

    # --- Bottom-left: Red band ---
    ax = axes[1, 0]
    valid_red = grid_Red[np.isfinite(grid_Red)]
    vmin_r, vmax_r = (np.percentile(valid_red, [2, 98]) if len(valid_red) > 0
                      else (0, 1))
    im = ax.imshow(grid_Red, extent=extent, aspect='equal',
                   cmap='Reds', vmin=vmin_r, vmax=vmax_r)
    ax.set_title('Red Band — 0.654 μm (absorbed by chlorophyll)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='W/m²/sr/μm')

    # --- Bottom-right: NIR band ---
    ax = axes[1, 1]
    valid_nir = grid_NIR[np.isfinite(grid_NIR)]
    vmin_n, vmax_n = (np.percentile(valid_nir, [2, 98]) if len(valid_nir) > 0
                      else (0, 1))
    im = ax.imshow(grid_NIR, extent=extent, aspect='equal',
                   cmap='Greens', vmin=vmin_n, vmax=vmax_n)
    ax.set_title('NIR Band — 0.866 μm (reflected by vegetation)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='W/m²/sr/μm')

    for ax in axes.flat:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    outname = f'plots/vegetation_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()


def main():
    flights = group_files_by_flight()

    print(f'Generating vegetation maps for {len(flights)} flights...\n')

    for fnum, info in sorted(flights.items()):
        files = info['files']
        day_night = info['day_night']

        if day_night == 'N':
            print(f'Flight {fnum}: nighttime, skipping NDVI.')
            continue

        print(f'Flight {fnum} ({len(files)} files, daytime)...')
        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        mosaic = build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night)
        plot_vegetation_flight(mosaic, fnum, info['comment'], len(files))
        print()

    print('Done.')


if __name__ == '__main__':
    main()

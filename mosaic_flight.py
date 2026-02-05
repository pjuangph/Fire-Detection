"""mosaic_flight.py - Assemble all flight lines into a single georeferenced mosaic.

For each flight, reads all HDF4 files, computes brightness temperature at 3.9 um
(fire channel) and 11.25 um (background), runs fire detection, and composites
everything onto a regular lat/lon grid. Later flight lines overwrite earlier ones,
so overlapping areas show the most recent observation.

Usage:
    python mosaic_flight.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lib import (
    group_files_by_flight, compute_grid_extent, build_mosaic, GRID_RES,
)


def plot_mosaic(grid_T4, grid_fire, lat_axis, lon_axis, flight_num, comment, n_files):
    """Plot a flight mosaic with fire overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Flight {flight_num} Mosaic — {comment}\n'
                 f'{n_files} flight lines composited', fontsize=13)

    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    # T4 brightness temperature
    valid_T4 = grid_T4[np.isfinite(grid_T4)]
    if len(valid_T4) > 0:
        vmin, vmax = np.percentile(valid_T4, [2, 98])
    else:
        vmin, vmax = 280, 320

    im = axes[0].imshow(grid_T4, extent=extent, aspect='equal',
                        cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'T4 Brightness Temp (3.9 μm)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[0], label='K', fraction=0.046, pad=0.04)

    # T4 with fire overlay
    axes[1].imshow(grid_T4, extent=extent, aspect='equal',
                   cmap='gray', vmin=vmin, vmax=vmax)

    fire_count = np.sum(grid_fire)
    if fire_count > 0:
        fire_rows, fire_cols = np.where(grid_fire)
        fire_lats = lat_axis[0] - fire_rows * GRID_RES  # lat decreases with row
        # Fix: lat_axis goes from lat_max to lat_min, row 0 = lat_max
        fire_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * fire_rows / (len(lat_axis) - 1)
        fire_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * fire_cols / (len(lon_axis) - 1)
        axes[1].scatter(fire_lons, fire_lats, s=0.2, c='red', alpha=0.7,
                        label=f'Fire ({fire_count:,} grid cells)')
        axes[1].legend(loc='upper right', markerscale=15, fontsize=9)

    axes[1].set_title(f'Fire Detection Overlay')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    outname = f'plots/mosaic_flight_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200)
    print(f'  Saved {outname}')
    plt.close()
    return outname


def main():
    flights = group_files_by_flight()

    print(f'Found {len(flights)} flights:\n')
    for fnum, info in sorted(flights.items()):
        print(f'  {fnum}: {len(info["files"])} lines — {info["comment"]}')
    print()

    for fnum, info in sorted(flights.items()):
        files = info['files']
        comment = info['comment']
        day_night = info['day_night']

        print(f'Processing flight {fnum} ({len(files)} files)...')
        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
        ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))
        print(f'  Grid: {nrows} x {ncols} ({GRID_RES*111000:.0f}m resolution)')
        print(f'  Extent: lat [{lat_min:.4f}, {lat_max:.4f}], '
              f'lon [{lon_min:.4f}, {lon_max:.4f}]')

        mosaic = build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night)

        valid_pix = np.sum(np.isfinite(mosaic['T4']))
        fire_pix = np.sum(mosaic['fire'])
        multi_pass_confirmed = np.sum(mosaic['fire'] & (mosaic['obs_count'] >= 2))
        single_pass_only = np.sum(mosaic['fire'] & (mosaic['obs_count'] < 2))
        coverage = 100.0 * valid_pix / (nrows * ncols)
        print(f'  Coverage: {coverage:.1f}% of grid filled')
        print(f'  Fire pixels: {fire_pix:,} ({multi_pass_confirmed:,} multi-pass, {single_pass_only:,} single-pass)')

        plot_mosaic(mosaic['T4'], mosaic['fire'], mosaic['lat_axis'], mosaic['lon_axis'],
                    fnum, comment, len(files))
        print()

    print('Done — all flight mosaics generated.')


if __name__ == '__main__':
    main()

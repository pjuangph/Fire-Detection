"""plot_burn_locations.py - Visualize burn locations, T4, T11, and detection error per flight.

Creates one PNG per flight with a 2x2 layout:
  Top-left:     Fire/false-positive locations on gray background
  Top-right:    T4 brightness temperature (3.9 μm fire channel)
  Bottom-left:  T11 brightness temperature (11.25 μm background channel)
  Bottom-right: T4 vs ΔT scatter with pre-burn false positives overlaid as error reference

The pre-burn flight (24-801-03) is processed first to establish false positive
characteristics. Those false positives are then shown in orange on every burn
flight's scatter plot so the error is always visible.

Usage:
    python plot_burn_locations.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mosaic_flight import (
    group_files_by_flight, compute_grid_extent, build_mosaic, GRID_RES
)


def plot_single_flight(grid_T4, grid_T11, grid_fire, lat_axis, lon_axis,
                       flight_num, comment, day_night, n_files,
                       fp_T4=None, fp_dT=None):
    """Create a 2x2 figure for one flight and save to plots/.

    Args:
        fp_T4, fp_dT: Pre-burn false positive T4 and ΔT arrays. When provided,
            these are overlaid on the scatter plot so the error is visible.
    """
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    # Color ranges
    valid_T4 = grid_T4[np.isfinite(grid_T4)]
    vmin_t4, vmax_t4 = (np.percentile(valid_T4, [2, 98]) if len(valid_T4) > 0
                        else (280, 320))
    valid_T11 = grid_T11[np.isfinite(grid_T11)]
    vmin_t11, vmax_t11 = (np.percentile(valid_T11, [2, 98]) if len(valid_T11) > 0
                          else (280, 320))

    dn_label = 'Night' if day_night == 'N' else 'Day'
    fire_count = np.sum(grid_fire)
    T4_thresh = 310.0 if day_night == 'N' else 325.0
    is_preburn = flight_num == '24-801-03'
    detection_label = 'False Positives' if is_preburn else 'Fire'
    detection_color = 'orange' if is_preburn else 'red'

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Flight {flight_num} — {comment}\n'
                 f'{dn_label}, {n_files} flight lines | T4 threshold: {T4_thresh:.0f} K',
                 fontsize=14, fontweight='bold')

    # --- Top-left: Fire/false-positive locations ---
    ax = axes[0, 0]
    ax.imshow(grid_T4, extent=extent, aspect='equal', cmap='gray', vmin=vmin_t4, vmax=vmax_t4)
    if fire_count > 0:
        fire_rows, fire_cols = np.where(grid_fire)
        fire_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * fire_rows / (len(lat_axis) - 1)
        fire_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * fire_cols / (len(lon_axis) - 1)
        ax.scatter(fire_lons, fire_lats, s=0.3, c=detection_color, alpha=0.7,
                   label=f'{detection_label} ({fire_count:,} cells)')
        ax.legend(loc='upper right', markerscale=15, fontsize=9)
    ax.set_title(f'{detection_label} Locations ({fire_count:,} cells)')
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Longitude')

    # --- Top-right: T4 brightness temperature ---
    ax = axes[0, 1]
    im_t4 = ax.imshow(grid_T4, extent=extent, aspect='equal',
                       cmap='inferno', vmin=vmin_t4, vmax=vmax_t4)
    ax.set_title('T4 — 3.9 μm (fire channel)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im_t4, ax=ax, fraction=0.046, pad=0.04, label='K')

    # --- Bottom-left: T11 brightness temperature ---
    ax = axes[1, 0]
    im_t11 = ax.imshow(grid_T11, extent=extent, aspect='equal',
                        cmap='inferno', vmin=vmin_t11, vmax=vmax_t11)
    ax.set_title('T11 — 11.25 μm (background channel)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im_t11, ax=ax, fraction=0.046, pad=0.04, label='K')

    # --- Bottom-right: T4 vs ΔT detection space scatter ---
    ax = axes[1, 1]

    # Extract pixel populations for this flight
    fire_T4_vals = grid_T4[grid_fire]
    fire_T11_vals = grid_T11[grid_fire]
    fire_dT_vals = fire_T4_vals - fire_T11_vals

    bg_mask = np.isfinite(grid_T4) & ~grid_fire
    bg_T4 = grid_T4[bg_mask]
    bg_dT = bg_T4 - grid_T11[bg_mask]

    rng = np.random.default_rng(42)

    # Background (gray)
    n_bg = min(5000, len(bg_T4))
    if n_bg > 0:
        idx = rng.choice(len(bg_T4), n_bg, replace=False)
        ax.scatter(bg_T4[idx], bg_dT[idx], s=1, c='gray', alpha=0.3, label='Background')

    # Pre-burn false positives (orange) — shown on EVERY plot as error reference
    if fp_T4 is not None and len(fp_T4) > 0:
        ax.scatter(fp_T4, fp_dT, s=12, c='orange', alpha=0.9, edgecolors='darkorange',
                   linewidths=0.5, zorder=4,
                   label=f'Pre-burn false positives ({len(fp_T4):,})')

    # This flight's detections (red for burn, orange for pre-burn)
    if len(fire_T4_vals) > 0:
        n_fire = min(5000, len(fire_T4_vals))
        idx = rng.choice(len(fire_T4_vals), n_fire, replace=False)
        if is_preburn:
            # Already shown as false positives above, skip duplicate
            pass
        else:
            ax.scatter(fire_T4_vals[idx], fire_dT_vals[idx], s=3, c='red', alpha=0.7,
                       zorder=3, label=f'Fire detections ({fire_count:,})')

    # Threshold lines
    ax.axvline(T4_thresh, color='blue', linestyle='--', linewidth=1, alpha=0.6,
               label=f'T4 thresh ({T4_thresh:.0f} K)')
    ax.axhline(10, color='green', linestyle='--', linewidth=1, alpha=0.6,
               label='ΔT thresh (10 K)')

    ax.set_xlabel('T4 [K]')
    ax.set_ylabel('ΔT = T4 − T11 [K]')
    if is_preburn:
        ax.set_title(f'Error: {fire_count:,} false positives in detection space')
        ax.set_xlim(250, 500)
        ax.set_ylim(-20, 200)
    else:
        ax.set_title('Detection Space (orange = pre-burn error, red = fire)')
        ax.set_xlim(250, 750)
        ax.set_ylim(-20, 400)

    ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    outname = f'plots/burn_locations_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'  Saved {outname}')
    plt.close()

    # Print stats
    valid_count = np.sum(np.isfinite(grid_T4))
    rate = 100.0 * fire_count / max(valid_count, 1)
    print(f'  {detection_label}: {fire_count:,} / {valid_count:,} pixels ({rate:.4f}%)')
    if fire_count > 0:
        print(f'    T4:  {np.nanmin(fire_T4_vals):.1f} – {np.nanmax(fire_T4_vals):.1f} K '
              f'(mean {np.nanmean(fire_T4_vals):.1f} K)')
        print(f'    T11: {np.nanmin(fire_T11_vals):.1f} – {np.nanmax(fire_T11_vals):.1f} K '
              f'(mean {np.nanmean(fire_T11_vals):.1f} K)')
        print(f'    ΔT:  {np.nanmin(fire_dT_vals):.1f} – {np.nanmax(fire_dT_vals):.1f} K '
              f'(mean {np.nanmean(fire_dT_vals):.1f} K)')

    return fire_T4_vals, fire_dT_vals


def main():
    flights = group_files_by_flight()

    print(f'Processing {len(flights)} flights (one PNG each)...\n')

    # --- Step 1: Process pre-burn flight first to get false positive reference ---
    fp_T4, fp_dT = None, None
    pre_fnum = '24-801-03'
    if pre_fnum in flights:
        info = flights[pre_fnum]
        files = info['files']
        print(f'Flight {pre_fnum} ({len(files)} files, {info["day_night"]}) — building error reference...')
        print(f'  Source files:')
        for fi in files:
            print(f'    {os.path.basename(fi)}')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        grid_T4, grid_T11, grid_fire, lat_axis, lon_axis = build_mosaic(
            files, lat_min, lat_max, lon_min, lon_max, info['day_night'])

        fp_T4, fp_dT = plot_single_flight(
            grid_T4, grid_T11, grid_fire, lat_axis, lon_axis,
            pre_fnum, info['comment'], info['day_night'], len(files),
            fp_T4=None, fp_dT=None)
        print(f'  → {len(fp_T4):,} false positive pixels will be shown on all burn flights\n')

    # --- Step 2: Process burn flights with false positive overlay ---
    for fnum, info in sorted(flights.items()):
        if fnum == pre_fnum:
            continue
        files = info['files']
        print(f'Flight {fnum} ({len(files)} files, {info["day_night"]})...')
        print(f'  Source files:')
        for fi in files:
            print(f'    {os.path.basename(fi)}')

        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        grid_T4, grid_T11, grid_fire, lat_axis, lon_axis = build_mosaic(
            files, lat_min, lat_max, lon_min, lon_max, info['day_night'])

        plot_single_flight(grid_T4, grid_T11, grid_fire, lat_axis, lon_axis,
                           fnum, info['comment'], info['day_night'], len(files),
                           fp_T4=fp_T4, fp_dT=fp_dT)
        print()

    print('Done.')


if __name__ == '__main__':
    main()

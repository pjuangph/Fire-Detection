"""Shared rendering and simulation logic for real-time fire detection."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

from lib import (
    compute_ndvi, process_sweep, get_fire_mask,
    detect_fire_zones, compute_cell_area_m2, format_area,
)


def render_frame(gs: dict[str, Any], fire_mask: np.ndarray,
                 frame_num: int, n_total: int,
                 flight_num: str, comment: str,
                 outdir: str, cell_area_m2: float,
                 detector_name: str = 'simple') -> str:
    """Render one frame of the real-time simulation as a PNG."""
    plt.rcParams.update({'font.size': 18})

    has_vnir = np.any(np.isfinite(gs['NIR']))
    lat_axis = gs['lat_axis']
    lon_axis = gs['lon_axis']
    extent = (lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0])

    fig, ax = plt.subplots(figsize=(16, 14))

    # --- Background layer ---
    if has_vnir:
        display_ndvi = compute_ndvi(gs['Red'], gs['NIR'])
        bg = ax.imshow(display_ndvi, extent=extent, aspect='equal',
                       cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        cbar = plt.colorbar(bg, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('NDVI', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    else:
        valid_T4 = gs['T4'][np.isfinite(gs['T4'])]
        vmin = np.percentile(valid_T4, 2) if len(valid_T4) > 0 else 280
        vmax = np.percentile(valid_T4, 98) if len(valid_T4) > 0 else 320
        bg = ax.imshow(gs['T4'], extent=extent, aspect='equal',
                       cmap='inferno', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(bg, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('T4 [K]', fontsize=18)
        cbar.ax.tick_params(labelsize=18)

    # --- Fire overlay ---
    fire_count = int(np.sum(fire_mask))
    n_zones = 0
    zone_sizes = []

    if fire_count > 0:
        labels, n_zones, zone_sizes = detect_fire_zones(fire_mask)

        fire_rows, fire_cols = np.where(fire_mask)
        fire_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * fire_rows / max(len(lat_axis) - 1, 1)
        fire_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * fire_cols / max(len(lon_axis) - 1, 1)
        ax.scatter(fire_lons, fire_lats, s=6, c='red', alpha=0.9,
                   edgecolors='black', linewidths=0.3, zorder=5)

        # Vegetation-confirmed fire pixels in orange
        if 'veg_confirmed' in gs:
            veg_fire = fire_mask & gs['veg_confirmed']
            veg_count = int(np.sum(veg_fire))
            if veg_count > 0:
                vf_rows, vf_cols = np.where(veg_fire)
                vf_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * vf_rows / max(len(lat_axis) - 1, 1)
                vf_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * vf_cols / max(len(lon_axis) - 1, 1)
                ax.scatter(vf_lons, vf_lats, s=6, c='#FF00FF', alpha=0.9,
                           edgecolors='black', linewidths=0.3, zorder=6)

        # Bounding boxes around top 3 fire zones with ID at lower-right
        for zone_id, size in zone_sizes[:3]:
            zone_mask = labels == zone_id
            zr, zc = np.where(zone_mask)

            r_min, r_max = zr.min(), zr.max()
            c_min, c_max = zc.min(), zc.max()
            lat_top = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * r_min / max(len(lat_axis) - 1, 1)
            lat_bot = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * r_max / max(len(lat_axis) - 1, 1)
            lon_left = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * c_min / max(len(lon_axis) - 1, 1)
            lon_right = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * c_max / max(len(lon_axis) - 1, 1)
            box_x = min(lon_left, lon_right)
            box_y = min(lat_top, lat_bot)
            box_w = abs(lon_right - lon_left)
            box_h = abs(lat_bot - lat_top)
            ax.add_patch(Rectangle(
                (box_x, box_y), box_w, box_h,
                linewidth=3, edgecolor='black', facecolor='none',
                linestyle='-', zorder=8))

            # Zone ID label at lower-right corner
            ax.text(
                box_x + box_w, box_y, f'Z{zone_id}',
                fontsize=11, color='#39FF14', fontweight='bold',
                ha='left', va='bottom', zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground='black')])

    # --- Stats box ---
    total_area = fire_count * cell_area_m2
    coverage = 100.0 * np.sum(np.isfinite(gs['T4'])) / (gs['nrows'] * gs['ncols'])
    dn_label = 'NDVI' if has_vnir else 'T4'

    veg_confirmed_count = 0
    if 'veg_confirmed' in gs:
        veg_confirmed_count = int(np.sum(fire_mask & gs['veg_confirmed']))

    stats_lines = [
        f'Sweep {frame_num}/{n_total} [{dn_label}]',
        f'Coverage: {coverage:.1f}%',
        f'Fire pixels: {fire_count:,}',
        f'Veg-confirmed: {veg_confirmed_count:,}',
        f'Total fire area: {format_area(total_area)}',
        f'Fire zones: {n_zones}',
    ]
    if zone_sizes:
        stats_lines.append('')
        for zone_id, size in zone_sizes[:3]:
            stats_lines.append(
                f'  Zone {zone_id}: {format_area(size * cell_area_m2)} '
                f'({size:,} px)')

    ax.text(0.02, 0.98, '\n'.join(stats_lines),
            transform=ax.transAxes, fontsize=18,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.92, edgecolor='gray'))

    # --- Title and labels ---
    ax.set_title(
        f'Real-Time Fire Detection \u2014 Flight {flight_num}\n{comment}',
        fontsize=18, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.tick_params(labelsize=18)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=10, label='Thermal fire'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF00FF',
               markeredgecolor='black', markersize=10, label='Veg-confirmed fire'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=18)

    plt.tight_layout()
    flight_clean = flight_num.replace('-', '')
    outpath = os.path.join(
        outdir, f'{detector_name}-{flight_clean}-{frame_num:03d}.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

    return outpath


def simulate_flight(flight_num: str, files: list[str],
                    comment: str, gs: dict[str, Any],
                    ml_model: Any = None,
                    detector_name: str = 'simple') -> None:
    """Simulate real-time fire detection for one flight."""
    print('=' * 60)
    print(f'Real-Time Fire Detection Simulation')
    print(f'Flight {flight_num}: {comment}')
    print(f'{len(files)} sweeps (day/night auto-detected per sweep)')
    print('=' * 60)

    outdir = 'plots/realtime'
    os.makedirs(outdir, exist_ok=True)
    flight_clean = flight_num.replace('-', '')

    print(f'\nSimulating {len(files)} sweeps \u2192 {outdir}/{detector_name}-{flight_clean}-*.png\n')

    pixel_rows = []
    cell_area = 0.0

    for i, filepath in enumerate(files):
        name = os.path.basename(filepath)
        n_new_fire, detected_dn = process_sweep(
            filepath, gs, pixel_rows, day_night='auto',
            flight_num=flight_num)

        lat_center = (gs['lat_min'] + gs['lat_max']) / 2
        cell_area = compute_cell_area_m2(lat_center)

        if i == 0:
            print(f'Initial grid: {gs["nrows"]} x {gs["ncols"]}, '
                  f'cell area: {cell_area:.0f} m\u00b2')

        if ml_model is not None:
            fire_mask = ml_model.predict_from_gs(gs)
        else:
            fire_mask = get_fire_mask(gs)
        fire_total = int(np.sum(fire_mask))

        render_frame(
            gs, fire_mask,
            i + 1, len(files), flight_num, comment,
            outdir, cell_area, detector_name=detector_name)

        dn_tag = 'day' if detected_dn == 'D' else 'night'
        print(f'  [{i+1:2d}/{len(files)}] {name} [{dn_tag}] \u2014 '
              f'new fire: {n_new_fire:,}, total: {fire_total:,}')

    # Final summary
    if ml_model is not None:
        fire_mask = ml_model.predict_from_gs(gs)
    else:
        fire_mask = get_fire_mask(gs)
    fire_total = int(np.sum(fire_mask))
    total_area = fire_total * cell_area

    print(f'\n{"=" * 60}')
    print(f'Simulation complete.')
    print(f'  Final fire pixels: {fire_total:,}')
    print(f'  Total fire area:   {format_area(total_area)}')
    if fire_total > 0:
        _, n_zones, zone_sizes = detect_fire_zones(fire_mask)
        print(f'  Fire zones:        {n_zones}')
        for zone_id, size in zone_sizes[:5]:
            print(f'    Zone {zone_id}: {format_area(size * cell_area)} '
                  f'({size:,} px)')
    print(f'\n  Output: {outdir}/{detector_name}-{flight_clean}-*.png ({len(files)} frames)')
    print(f'\n  To create GIF:')
    print(f'    convert -delay 50 -loop 0 '
          f'{outdir}/{detector_name}-{flight_clean}-*.png '
          f'{outdir}/{detector_name}-{flight_clean}.gif')

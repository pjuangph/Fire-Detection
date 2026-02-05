"""realtime_fire.py - Simulate real-time fire detection as the plane sweeps.

Processes each flight line one at a time, incrementally building a mosaic
and applying multi-pass fire detection. Outputs one PNG per sweep so
the evolution of fire detection can be animated as a GIF.

Day/night is auto-detected per sweep from VNIR radiance levels: if NIR
has meaningful signal, there's sunlight and NDVI vegetation is shown.
This is more robust than Solar Zenith Angle because it also handles
cloud cover (no solar signal even when geometrically daytime).

The operator view shows:
  - Green: vegetation (NDVI, when sunlight detected)
  - T4 thermal background (when no sunlight)
  - Red: predicted fire locations
  - Fire zone labels with area in m² or hectares
  - Running statistics (fire count, total area, zone breakdown)

Usage:
    python realtime_fire.py           # all flights
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lib import (
    group_files_by_flight, compute_grid_extent,
    compute_ndvi, init_grid_state, process_sweep, get_fire_mask,
    detect_fire_zones, compute_cell_area_m2, format_area,
)


# ── Rendering ─────────────────────────────────────────────────


def render_frame(gs, fire_mask, frame_num, n_total,
                 flight_num, comment, outdir, cell_area_m2):
    """Render one frame of the real-time simulation as a PNG.

    Background layer is chosen by checking if the grid has accumulated
    usable VNIR data (any finite NIR pixels), not by the last sweep's
    day/night flag. This way, daytime VNIR data persists even after
    nighttime sweeps are processed.
    """
    plt.rcParams.update({'font.size': 18})

    has_vnir = np.any(np.isfinite(gs['NIR']))
    lat_axis = gs['lat_axis']
    lon_axis = gs['lon_axis']
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    fig, ax = plt.subplots(figsize=(16, 14))

    # --- Background layer ---
    if has_vnir:
        display_ndvi = compute_ndvi(gs['Red'], gs['NIR'])
        bg = ax.imshow(display_ndvi, extent=extent, aspect='equal',
                       cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        cbar = plt.colorbar(bg, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('NDVI', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
    else:
        valid_T4 = gs['T4'][np.isfinite(gs['T4'])]
        vmin = np.percentile(valid_T4, 2) if len(valid_T4) > 0 else 280
        vmax = np.percentile(valid_T4, 98) if len(valid_T4) > 0 else 320
        bg = ax.imshow(gs['T4'], extent=extent, aspect='equal',
                       cmap='inferno', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(bg, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('T4 [K]', fontsize=18)
        cbar.ax.tick_params(labelsize=14)

    # --- Fire overlay ---
    fire_count = int(np.sum(fire_mask))
    n_zones = 0
    zone_sizes = []

    if fire_count > 0:
        labels, n_zones, zone_sizes = detect_fire_zones(fire_mask)

        fire_rows, fire_cols = np.where(fire_mask)
        fire_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * fire_rows / max(len(lat_axis) - 1, 1)
        fire_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * fire_cols / max(len(lon_axis) - 1, 1)
        ax.scatter(fire_lons, fire_lats, s=1.5, c='red', alpha=0.8, zorder=5)

        # Label top fire zones at their centroids
        for zone_id, size in zone_sizes[:10]:
            zone_mask = labels == zone_id
            zr, zc = np.where(zone_mask)
            cy = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * zr.mean() / max(len(lat_axis) - 1, 1)
            cx = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * zc.mean() / max(len(lon_axis) - 1, 1)

            area = size * cell_area_m2
            ax.annotate(
                f'Z{zone_id}\n{format_area(area)}',
                (cx, cy), fontsize=12, color='yellow', fontweight='bold',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='black', alpha=0.75, edgecolor='yellow'))

    # --- Stats box ---
    total_area = fire_count * cell_area_m2
    coverage = 100.0 * np.sum(np.isfinite(gs['T4'])) / (gs['nrows'] * gs['ncols'])
    dn_label = 'NDVI' if has_vnir else 'T4'

    stats_lines = [
        f'Sweep {frame_num}/{n_total} [{dn_label}]',
        f'Coverage: {coverage:.1f}%',
        f'Fire pixels: {fire_count:,}',
        f'Total fire area: {format_area(total_area)}',
        f'Fire zones: {n_zones}',
    ]
    if zone_sizes:
        stats_lines.append('')
        for zone_id, size in zone_sizes[:5]:
            stats_lines.append(
                f'  Zone {zone_id}: {format_area(size * cell_area_m2)} '
                f'({size:,} px)')

    ax.text(0.02, 0.98, '\n'.join(stats_lines),
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.92, edgecolor='gray'))

    # --- Title and labels ---
    ax.set_title(
        f'Real-Time Fire Detection \u2014 Flight {flight_num}\n{comment}',
        fontsize=18, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.tick_params(labelsize=14)

    plt.tight_layout()
    outpath = os.path.join(outdir, f'frame_{frame_num:03d}.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

    return outpath


# ── Simulation ────────────────────────────────────────────────


def simulate_flight(flight_num, files, comment):
    """Simulate real-time fire detection for one flight.

    Day/night is auto-detected per sweep from VNIR radiance.

    Args:
        flight_num: flight identifier (e.g. '24-801-04').
        files: list of HDF file paths for this flight.
        comment: flight comment from HDF metadata.
    """
    print('=' * 60)
    print(f'Real-Time Fire Detection Simulation')
    print(f'Flight {flight_num}: {comment}')
    print(f'{len(files)} sweeps (day/night auto-detected per sweep)')
    print('=' * 60)

    # Compute grid extent for the full flight
    lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
    gs = init_grid_state(lat_min, lat_max, lon_min, lon_max)

    lat_center = (lat_min + lat_max) / 2
    cell_area = compute_cell_area_m2(lat_center)
    print(f'Grid: {gs["nrows"]} x {gs["ncols"]}, cell area: {cell_area:.0f} m\u00b2')

    outdir = f'plots/realtime_{flight_num.replace("-", "")}'
    os.makedirs(outdir, exist_ok=True)

    print(f'\nSimulating {len(files)} sweeps \u2192 {outdir}/\n')

    pixel_rows = []

    for i, filepath in enumerate(files):
        name = os.path.basename(filepath)
        n_new_fire, detected_dn = process_sweep(
            filepath, gs, pixel_rows, day_night='auto', flight_num=flight_num)

        fire_mask = get_fire_mask(gs)
        fire_total = int(np.sum(fire_mask))

        render_frame(
            gs, fire_mask,
            i + 1, len(files), flight_num, comment,
            outdir, cell_area)

        dn_tag = 'day' if detected_dn == 'D' else 'night'
        print(f'  [{i+1:2d}/{len(files)}] {name} [{dn_tag}] \u2014 '
              f'new fire: {n_new_fire:,}, total: {fire_total:,}')

    # Final summary
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
    print(f'\n  Output: {outdir}/ ({len(files)} frames)')
    print(f'\n  To create GIF:')
    print(f'    convert -delay 50 -loop 0 '
          f'{outdir}/frame_*.png {outdir}/animation.gif')


def main():
    flights = group_files_by_flight()

    print(f'Scanned {len(flights)} flights:')
    for fnum, info in sorted(flights.items()):
        print(f'  {fnum}: {len(info["files"])} lines — {info["comment"]}')
    print()

    for fnum, info in sorted(flights.items()):
        simulate_flight(fnum, info['files'], info['comment'])
        print()

    print('All simulations complete.')


if __name__ == '__main__':
    main()

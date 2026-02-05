"""Mosaic gridding: build flight mosaics, incremental sweep processing."""

import os
import numpy as np

from lib.constants import GRID_RES
from lib.io import process_file
from lib.fire import detect_fire_simple
from lib.vegetation import compute_ndvi, has_sunlight


def build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night='D'):
    """Build a gridded mosaic from a list of flight line files.

    Files are processed in order (chronological), so later lines overwrite earlier
    ones in overlapping areas. A multi-pass consistency filter requires fire to be
    detected in >=2 passes for pixels observed multiple times, filtering out
    angle-dependent false positives (e.g. solar reflection).

    Returns dict with keys:
        T4, T11, SWIR: gridded brightness temp / radiance arrays
        Red, NIR: gridded VNIR radiance (NaN for nighttime flights)
        NDVI: computed from gridded Red/NIR (NaN for nighttime flights)
        fire: boolean fire mask after multi-pass filter
        lat_axis, lon_axis: coordinate axes
        fire_count, obs_count: per-cell detection and observation counts
        day_night: 'D' or 'N' flag
    """
    nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
    ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))

    grid_T4 = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid_T11 = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid_SWIR = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid_Red = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid_NIR = np.full((nrows, ncols), np.nan, dtype=np.float32)
    grid_fire_count = np.zeros((nrows, ncols), dtype=np.int32)
    grid_obs_count = np.zeros((nrows, ncols), dtype=np.int32)

    T4_thresh = 310.0 if day_night == 'N' else 325.0

    for i, filepath in enumerate(files):
        name = os.path.basename(filepath)
        print(f'  [{i+1}/{len(files)}] {name}')

        pf = process_file(filepath)
        T4, T11, SWIR = pf['T4'], pf['T11'], pf['SWIR']
        lat, lon = pf['lat'], pf['lon']
        fire = detect_fire_simple(T4, T11, T4_thresh=T4_thresh)

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
        grid_T4[r, c] = T4[in_bounds]
        grid_T11[r, c] = T11[in_bounds]
        grid_SWIR[r, c] = SWIR[in_bounds]
        grid_obs_count[r, c] += 1
        grid_fire_count[r, c] += fire[in_bounds].astype(np.int32)

        if day_night != 'N':
            grid_Red[r, c] = pf['Red'][in_bounds]
            grid_NIR[r, c] = pf['NIR'][in_bounds]

    # Multi-pass consistency filter
    multi_pass = grid_obs_count >= 2
    grid_fire = np.where(
        multi_pass,
        grid_fire_count >= 2,
        grid_fire_count >= 1
    )

    if day_night != 'N':
        grid_NDVI = compute_ndvi(grid_Red, grid_NIR)
    else:
        grid_NDVI = np.full((nrows, ncols), np.nan, dtype=np.float32)

    lat_axis = np.linspace(lat_max, lat_min, nrows)
    lon_axis = np.linspace(lon_min, lon_max, ncols)

    return {
        'T4': grid_T4, 'T11': grid_T11, 'SWIR': grid_SWIR,
        'Red': grid_Red, 'NIR': grid_NIR, 'NDVI': grid_NDVI,
        'fire': grid_fire,
        'lat_axis': lat_axis, 'lon_axis': lon_axis,
        'fire_count': grid_fire_count, 'obs_count': grid_obs_count,
        'day_night': day_night,
    }


def init_grid_state(lat_min, lat_max, lon_min, lon_max):
    """Initialize empty grid state for incremental mosaic building."""
    nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
    ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))

    return {
        'T4': np.full((nrows, ncols), np.nan, dtype=np.float32),
        'T11': np.full((nrows, ncols), np.nan, dtype=np.float32),
        'SWIR': np.full((nrows, ncols), np.nan, dtype=np.float32),
        'Red': np.full((nrows, ncols), np.nan, dtype=np.float32),
        'NIR': np.full((nrows, ncols), np.nan, dtype=np.float32),
        'fire_count': np.zeros((nrows, ncols), dtype=np.int32),
        'obs_count': np.zeros((nrows, ncols), dtype=np.int32),
        'nrows': nrows, 'ncols': ncols,
        'lat_min': lat_min, 'lat_max': lat_max,
        'lon_min': lon_min, 'lon_max': lon_max,
        'lat_axis': np.linspace(lat_max, lat_min, nrows),
        'lon_axis': np.linspace(lon_min, lon_max, ncols),
    }


def process_sweep(filepath, gs, pixel_rows, day_night='auto', flight_num=''):
    """Process one sweep file and update grid state in-place.

    Also appends per-pixel data to pixel_rows list for DataFrame
    construction later.

    Args:
        filepath: path to HDF file.
        gs: grid state dict (modified in-place).
        pixel_rows: list to append per-pixel dicts to (modified in-place).
        day_night: 'D', 'N', or 'auto'. When 'auto', checks VNIR radiance
                   to determine if the scene has sunlight (robust to cloud).
        flight_num: flight identifier string.

    Returns:
        (n_new_fire, day_night): fire pixel count and detected day/night flag.
    """
    pf = process_file(filepath)
    T4, T11, SWIR = pf['T4'], pf['T11'], pf['SWIR']
    lat, lon = pf['lat'], pf['lon']

    # Auto-detect day/night from VNIR radiance
    if day_night == 'auto':
        day_night = 'D' if has_sunlight(pf['Red'], pf['NIR']) else 'N'
    gs['day_night'] = day_night

    T4_thresh = 310.0 if day_night == 'N' else 325.0
    fire = detect_fire_simple(T4, T11, T4_thresh=T4_thresh)

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(T4)
    row_idx = ((gs['lat_max'] - lat) / GRID_RES).astype(np.int32)
    col_idx = ((lon - gs['lon_min']) / GRID_RES).astype(np.int32)
    in_bounds = (
        valid &
        (row_idx >= 0) & (row_idx < gs['nrows']) &
        (col_idx >= 0) & (col_idx < gs['ncols'])
    )

    r = row_idx[in_bounds]
    c = col_idx[in_bounds]
    gs['T4'][r, c] = T4[in_bounds]
    gs['T11'][r, c] = T11[in_bounds]
    gs['SWIR'][r, c] = SWIR[in_bounds]
    gs['obs_count'][r, c] += 1
    gs['fire_count'][r, c] += fire[in_bounds].astype(np.int32)

    # VNIR: keep the best-illuminated observation per pixel.
    # Every sweep contributes; per-pixel NIR comparison decides if applied.
    # Nighttime sweeps have near-zero NIR so they won't overwrite good
    # daytime data, but daytime sweeps always improve on nighttime data.
    new_nir = pf['NIR'][in_bounds]
    old_nir = gs['NIR'][r, c]
    better = np.isnan(old_nir) | (new_nir > old_nir)
    gs['Red'][r, c] = np.where(better, pf['Red'][in_bounds], gs['Red'][r, c])
    gs['NIR'][r, c] = np.where(better, new_nir, old_nir)

    # Use per-pixel NDVI where sunlight exists; nighttime pixels stay NaN
    NDVI = pf['NDVI']
    pixel_rows.append({
        'flight': flight_num,
        'file': os.path.basename(filepath),
        'lat': gs['lat_max'] - r * GRID_RES,
        'lon': gs['lon_min'] + c * GRID_RES,
        'T4': T4[in_bounds],
        'T11': T11[in_bounds],
        'dT': (T4 - T11)[in_bounds],
        'SWIR': SWIR[in_bounds],
        'NDVI': NDVI[in_bounds],
        'fire': fire[in_bounds],
    })

    return int(fire[in_bounds].sum()), day_night


def get_fire_mask(gs):
    """Apply multi-pass consistency filter to current grid state."""
    multi_pass = gs['obs_count'] >= 2
    return np.where(
        multi_pass,
        gs['fire_count'] >= 2,
        gs['fire_count'] >= 1,
    )

"""Mosaic gridding: build flight mosaics, incremental sweep processing."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from pyhdf.SD import SD, SDC

from lib.constants import GRID_RES, VEG_LOSS_THRESHOLD
from lib.io import process_file
from lib.fire import detect_fire_simple
from lib.vegetation import compute_ndvi, has_sunlight


def _file_extent(filepath: str) -> tuple[float, float, float, float]:
    """Get lat/lon bounding box of a single HDF file from corner attributes.

    Returns:
        (lat_min, lat_max, lon_min, lon_max) with 0.005° buffer.
    """
    sd = SD(filepath, SDC.READ)
    attrs = sd.attributes()
    lat_min = min(attrs['lat_LL'], attrs['lat_UL'])
    lat_max = max(attrs['lat_LR'], attrs['lat_UR'])
    lon_min = min(attrs['lon_UL'], attrs['lon_UR'])
    lon_max = max(attrs['lon_LL'], attrs['lon_LR'])
    sd.end()
    buf = 0.005
    return lat_min - buf, lat_max + buf, lon_min - buf, lon_max + buf


def _expand_grid(gs: dict[str, Any],
                 new_lat_min: float, new_lat_max: float,
                 new_lon_min: float, new_lon_max: float) -> None:
    """Expand all grid arrays in-place to cover new bounds.

    Allocates larger arrays and copies old data at the correct offset.
    """
    old_nrows, old_ncols = gs['nrows'], gs['ncols']
    new_nrows = int(np.ceil((new_lat_max - new_lat_min) / GRID_RES))
    new_ncols = int(np.ceil((new_lon_max - new_lon_min) / GRID_RES))

    # Offset of old grid within new grid
    row_off = round((new_lat_max - gs['lat_max']) / GRID_RES)
    col_off = round((gs['lon_min'] - new_lon_min) / GRID_RES)

    # Expand float arrays (NaN-filled)
    for key in ('T4', 'T11', 'SWIR', 'Red', 'NIR'):
        new_arr = np.full((new_nrows, new_ncols), np.nan, dtype=np.float32)
        new_arr[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs[key]
        gs[key] = new_arr

    # Expand int arrays (zero-filled)
    for key in ('fire_count', 'obs_count'):
        new_arr = np.zeros((new_nrows, new_ncols), dtype=np.int32)
        new_arr[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs[key]
        gs[key] = new_arr

    # Expand NDVI_baseline (NaN-filled)
    new_bl = np.full((new_nrows, new_ncols), np.nan, dtype=np.float32)
    new_bl[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs['NDVI_baseline']
    gs['NDVI_baseline'] = new_bl

    # Expand veg_confirmed (False-filled)
    new_vc = np.zeros((new_nrows, new_ncols), dtype=np.bool_)
    new_vc[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs['veg_confirmed']
    gs['veg_confirmed'] = new_vc

    # Expand running accumulators for ML features
    for key, fill, dtype in [
        ('T4_max', -np.inf, np.float32),
        ('dT_max', -np.inf, np.float32),
        ('SWIR_max', -np.inf, np.float32),
        ('NDVI_min', np.inf, np.float32),
    ]:
        new_arr = np.full((new_nrows, new_ncols), fill, dtype=dtype)
        new_arr[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs[key]
        gs[key] = new_arr

    for key in ('T4_sum', 'T11_sum', 'SWIR_sum', 'Red_sum', 'NIR_sum', 'NDVI_sum'):
        new_arr = np.zeros((new_nrows, new_ncols), dtype=np.float64)
        new_arr[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs[key]
        gs[key] = new_arr

    new_nobs = np.zeros((new_nrows, new_ncols), dtype=np.int32)
    new_nobs[row_off:row_off + old_nrows, col_off:col_off + old_ncols] = gs['NDVI_obs']
    gs['NDVI_obs'] = new_nobs

    gs['nrows'] = new_nrows
    gs['ncols'] = new_ncols
    gs['lat_min'] = new_lat_min
    gs['lat_max'] = new_lat_max
    gs['lon_min'] = new_lon_min
    gs['lon_max'] = new_lon_max
    gs['lat_axis'] = np.linspace(new_lat_max, new_lat_min, new_nrows)
    gs['lon_axis'] = np.linspace(new_lon_min, new_lon_max, new_ncols)

    print(f'  Grid expanded: {old_nrows}x{old_ncols} -> {new_nrows}x{new_ncols}')


def build_mosaic(files: list[str], lat_min: float, lat_max: float,
                 lon_min: float, lon_max: float,
                 day_night: str = 'D') -> dict[str, Any]:
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


def init_grid_state(lat_min: float | None = None, lat_max: float | None = None,
                    lon_min: float | None = None, lon_max: float | None = None) -> dict[str, Any]:
    """Initialize empty grid state for incremental mosaic building.

    When called with no arguments, returns an empty sentinel grid (0x0)
    that will be populated on the first call to process_sweep().
    """
    if lat_min is None:
        return {
            'T4': np.empty((0, 0), dtype=np.float32),
            'T11': np.empty((0, 0), dtype=np.float32),
            'SWIR': np.empty((0, 0), dtype=np.float32),
            'Red': np.empty((0, 0), dtype=np.float32),
            'NIR': np.empty((0, 0), dtype=np.float32),
            'fire_count': np.empty((0, 0), dtype=np.int32),
            'obs_count': np.empty((0, 0), dtype=np.int32),
            'NDVI_baseline': np.empty((0, 0), dtype=np.float32),
            'veg_confirmed': np.empty((0, 0), dtype=np.bool_),
            # Running accumulators for ML aggregate features
            'T4_max': np.empty((0, 0), dtype=np.float32),
            'T4_sum': np.empty((0, 0), dtype=np.float64),
            'T11_sum': np.empty((0, 0), dtype=np.float64),
            'dT_max': np.empty((0, 0), dtype=np.float32),
            'SWIR_max': np.empty((0, 0), dtype=np.float32),
            'SWIR_sum': np.empty((0, 0), dtype=np.float64),
            'Red_sum': np.empty((0, 0), dtype=np.float64),
            'NIR_sum': np.empty((0, 0), dtype=np.float64),
            'NDVI_min': np.empty((0, 0), dtype=np.float32),
            'NDVI_sum': np.empty((0, 0), dtype=np.float64),
            'NDVI_obs': np.empty((0, 0), dtype=np.int32),
            'nrows': 0, 'ncols': 0,
            'lat_min': None, 'lat_max': None,
            'lon_min': None, 'lon_max': None,
            'lat_axis': np.array([]), 'lon_axis': np.array([]),
        }

    assert lat_max is not None and lon_min is not None and lon_max is not None
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
        'NDVI_baseline': np.full((nrows, ncols), np.nan, dtype=np.float32),
        'veg_confirmed': np.zeros((nrows, ncols), dtype=np.bool_),
        # Running accumulators for ML aggregate features
        'T4_max': np.full((nrows, ncols), -np.inf, dtype=np.float32),
        'T4_sum': np.zeros((nrows, ncols), dtype=np.float64),
        'T11_sum': np.zeros((nrows, ncols), dtype=np.float64),
        'dT_max': np.full((nrows, ncols), -np.inf, dtype=np.float32),
        'SWIR_max': np.full((nrows, ncols), -np.inf, dtype=np.float32),
        'SWIR_sum': np.zeros((nrows, ncols), dtype=np.float64),
        'Red_sum': np.zeros((nrows, ncols), dtype=np.float64),
        'NIR_sum': np.zeros((nrows, ncols), dtype=np.float64),
        'NDVI_min': np.full((nrows, ncols), np.inf, dtype=np.float32),
        'NDVI_sum': np.zeros((nrows, ncols), dtype=np.float64),
        'NDVI_obs': np.zeros((nrows, ncols), dtype=np.int32),
        'nrows': nrows, 'ncols': ncols,
        'lat_min': lat_min, 'lat_max': lat_max,
        'lon_min': lon_min, 'lon_max': lon_max,
        'lat_axis': np.linspace(lat_max, lat_min, nrows),
        'lon_axis': np.linspace(lon_min, lon_max, ncols),
    }


def process_sweep(filepath: str, gs: dict[str, Any],
                  pixel_rows: list[dict[str, Any]],
                  day_night: str = 'auto',
                  flight_num: str = '',
                  fire_fn: Any = None) -> tuple[int, str]:
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
        fire_fn: optional callable(T4, T11, NDVI) -> bool mask. When
                 provided, replaces detect_fire_simple for fire detection.

    Returns:
        (n_new_fire, day_night): fire pixel count and detected day/night flag.
    """
    # Get this file's extent and init/expand grid as needed
    file_lat_min, file_lat_max, file_lon_min, file_lon_max = _file_extent(filepath)

    if gs['nrows'] == 0:
        # First sweep: initialize grid from this file's extent
        gs.update(init_grid_state(file_lat_min, file_lat_max,
                                  file_lon_min, file_lon_max))
    else:
        need_expand = (
            file_lat_min < gs['lat_min'] or file_lat_max > gs['lat_max'] or
            file_lon_min < gs['lon_min'] or file_lon_max > gs['lon_max']
        )
        if need_expand:
            _expand_grid(
                gs,
                min(gs['lat_min'], file_lat_min),
                max(gs['lat_max'], file_lat_max),
                min(gs['lon_min'], file_lon_min),
                max(gs['lon_max'], file_lon_max),
            )

    pf = process_file(filepath)
    T4, T11, SWIR = pf['T4'], pf['T11'], pf['SWIR']
    lat, lon = pf['lat'], pf['lon']

    # Auto-detect day/night from VNIR radiance
    if day_night == 'auto':
        day_night = 'D' if has_sunlight(pf['Red'], pf['NIR']) else 'N'
    gs['day_night'] = day_night

    if fire_fn is not None:
        fire = fire_fn(T4, T11, pf['NDVI'])
    else:
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

    # ── Running accumulators for ML aggregate features ──
    T4_ib = T4[in_bounds]
    T11_ib = T11[in_bounds]
    SWIR_ib = SWIR[in_bounds]
    gs['T4_max'][r, c] = np.maximum(gs['T4_max'][r, c], T4_ib)
    gs['T4_sum'][r, c] += T4_ib.astype(np.float64)
    gs['T11_sum'][r, c] += T11_ib.astype(np.float64)
    gs['dT_max'][r, c] = np.maximum(gs['dT_max'][r, c], T4_ib - T11_ib)
    gs['SWIR_max'][r, c] = np.maximum(gs['SWIR_max'][r, c], SWIR_ib)
    gs['SWIR_sum'][r, c] += SWIR_ib.astype(np.float64)
    Red_ib = pf['Red'][in_bounds]
    NIR_ib = pf['NIR'][in_bounds]
    gs['Red_sum'][r, c] += np.where(np.isfinite(Red_ib), Red_ib, 0.0).astype(np.float64)
    gs['NIR_sum'][r, c] += np.where(np.isfinite(NIR_ib), NIR_ib, 0.0).astype(np.float64)

    # ── Current sweep NDVI ──
    sweep_ndvi = pf['NDVI'][in_bounds]
    is_sunlit = np.isfinite(sweep_ndvi)

    # Daytime NDVI accumulators
    if np.any(is_sunlit):
        rs, cs, nv = r[is_sunlit], c[is_sunlit], sweep_ndvi[is_sunlit]
        gs['NDVI_min'][rs, cs] = np.minimum(gs['NDVI_min'][rs, cs], nv)
        gs['NDVI_sum'][rs, cs] += nv.astype(np.float64)
        gs['NDVI_obs'][rs, cs] += 1

    # ── Set NDVI baseline (first valid daytime obs, write-once) ──
    no_baseline = np.isnan(gs['NDVI_baseline'][r, c])
    gs['NDVI_baseline'][r, c] = np.where(
        no_baseline & is_sunlit, sweep_ndvi, gs['NDVI_baseline'][r, c])

    # ── Detect vegetation loss at fire pixels ──
    fire_here = fire[in_bounds]
    has_baseline = np.isfinite(gs['NDVI_baseline'][r, c])
    is_night = ~is_sunlit

    # Daytime: fire + NDVI drop confirms vegetation loss
    with np.errstate(invalid='ignore'):
        veg_drop = gs['NDVI_baseline'][r, c] - sweep_ndvi
        day_veg_lost = has_baseline & is_sunlit & (veg_drop >= VEG_LOSS_THRESHOLD)

    # Nighttime: fire alone confirms vegetation loss (fewer FP at night)
    night_veg_lost = has_baseline & is_night

    newly_confirmed = fire_here & (day_veg_lost | night_veg_lost)
    gs['veg_confirmed'][r, c] = gs['veg_confirmed'][r, c] | newly_confirmed

    # ── VNIR update: two paths ──
    # Non-fire pixels: keep best-illuminated (highest NIR)
    # Veg-confirmed pixels: take latest observation (show burn scar)
    new_nir = pf['NIR'][in_bounds]
    new_red = pf['Red'][in_bounds]
    old_nir = gs['NIR'][r, c]
    old_red = gs['Red'][r, c]

    best_illuminated = np.isnan(old_nir) | (new_nir > old_nir)
    take_latest = gs['veg_confirmed'][r, c]
    use_new = best_illuminated | take_latest

    gs['Red'][r, c] = np.where(use_new, new_red, old_red)
    gs['NIR'][r, c] = np.where(use_new, new_nir, old_nir)

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

    # ── Clear accumulators where fire has passed (veg gone, no current fire) ──
    cleared = gs['veg_confirmed'] & (gs['fire_count'] == 0)
    if np.any(cleared):
        gs['T4_max'][cleared] = -np.inf
        gs['T4_sum'][cleared] = 0
        gs['T11_sum'][cleared] = 0
        gs['dT_max'][cleared] = -np.inf
        gs['SWIR_max'][cleared] = -np.inf
        gs['SWIR_sum'][cleared] = 0
        gs['Red_sum'][cleared] = 0
        gs['NIR_sum'][cleared] = 0
        gs['NDVI_min'][cleared] = np.inf
        gs['NDVI_sum'][cleared] = 0
        gs['NDVI_obs'][cleared] = 0
        gs['obs_count'][cleared] = 0

    return int(fire[in_bounds].sum()), day_night


def get_fire_mask(gs: dict[str, Any]) -> np.ndarray:
    """Apply multi-pass consistency filter with vegetation confirmation.

    Standard rule: pixels observed >=2 times need fire in >=2 passes.
    Override: pixels with veg_confirmed=True are treated as fire even
    with only 1 thermal fire detection (vegetation loss is independent
    confirmation that the thermal signal was real fire).
    """
    multi_pass = gs['obs_count'] >= 2
    standard_fire = np.where(
        multi_pass,
        gs['fire_count'] >= 2,
        gs['fire_count'] >= 1,
    )
    veg_boost = gs['veg_confirmed'] & (gs['fire_count'] >= 1)
    return standard_fire | veg_boost

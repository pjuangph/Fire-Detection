"""Analysis utilities: pixel tables, location statistics, area computation."""

import os
import numpy as np

from lib.constants import GRID_RES
from lib.io import process_file
from lib.fire import detect_fire_simple


def build_pixel_table(files, lat_min, lat_max, lon_min, lon_max,
                      day_night='D', flight_num=''):
    """Build a per-pixel DataFrame from all files in a flight.

    Each valid pixel from each file is one row. Grid cells observed
    by multiple flight lines appear as multiple rows with different
    'file' values, preserving all observations for statistics.

    Args:
        files: list of HDF file paths for one flight.
        lat_min, lat_max, lon_min, lon_max: grid extent [degrees].
        day_night: 'D' or 'N' flag.
        flight_num: flight identifier string.

    Returns:
        pd.DataFrame with columns:
            flight, file, lat, lon, T4, T11, dT, SWIR, NDVI, fire
    """
    import pandas as pd

    nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
    ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))
    T4_thresh = 310.0 if day_night == 'N' else 325.0

    rows_list = []
    for filepath in files:
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

        rows_list.append(pd.DataFrame({
            'flight': flight_num,
            'file': os.path.basename(filepath),
            'lat': grid_lat,
            'lon': grid_lon,
            'T4': T4[in_bounds],
            'T11': T11[in_bounds],
            'dT': dT[in_bounds],
            'SWIR': SWIR[in_bounds],
            'NDVI': NDVI[in_bounds],
            'fire': fire[in_bounds],
        }))

    return pd.concat(rows_list, ignore_index=True)


def compute_location_stats(pixel_df):
    """Compute per-location statistics from a pixel table.

    Groups by (lat, lon) and computes mean, std, count, and fire
    detection rate for each grid cell across all observations.

    Args:
        pixel_df: DataFrame from build_pixel_table().

    Returns:
        pd.DataFrame with one row per unique (lat, lon), columns:
            lat, lon, T4_mean, T4_std, T11_mean, T11_std,
            dT_mean, dT_std, SWIR_mean, SWIR_std,
            NDVI_mean, NDVI_std, fire_rate, obs_count
    """
    agg = pixel_df.groupby(['lat', 'lon']).agg(
        T4_mean=('T4', 'mean'),
        T4_std=('T4', 'std'),
        T11_mean=('T11', 'mean'),
        T11_std=('T11', 'std'),
        dT_mean=('dT', 'mean'),
        dT_std=('dT', 'std'),
        SWIR_mean=('SWIR', 'mean'),
        SWIR_std=('SWIR', 'std'),
        NDVI_mean=('NDVI', 'mean'),
        NDVI_std=('NDVI', 'std'),
        fire_rate=('fire', 'mean'),
        obs_count=('fire', 'count'),
    ).reset_index()

    return agg


def compute_cell_area_m2(lat_center_deg):
    """Area of one grid cell in m² at the given latitude."""
    dy = GRID_RES * 111_000
    dx = GRID_RES * 111_000 * np.cos(np.radians(lat_center_deg))
    return dx * dy


def format_area(area_m2):
    """Format area as m² or hectares."""
    if area_m2 >= 10_000:
        return f'{area_m2 / 10_000:.1f} ha'
    return f'{area_m2:,.0f} m\u00b2'

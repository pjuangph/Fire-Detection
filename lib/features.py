"""Feature engineering for ML fire detection.

Computes 12 aggregate features per grid-cell location from pixel-level
observations. These features mirror the running accumulators maintained
by process_sweep() in lib/mosaic.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

NDArrayFloat = npt.NDArray[np.floating[Any]]


def build_location_features(
    pixel_df: pd.DataFrame,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute 12 aggregate features per grid-cell location from pixel table.

    Groups by (lat, lon) and computes running statistics that mirror
    what process_sweep() maintains in gs accumulators. Features are designed
    to normalize each pixel's temporal context for fire detection.

    Feature selection based on MODIS fire detection literature:
    - Giglio et al. (2003) "An Enhanced Contextual Fire Detection Algorithm"
    - Schroeder et al. (2014) "The New VIIRS 375m Active Fire Detection"

    The 12 aggregate features:
        1. T4_max: Peak 3.9um brightness temperature (fire spike)
        2. T4_mean: Average thermal state (normalizes peak)
        3. T11_mean: Background 11um temperature (stable reference)
        4. dT_max: Strongest T4-T11 difference (fire signature)
        5. SWIR_max: Peak 2.2um radiance (fire emits strongly in SWIR)
        6. SWIR_mean: Average SWIR radiance (normalizes peak)
        7. Red_mean: Average Red radiance (~0 at night, ~40+ day)
        8. NIR_mean: Average NIR radiance (~0 at night, ~40+ day)
        9. NDVI_min: Lowest vegetation index (burn scar indicator)
       10. NDVI_mean: Average vegetation (normalizes drop)
       11. NDVI_drop: First NDVI - min NDVI (temporal vegetation loss)
       12. obs_count: Number of observations (reliability indicator)

    Args:
        pixel_df: DataFrame from build_pixel_table() with columns
            lat, lon, T4, T11, dT, SWIR, Red, NIR, NDVI, fire.

    Returns:
        Tuple of (X, y, lats, lons) where:
            - X: (N_locations, 12) float32 feature matrix
            - y: (N_locations,) float32 labels (1 if any observation was fire)
            - lats: (N_locations,) latitude coordinates
            - lons: (N_locations,) longitude coordinates
    """
    grouped = pixel_df.groupby(['lat', 'lon'])

    T4_max = np.asarray(grouped['T4'].max().values)
    T4_mean = np.asarray(grouped['T4'].mean().values)
    T11_mean = np.asarray(grouped['T11'].mean().values)
    dT_max = np.asarray(grouped['dT'].max().values)

    # SWIR features: Ch 22 (2.162 um) -- fire emits strongly in SWIR due to
    # Planck radiation from hot sources. At night, only thermal emission
    # contributes, so high SWIR at night = definite fire.
    swir_max = np.asarray(grouped['SWIR'].max().values)
    swir_mean = np.asarray(grouped['SWIR'].mean().values)

    # NDVI features: only from daytime (finite NDVI) observations
    ndvi_min = np.asarray(grouped['NDVI'].min().values)
    ndvi_mean = np.asarray(grouped['NDVI'].mean().values)

    # NDVI_drop: first NDVI - min NDVI (vegetation loss over time)
    def ndvi_drop_fn(g: pd.Series) -> float:
        finite = g[np.isfinite(g)]
        if len(finite) < 2:
            return 0.0
        return float(finite.iloc[0] - finite.min())
    ndvi_drop = np.asarray(grouped['NDVI'].apply(ndvi_drop_fn).values)

    obs_count = np.asarray(grouped['T4'].count().values)

    # Label: 1 if any observation at this location was fire
    fire_rate = np.asarray(grouped['fire'].mean().values)
    y = (fire_rate > 0).astype(np.float32)

    # Fill NaN NDVI features with 0 (night-only pixels)
    ndvi_min = np.where(np.isfinite(ndvi_min), ndvi_min, 0.0)
    ndvi_mean = np.where(np.isfinite(ndvi_mean), ndvi_mean, 0.0)
    ndvi_drop = np.where(np.isfinite(ndvi_drop), ndvi_drop, 0.0)

    # Fill NaN SWIR with 0 (shouldn't happen, but safety)
    swir_max = np.where(np.isfinite(swir_max), swir_max, 0.0)
    swir_mean = np.where(np.isfinite(swir_mean), swir_mean, 0.0)

    # Red/NIR mean: implicit day/night indicator
    # Daytime: Red ~20-60, NIR ~20-60 W/m^2/sr/um (solar reflection)
    # Nighttime: Red ~0, NIR ~0 (no solar signal, just sensor noise)
    red_mean = np.asarray(grouped['Red'].mean().values)
    nir_mean = np.asarray(grouped['NIR'].mean().values)
    red_mean = np.where(np.isfinite(red_mean), red_mean, 0.0)
    nir_mean = np.where(np.isfinite(nir_mean), nir_mean, 0.0)

    X = np.stack([
        T4_max, T4_mean, T11_mean, dT_max,
        swir_max, swir_mean,
        red_mean, nir_mean,
        ndvi_min, ndvi_mean, ndvi_drop, obs_count,
    ], axis=1).astype(np.float32)

    # Replace any remaining non-finite with 0
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)

    coords = grouped['T4'].count().reset_index()
    lats = np.asarray(coords['lat'].values)
    lons = np.asarray(coords['lon'].values)

    return X, y, lats, lons

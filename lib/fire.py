"""Fire detection algorithms: threshold, contextual, and zone analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label as ndimage_label


def detect_fire_simple(T4: np.ndarray, T11: np.ndarray,
                       T4_thresh: float = 325.0,
                       dT_thresh: float = 10.0) -> np.ndarray:
    """Simple absolute fire detection (no contextual test, for speed on mosaics).

    Args:
        T4: Brightness temperature at ~3.9 μm [K].
        T11: Brightness temperature at ~11.25 μm [K].
        T4_thresh: Absolute fire threshold [K]. 325 K (52°C) for daytime,
                   310 K (37°C) for nighttime.
        dT_thresh: Minimum T4-T11 difference [K].
    """
    dT = T4 - T11
    return (T4 > T4_thresh) & (dT > dT_thresh)


def is_daytime(solar_zenith: np.ndarray, threshold: float = 85.0) -> np.ndarray:
    """Return boolean mask: True where pixel is daytime (SZA < threshold).

    Args:
        solar_zenith: Solar zenith angle [degrees]. 0° = sun overhead, 90° = horizon.
        threshold: Day/night boundary [degrees]. MODIS MOD14 uses 85°.
    """
    return solar_zenith < threshold


def _contextual_stats(arr: np.ndarray,
                      window: int = 61) -> tuple[np.ndarray, np.ndarray]:
    """Compute NaN-aware local mean and std using a cumulative-sum box filter.

    Args:
        arr: 2D array with possible NaN values.
        window: square window size (must be odd).

    Returns:
        (local_mean, local_std) arrays, same shape as arr.
    """
    half = window // 2

    valid = (~np.isnan(arr)).astype(np.float64)
    filled = np.where(np.isnan(arr), 0.0, arr).astype(np.float64)

    valid_p = np.pad(valid, half, mode='reflect')
    filled_p = np.pad(filled, half, mode='reflect')
    filled2_p = np.pad(filled ** 2, half, mode='reflect')

    def sat(x):
        s = np.cumsum(np.cumsum(x, axis=0), axis=1)
        s = np.pad(s, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        return s

    sv = sat(valid_p)
    sf = sat(filled_p)
    sf2 = sat(filled2_p)

    rows, cols = arr.shape
    r1 = np.arange(rows)[:, None]
    c1 = np.arange(cols)[None, :]
    r2 = r1 + window
    c2 = c1 + window

    def rect_sum(s):
        return (s[r2, c2] - s[r1, c2] - s[r2, c1] + s[r1, c1])

    count = rect_sum(sv)
    sum_f = rect_sum(sf)
    sum_f2 = rect_sum(sf2)

    count = np.maximum(count, 1)
    local_mean = sum_f / count
    local_var = sum_f2 / count - local_mean ** 2
    local_std = np.sqrt(np.maximum(local_var, 0))

    return local_mean, local_std


def detect_fire(T4: np.ndarray, T11: np.ndarray, daytime: np.ndarray,
                T4_day_thresh: float = 325.0,
                T4_night_thresh: float = 310.0,
                delta_T_thresh: float = 10.0,
                context_window: int = 61,
                context_sigma: float = 3.0) -> dict[str, Any]:
    """Run fire detection on a MASTER scene with contextual anomaly test.

    Args:
        T4: Brightness temperature at ~3.9 μm [K].
        T11: Brightness temperature at ~11.25 μm [K].
        daytime: Boolean mask, True = daytime pixel.
        T4_day_thresh: Daytime absolute T4 threshold [K].
        T4_night_thresh: Nighttime absolute T4 threshold [K].
        delta_T_thresh: Minimum T4-T11 difference [K].
        context_window: Sliding window size [pixels].
        context_sigma: Number of std deviations above local mean for anomaly.

    Returns dict with detection masks and intermediate arrays.
    """
    delta_T = T4 - T11

    threshold = np.where(daytime, T4_day_thresh, T4_night_thresh)
    absolute_mask = (T4 > threshold) & (delta_T > delta_T_thresh)

    bg_mean_T4, bg_std_T4 = _contextual_stats(T4, context_window)
    bg_mean_dT, bg_std_dT = _contextual_stats(delta_T, context_window)

    contextual_mask = (
        (T4 > bg_mean_T4 + context_sigma * bg_std_T4) &
        (delta_T > bg_mean_dT + context_sigma * bg_std_dT) &
        (delta_T > delta_T_thresh)
    )

    combined_mask = absolute_mask | contextual_mask

    return {
        'absolute_mask': absolute_mask,
        'contextual_mask': contextual_mask,
        'combined_mask': combined_mask,
        'T4': T4,
        'T11': T11,
        'delta_T': delta_T,
        'daytime': daytime,
    }


def detect_fire_zones(fire_mask: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    """Find connected fire zones using 8-connectivity.

    Returns:
        labels: 2D int array (0 = no fire, 1..N = zone ID)
        n_zones: number of zones
        zone_sizes: list of (zone_id, pixel_count) sorted largest-first
    """
    structure = np.ones((3, 3))
    labels, n_zones = ndimage_label(fire_mask, structure=structure)
    zone_sizes = []
    for z in range(1, n_zones + 1):
        zone_sizes.append((z, int(np.sum(labels == z))))
    zone_sizes.sort(key=lambda x: -x[1])
    return labels, n_zones, zone_sizes

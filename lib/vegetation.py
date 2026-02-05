"""Vegetation index computation."""

from __future__ import annotations

import numpy as np


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index from radiance.

    NDVI = (NIR - Red) / (NIR + Red)

    Works with radiance because the ratio cancels solar irradiance
    to first order (same sun angle, same atmospheric path).

    Args:
        red: Red band radiance [W/m²/sr/μm], any shape.
        nir: NIR band radiance [W/m²/sr/μm], same shape as red.

    Returns:
        NDVI array, range [-1, 1]. NaN where inputs are invalid
        or denominator is zero.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = (nir - red) / (nir + red)
    ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
    return ndvi


def has_sunlight(red: np.ndarray, nir: np.ndarray, threshold: float = 5.0) -> bool:
    """Check if VNIR bands have meaningful solar signal.

    At night or under heavy cloud, reflected-solar bands (Red, NIR) read
    near zero. Checking actual radiance is more robust than Solar Zenith
    Angle because SZA only tells you if the sun is geometrically above
    the horizon — it can't detect cloud blocking the signal.

    Empirical MASTER NIR radiance ranges:
        Daytime:   median ~40, min ~7 W/m²/sr/μm
        Nighttime: median ~0.2, p95 ~0.5 W/m²/sr/μm (sensor noise)

    Args:
        red: Red band radiance [W/m²/sr/μm], any shape.
        nir: NIR band radiance [W/m²/sr/μm], same shape.
        threshold: minimum median NIR radiance to consider "sunlit"
                   [W/m²/sr/μm]. Default 5.0 sits between nighttime
                   noise (~0.5) and daytime minimum (~7).

    Returns:
        True if the scene has usable solar illumination.
    """
    valid_nir = nir[np.isfinite(nir) & (nir > 0)]
    if len(valid_nir) == 0:
        return False
    return float(np.median(valid_nir)) > threshold


def detect_vegetation_loss(baseline: np.ndarray, current: np.ndarray,
                           threshold: float = 0.15) -> np.ndarray:
    """Detect significant vegetation loss by comparing NDVI to baseline.

    A drop of >= threshold from the first-observed NDVI indicates that
    vegetation has been consumed (e.g. by fire). Uses absolute drop,
    not percentage, because a 0.15 drop is physically meaningful
    regardless of the baseline magnitude.

    Args:
        baseline: NDVI baseline array (first valid daytime observation).
        current: Current NDVI array (same shape as baseline).
        threshold: Minimum NDVI drop to flag as vegetation loss.

    Returns:
        Boolean mask, True where baseline - current >= threshold and
        both values are finite.
    """
    with np.errstate(invalid='ignore'):
        drop = baseline - current
        return np.isfinite(drop) & (drop >= threshold)

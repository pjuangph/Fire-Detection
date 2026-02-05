"""HDF file I/O: reading MASTER L1B data, grouping by flight, coordinate extents."""

import glob
import os
import numpy as np
from pyhdf.SD import SD, SDC

from lib.constants import (
    H_PLANCK, C_LIGHT, K_BOLTZ,
    CH_T4, CH_T11, CH_SWIR, CH_RED, CH_NIR,
)
from lib.vegetation import compute_ndvi


def radiance_to_bt(radiance_um, wavelength_um):
    """Convert spectral radiance to brightness temperature via inverse Planck.

    Args:
        radiance_um: Spectral radiance [W/m²/sr/μm] (MASTER native units).
        wavelength_um: Effective central wavelength [μm].

    Returns:
        Brightness temperature [K]. NaN where radiance is invalid.
    """
    lam = wavelength_um * 1e-6       # [μm] -> [m]
    L = radiance_um * 1e6            # [W/m²/sr/μm] -> [W/m²/sr/m]
    c1 = 2.0 * H_PLANCK * C_LIGHT**2  # first radiation constant [W·m²]
    c2 = H_PLANCK * C_LIGHT / K_BOLTZ  # second radiation constant [m·K]
    with np.errstate(invalid='ignore', divide='ignore'):
        T = c2 / lam / np.log(1.0 + c1 / (lam**5 * L))  # [K]
    T = np.where(np.isfinite(T) & (T > 0), T, np.nan)
    return T


def process_file(filepath):
    """Load one HDF4 file and return per-pixel data as a dict.

    Returns dict with keys:
        T4:   Brightness temperature at ~3.9 μm [K], shape (scanlines, 716).
        T11:  Brightness temperature at ~11.25 μm [K], shape (scanlines, 716).
        SWIR: Calibrated radiance at ~2.16 μm [W/m²/sr/μm], shape (scanlines, 716).
        Red:  Calibrated radiance at ~0.654 μm [W/m²/sr/μm], shape (scanlines, 716).
        NIR:  Calibrated radiance at ~0.866 μm [W/m²/sr/μm], shape (scanlines, 716).
        NDVI: Normalized Difference Vegetation Index [-1, 1], shape (scanlines, 716).
        lat:  Pixel latitude [degrees], shape (scanlines, 716).
        lon:  Pixel longitude [degrees], shape (scanlines, 716).
    """
    f = SD(filepath, SDC.READ)

    cal_ds = f.select('CalibratedData')
    scale_factors = cal_ds.attributes()['scale_factor']
    raw_t4 = cal_ds[:, CH_T4, :].astype(np.float32) * scale_factors[CH_T4]
    raw_t11 = cal_ds[:, CH_T11, :].astype(np.float32) * scale_factors[CH_T11]
    raw_swir = cal_ds[:, CH_SWIR, :].astype(np.float32) * scale_factors[CH_SWIR]
    raw_red = cal_ds[:, CH_RED, :].astype(np.float32) * scale_factors[CH_RED]
    raw_nir = cal_ds[:, CH_NIR, :].astype(np.float32) * scale_factors[CH_NIR]
    cal_ds.endaccess()

    lat = f.select('PixelLatitude')[:]
    lon = f.select('PixelLongitude')[:]

    eff_wl = f.select('EffectiveCentralWavelength_IR_bands')[:]
    temp_slope = f.select('TemperatureCorrectionSlope')[:]
    temp_intercept = f.select('TemperatureCorrectionIntercept')[:]

    f.end()

    # Mask fill values (-999) and negative radiance
    raw_t4[raw_t4 < 0] = np.nan
    raw_t11[raw_t11 < 0] = np.nan
    raw_swir[raw_swir < 0] = np.nan
    raw_red[raw_red < 0] = np.nan
    raw_nir[raw_nir < 0] = np.nan
    lat[lat == -999.0] = np.nan
    lon[lon == -999.0] = np.nan

    # Radiance -> brightness temperature for thermal channels
    T4 = radiance_to_bt(raw_t4, eff_wl[CH_T4])
    T4 = temp_slope[CH_T4] * T4 + temp_intercept[CH_T4]

    T11 = radiance_to_bt(raw_t11, eff_wl[CH_T11])
    T11 = temp_slope[CH_T11] * T11 + temp_intercept[CH_T11]

    # SWIR, Red, NIR stay as radiance (reflected solar, not thermal)
    SWIR = raw_swir
    Red = raw_red
    NIR = raw_nir
    NDVI = compute_ndvi(Red, NIR)

    return {
        'T4': T4, 'T11': T11, 'SWIR': SWIR,
        'Red': Red, 'NIR': NIR, 'NDVI': NDVI,
        'lat': lat, 'lon': lon,
    }


def group_files_by_flight():
    """Group HDF files by flight number, sorted by start time within each flight."""
    files = sorted(glob.glob('ignite_fire_data/*.hdf'))
    flights = {}
    for f in files:
        sd = SD(f, SDC.READ)
        attrs = sd.attributes()
        fnum = attrs['FlightNumber']
        comment = attrs.get('FlightComment', '')
        day_night = attrs.get('day_night_flag', '?')
        sd.end()
        if fnum not in flights:
            flights[fnum] = {'files': [], 'comment': comment, 'day_night': day_night}
        flights[fnum]['files'].append(f)
    return flights


def compute_grid_extent(files):
    """Compute the lat/lon bounding box across all files in a flight."""
    lat_min, lat_max = 90.0, -90.0
    lon_min, lon_max = 180.0, -180.0

    for filepath in files:
        sd = SD(filepath, SDC.READ)
        attrs = sd.attributes()
        lat_min = min(lat_min, attrs['lat_LL'], attrs['lat_UL'])
        lat_max = max(lat_max, attrs['lat_LR'], attrs['lat_UR'])
        lon_min = min(lon_min, attrs['lon_UL'], attrs['lon_UR'])
        lon_max = max(lon_max, attrs['lon_LL'], attrs['lon_LR'])
        sd.end()

    buf = 0.005  # [degrees] ≈ 550 m
    return lat_min - buf, lat_max + buf, lon_min - buf, lon_max + buf

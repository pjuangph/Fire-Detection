"""mosaic_flight.py - Assemble all flight lines into a single georeferenced mosaic.

For each flight, reads all HDF4 files, computes brightness temperature at 3.9 um
(fire channel) and 11.25 um (background), runs fire detection, and composites
everything onto a regular lat/lon grid. Later flight lines overwrite earlier ones,
so overlapping areas show the most recent observation.

Usage:
    python mosaic_flight.py
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyhdf.SD import SD, SDC

# Physical constants
H_PLANCK = 6.62607015e-34   # Planck constant [J·s]
C_LIGHT  = 2.99792458e8     # speed of light [m/s]
K_BOLTZ  = 1.380649e-23     # Boltzmann constant [J/K]

# MASTER channel indices (0-based)
CH_T4   = 30   # Ch 31: effective wavelength 3.9029 μm, MWIR fire channel
CH_T11  = 47   # Ch 48: effective wavelength 11.3274 μm, TIR background channel
CH_SWIR = 21   # Ch 22: nominal wavelength 2.162 μm, SWIR solar reflection channel

# Grid resolution: 0.00025 degrees ≈ 28 m at 36°N latitude.
# Native MASTER pixel spacing is ~8 m; this is ~3× downsampled for speed.
# Adjust lower (e.g. 0.00007) for native resolution or higher (e.g. 0.001) for quick previews.
GRID_RES = 0.00025  # [degrees]


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
    """Load one HDF4 file and return T4, T11, SWIR, lat, lon arrays.

    Returns:
        T4:   Brightness temperature at ~3.9 μm [K], shape (scanlines, 716).
        T11:  Brightness temperature at ~11.25 μm [K], shape (scanlines, 716).
        SWIR: Calibrated radiance at ~2.16 μm [W/m²/sr/μm], shape (scanlines, 716).
        lat:  Pixel latitude [degrees], shape (scanlines, 716).
        lon:  Pixel longitude [degrees], shape (scanlines, 716).
    """
    f = SD(filepath, SDC.READ)

    cal_ds = f.select('CalibratedData')
    scale_factors = cal_ds.attributes()['scale_factor']
    # Read fire-relevant channels + SWIR; apply scale_factor -> [W/m²/sr/μm]
    raw_t4 = cal_ds[:, CH_T4, :].astype(np.float32) * scale_factors[CH_T4]     # [W/m²/sr/μm]
    raw_t11 = cal_ds[:, CH_T11, :].astype(np.float32) * scale_factors[CH_T11]   # [W/m²/sr/μm]
    raw_swir = cal_ds[:, CH_SWIR, :].astype(np.float32) * scale_factors[CH_SWIR] # [W/m²/sr/μm]
    cal_ds.endaccess()

    lat = f.select('PixelLatitude')[:]   # [degrees]
    lon = f.select('PixelLongitude')[:]  # [degrees]

    eff_wl = f.select('EffectiveCentralWavelength_IR_bands')[:]      # [μm]
    temp_slope = f.select('TemperatureCorrectionSlope')[:]           # [unitless]
    temp_intercept = f.select('TemperatureCorrectionIntercept')[:]   # [K]

    f.end()

    # Mask fill values (-999) and negative radiance
    raw_t4[raw_t4 < 0] = np.nan
    raw_t11[raw_t11 < 0] = np.nan
    raw_swir[raw_swir < 0] = np.nan
    lat[lat == -999.0] = np.nan
    lon[lon == -999.0] = np.nan

    # Radiance [W/m²/sr/μm] -> brightness temperature [K]
    T4 = radiance_to_bt(raw_t4, eff_wl[CH_T4])
    T4 = temp_slope[CH_T4] * T4 + temp_intercept[CH_T4]   # post-Planck correction [K]

    T11 = radiance_to_bt(raw_t11, eff_wl[CH_T11])
    T11 = temp_slope[CH_T11] * T11 + temp_intercept[CH_T11]  # post-Planck correction [K]

    # SWIR stays as radiance — it's a reflected solar band, not thermal emission.
    # No Planck inversion is appropriate. High SWIR radiance during daytime indicates
    # bright/reflective surfaces (rock, soil) that may cause T4 false positives.
    SWIR = raw_swir

    return T4, T11, SWIR, lat, lon


def detect_fire_simple(T4, T11, T4_thresh=325.0, dT_thresh=10.0):
    """Simple absolute fire detection (no contextual test, for speed on mosaics).

    Args:
        T4: Brightness temperature at ~3.9 μm [K].
        T11: Brightness temperature at ~11.25 μm [K].
        T4_thresh: Absolute fire threshold [K]. 325 K (52°C) for daytime,
                   310 K (37°C) for nighttime. Daytime is higher because
                   solar heating warms surfaces to 310-320 K naturally.
        dT_thresh: Minimum T4-T11 difference [K]. Filters out warm-but-not-fire
                   surfaces (sun-heated rock, roads) which have small dT.
                   Fire has large dT because it emits disproportionately at 3.9 μm.
    """
    dT = T4 - T11  # [K]
    return (T4 > T4_thresh) & (dT > dT_thresh)


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
        # Latitude LL = lower left corner, UL = Upper left, UR = Upper Right LR = Lower Right 
        # Latitude LL = lower left corner, UL = Upper left
        lat_min = min(lat_min, attrs['lat_LL'], attrs['lat_UL']) # Latitude LL = lower left corner, UL = Upper left
        lat_max = max(lat_max, attrs['lat_LR'], attrs['lat_UR'])
        lon_min = min(lon_min, attrs['lon_UL'], attrs['lon_UR'])
        lon_max = max(lon_max, attrs['lon_LL'], attrs['lon_LR'])
        sd.end()

    # Add a small buffer to avoid clipping edge pixels
    buf = 0.005  # [degrees] ≈ 550 m
    return lat_min - buf, lat_max + buf, lon_min - buf, lon_max + buf


def build_mosaic(files, lat_min, lat_max, lon_min, lon_max, day_night='D'):
    """Build a gridded mosaic from a list of flight line files.

    Files are processed in order (chronological), so later lines overwrite earlier
    ones in overlapping areas. A multi-pass consistency filter requires fire to be
    detected in >=2 passes for pixels observed multiple times, filtering out
    angle-dependent false positives (e.g. solar reflection).

    Returns gridded T4, T11, SWIR, fire mask, axes, fire_count, and obs_count.
    """
    nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
    ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))

    grid_T4 = np.full((nrows, ncols), np.nan, dtype=np.float32)    # [K]
    grid_T11 = np.full((nrows, ncols), np.nan, dtype=np.float32)  # [K]
    grid_SWIR = np.full((nrows, ncols), np.nan, dtype=np.float32)  # [W/m²/sr/μm]
    grid_fire_count = np.zeros((nrows, ncols), dtype=np.int32)     # times detected as fire
    grid_obs_count = np.zeros((nrows, ncols), dtype=np.int32)      # times observed (valid data)

    # Night threshold is lower (310 K) because there's no solar heating;
    # a 310 K pixel at night is already very anomalous.
    # Day threshold is higher (325 K) to avoid false positives from sun-heated ground.
    T4_thresh = 310.0 if day_night == 'N' else 325.0  # [K]

    for i, filepath in enumerate(files):
        name = os.path.basename(filepath)
        print(f'  [{i+1}/{len(files)}] {name}')

        T4, T11, SWIR, lat, lon = process_file(filepath)
        fire = detect_fire_simple(T4, T11, T4_thresh=T4_thresh)

        # Map pixel lat/lon [degrees] to grid row/col indices
        valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(T4)
        row_idx = ((lat_max - lat) / GRID_RES).astype(np.int32)   # row 0 = north (lat_max)
        col_idx = ((lon - lon_min) / GRID_RES).astype(np.int32)   # col 0 = west (lon_min)

        # Clip to grid bounds and combine with valid mask
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

    # Multi-pass consistency filter:
    # - Pixels observed >=2 times must trigger fire in >=2 passes (filters angle-dependent FP)
    # - Pixels observed only once keep their single detection (no multi-pass info available)
    multi_pass = grid_obs_count >= 2
    grid_fire = np.where(
        multi_pass,
        grid_fire_count >= 2,
        grid_fire_count >= 1
    )

    lat_axis = np.linspace(lat_max, lat_min, nrows)
    lon_axis = np.linspace(lon_min, lon_max, ncols)

    return grid_T4, grid_T11, grid_SWIR, grid_fire, lat_axis, lon_axis, grid_fire_count, grid_obs_count


def plot_mosaic(grid_T4, grid_fire, lat_axis, lon_axis, flight_num, comment, n_files):
    """Plot a flight mosaic with fire overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Flight {flight_num} Mosaic — {comment}\n'
                 f'{n_files} flight lines composited', fontsize=13)

    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]

    # T4 brightness temperature
    valid_T4 = grid_T4[np.isfinite(grid_T4)]
    if len(valid_T4) > 0:
        vmin, vmax = np.percentile(valid_T4, [2, 98])
    else:
        vmin, vmax = 280, 320

    im = axes[0].imshow(grid_T4, extent=extent, aspect='equal',
                        cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'T4 Brightness Temp (3.9 μm)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[0], label='K', fraction=0.046, pad=0.04)

    # T4 with fire overlay
    axes[1].imshow(grid_T4, extent=extent, aspect='equal',
                   cmap='gray', vmin=vmin, vmax=vmax)

    fire_count = np.sum(grid_fire)
    if fire_count > 0:
        fire_rows, fire_cols = np.where(grid_fire)
        fire_lats = lat_axis[0] - fire_rows * GRID_RES  # lat decreases with row
        # Fix: lat_axis goes from lat_max to lat_min, row 0 = lat_max
        fire_lats = lat_axis[0] + (lat_axis[-1] - lat_axis[0]) * fire_rows / (len(lat_axis) - 1)
        fire_lons = lon_axis[0] + (lon_axis[-1] - lon_axis[0]) * fire_cols / (len(lon_axis) - 1)
        axes[1].scatter(fire_lons, fire_lats, s=0.2, c='red', alpha=0.7,
                        label=f'Fire ({fire_count:,} grid cells)')
        axes[1].legend(loc='upper right', markerscale=15, fontsize=9)

    axes[1].set_title(f'Fire Detection Overlay')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    outname = f'plots/mosaic_flight_{flight_num.replace("-", "")}.png'
    plt.savefig(outname, dpi=200)
    print(f'  Saved {outname}')
    plt.close()
    return outname


def main():
    flights = group_files_by_flight()

    print(f'Found {len(flights)} flights:\n')
    for fnum, info in sorted(flights.items()):
        print(f'  {fnum}: {len(info["files"])} lines — {info["comment"]}')
    print()

    for fnum, info in sorted(flights.items()):
        files = info['files']
        comment = info['comment']
        day_night = info['day_night']

        print(f'Processing flight {fnum} ({len(files)} files)...')
        lat_min, lat_max, lon_min, lon_max = compute_grid_extent(files)
        nrows = int(np.ceil((lat_max - lat_min) / GRID_RES))
        ncols = int(np.ceil((lon_max - lon_min) / GRID_RES))
        print(f'  Grid: {nrows} x {ncols} ({GRID_RES*111000:.0f}m resolution)')
        print(f'  Extent: lat [{lat_min:.4f}, {lat_max:.4f}], '
              f'lon [{lon_min:.4f}, {lon_max:.4f}]')

        grid_T4, grid_T11, grid_SWIR, grid_fire, lat_axis, lon_axis, grid_fire_count, grid_obs_count = build_mosaic(
            files, lat_min, lat_max, lon_min, lon_max, day_night)

        valid_pix = np.sum(np.isfinite(grid_T4))
        fire_pix = np.sum(grid_fire)
        multi_pass_confirmed = np.sum(grid_fire & (grid_obs_count >= 2))
        single_pass_only = np.sum(grid_fire & (grid_obs_count < 2))
        coverage = 100.0 * valid_pix / (nrows * ncols)
        print(f'  Coverage: {coverage:.1f}% of grid filled')
        print(f'  Fire pixels: {fire_pix:,} ({multi_pass_confirmed:,} multi-pass, {single_pass_only:,} single-pass)')

        plot_mosaic(grid_T4, grid_fire, lat_axis, lon_axis,
                    fnum, comment, len(files))
        print()

    print('Done — all flight mosaics generated.')


if __name__ == '__main__':
    main()

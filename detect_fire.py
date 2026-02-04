"""detect_fire.py - Fire detection in MASTER L1B HDF4 data.

Implements a MOD14-inspired fire detection algorithm using:
  - Absolute brightness temperature thresholds (T4 > 325K daytime)
  - Contextual anomaly detection (T4 exceeds local background by N sigma)

Usage:
    python detect_fire.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC

# Physical constants
H_PLANCK = 6.62607015e-34   # Planck constant, J*s
C_LIGHT  = 2.99792458e8     # speed of light, m/s
K_BOLTZ  = 1.380649e-23     # Boltzmann constant, J/K

# MASTER channel indices (0-based)
CH_T4  = 30   # Ch 31: ~3.9 um, MWIR fire channel
CH_T11 = 47   # Ch 48: ~11.25 um, TIR background channel


# ── Data Loading ───────────────────────────────────────────

def load_master_file(filepath):
    """Load key datasets from a MASTER L1B HDF4 file.

    Returns dict with radiance (only Ch 31 and 48), lat, lon,
    solar zenith, effective wavelengths, and temp correction coefficients.
    """
    f = SD(filepath, SDC.READ)

    # Calibrated radiance for fire channels only (saves memory)
    cal_ds = f.select('CalibratedData')
    scale_factors = cal_ds.attributes()['scale_factor']

    raw_t4 = cal_ds[:, CH_T4, :].astype(np.float32) * scale_factors[CH_T4]
    raw_t11 = cal_ds[:, CH_T11, :].astype(np.float32) * scale_factors[CH_T11]
    cal_ds.endaccess()

    lat = f.select('PixelLatitude')[:]
    lon = f.select('PixelLongitude')[:]
    solar_zenith = f.select('SolarZenithAngle')[:]
    eff_wl = f.select('EffectiveCentralWavelength_IR_bands')[:]
    temp_slope = f.select('TemperatureCorrectionSlope')[:]
    temp_intercept = f.select('TemperatureCorrectionIntercept')[:]

    f.end()

    # Mask fill values
    fill = -999.0
    raw_t4[raw_t4 < 0] = np.nan
    raw_t11[raw_t11 < 0] = np.nan
    lat[lat == fill] = np.nan
    lon[lon == fill] = np.nan
    solar_zenith[solar_zenith == fill] = np.nan

    return {
        'radiance_t4': raw_t4,
        'radiance_t11': raw_t11,
        'lat': lat,
        'lon': lon,
        'solar_zenith': solar_zenith,
        'eff_wavelengths': eff_wl,
        'temp_corr_slope': temp_slope,
        'temp_corr_intercept': temp_intercept,
        'filename': os.path.basename(filepath),
    }


# ── Planck Conversion ─────────────────────────────────────

def radiance_to_brightness_temp(radiance_um, wavelength_um):
    """Convert spectral radiance (W/m2/sr/um) to brightness temperature (K).

    Uses the inverse Planck function with the effective central wavelength.
    """
    lam = wavelength_um * 1e-6          # meters
    L = radiance_um * 1e6               # W/m2/sr/m

    c1 = 2.0 * H_PLANCK * C_LIGHT**2   # W*m2
    c2 = H_PLANCK * C_LIGHT / K_BOLTZ   # m*K

    with np.errstate(invalid='ignore', divide='ignore'):
        T_b = c2 / lam / np.log(1.0 + c1 / (lam**5 * L))

    T_b = np.where(np.isfinite(T_b) & (T_b > 0), T_b, np.nan)
    return T_b


def apply_temp_correction(T_planck, slope, intercept):
    """Apply MASTER post-Planck temperature correction: T = slope*T + intercept."""
    return slope * T_planck + intercept


def compute_fire_channels(data):
    """Compute corrected brightness temperatures for T4 (~3.9um) and T11 (~11.25um).

    Returns (T4, T11) arrays in Kelvin.
    """
    eff_wl = data['eff_wavelengths']

    T4 = radiance_to_brightness_temp(data['radiance_t4'], eff_wl[CH_T4])
    T4 = apply_temp_correction(
        T4, data['temp_corr_slope'][CH_T4], data['temp_corr_intercept'][CH_T4])

    T11 = radiance_to_brightness_temp(data['radiance_t11'], eff_wl[CH_T11])
    T11 = apply_temp_correction(
        T11, data['temp_corr_slope'][CH_T11], data['temp_corr_intercept'][CH_T11])

    return T4, T11


# ── Fire Detection ─────────────────────────────────────────

def is_daytime(solar_zenith, threshold=85.0):
    """Return boolean mask: True where pixel is daytime (SZA < threshold)."""
    return solar_zenith < threshold


def _contextual_stats(arr, window=61):
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

    # Pad with reflection
    valid_p = np.pad(valid, half, mode='reflect')
    filled_p = np.pad(filled, half, mode='reflect')
    filled2_p = np.pad(filled ** 2, half, mode='reflect')

    # 2D cumulative sum (summed area table) with leading row/col of zeros
    def sat(x):
        s = np.cumsum(np.cumsum(x, axis=0), axis=1)
        # Prepend a row and column of zeros for clean inclusion-exclusion
        s = np.pad(s, ((1, 0), (1, 0)), mode='constant', constant_values=0)
        return s

    sv = sat(valid_p)
    sf = sat(filled_p)
    sf2 = sat(filled2_p)

    # Extract rectangle sums using inclusion-exclusion
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


def detect_fire(T4, T11, daytime,
                T4_day_thresh=325.0,
                T4_night_thresh=310.0,
                delta_T_thresh=10.0,
                context_window=61,
                context_sigma=3.0):
    """Run fire detection on a MASTER scene.

    Returns dict with detection masks and intermediate arrays.
    """
    delta_T = T4 - T11

    # --- Absolute threshold test ---
    threshold = np.where(daytime, T4_day_thresh, T4_night_thresh)
    absolute_mask = (T4 > threshold) & (delta_T > delta_T_thresh)

    # --- Contextual anomaly test ---
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


# ── Output ─────────────────────────────────────────────────

def print_summary(filepath, result):
    """Print fire detection summary to console."""
    T4, T11, dT = result['T4'], result['T11'], result['delta_T']
    day = result['daytime']
    abs_n = np.nansum(result['absolute_mask'])
    ctx_n = np.nansum(result['contextual_mask'])
    comb_n = np.nansum(result['combined_mask'])
    valid = np.sum(np.isfinite(T4))

    print('=' * 60)
    print(f'Fire Detection: {os.path.basename(filepath)}')
    print('=' * 60)
    print(f'  T4  range: {np.nanmin(T4):.1f} - {np.nanmax(T4):.1f} K  '
          f'({np.nanmin(T4)-273.15:.1f} - {np.nanmax(T4)-273.15:.1f} C)')
    print(f'  T11 range: {np.nanmin(T11):.1f} - {np.nanmax(T11):.1f} K')
    print(f'  dT  range: {np.nanmin(dT):.1f} - {np.nanmax(dT):.1f} K')
    pct_day = 100.0 * np.nansum(day) / max(np.sum(np.isfinite(day)), 1)
    print(f'  Day/Night: {pct_day:.1f}% daytime')
    print(f'  Absolute threshold fires:  {abs_n:,} pixels')
    print(f'  Contextual anomaly fires:  {ctx_n:,} pixels')
    pct = 100.0 * comb_n / max(valid, 1)
    print(f'  Combined fire pixels:      {comb_n:,} ({pct:.3f}% of valid)')
    print('-' * 60)
    print()


def plot_detection(data, result, filepath, suffix=''):
    """Plot 2x2 panel: T4, T11, delta-T, fire overlay."""
    T4, T11, dT = result['T4'], result['T11'], result['delta_T']
    fire = result['combined_mask']
    name = os.path.basename(filepath)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Fire Detection: {name}', fontsize=13)

    # T4
    v = np.nanpercentile(T4, [2, 98])
    im0 = axes[0, 0].imshow(T4, aspect='auto', cmap='inferno', vmin=v[0], vmax=v[1])
    axes[0, 0].set_title(f'T4 (3.9 um) Brightness Temp')
    plt.colorbar(im0, ax=axes[0, 0], label='K', fraction=0.046)

    # T11
    v = np.nanpercentile(T11, [2, 98])
    im1 = axes[0, 1].imshow(T11, aspect='auto', cmap='inferno', vmin=v[0], vmax=v[1])
    axes[0, 1].set_title(f'T11 (11.25 um) Brightness Temp')
    plt.colorbar(im1, ax=axes[0, 1], label='K', fraction=0.046)

    # Delta-T
    v = np.nanpercentile(dT, [2, 98])
    im2 = axes[1, 0].imshow(dT, aspect='auto', cmap='RdYlBu_r', vmin=v[0], vmax=v[1])
    axes[1, 0].set_title(f'Delta-T (T4 - T11)')
    plt.colorbar(im2, ax=axes[1, 0], label='K', fraction=0.046)

    # Fire overlay
    axes[1, 1].imshow(T4, aspect='auto', cmap='gray',
                      vmin=np.nanpercentile(T4, 2), vmax=np.nanpercentile(T4, 98))
    fire_y, fire_x = np.where(fire)
    abs_y, abs_x = np.where(result['absolute_mask'])
    ctx_only = result['contextual_mask'] & ~result['absolute_mask']
    ctx_y, ctx_x = np.where(ctx_only)
    axes[1, 1].scatter(ctx_x, ctx_y, s=0.3, c='orange', alpha=0.7, label=f'Contextual ({len(ctx_y):,})')
    axes[1, 1].scatter(abs_x, abs_y, s=0.3, c='red', alpha=0.7, label=f'Absolute ({len(abs_y):,})')
    axes[1, 1].set_title(f'Fire Pixels ({np.sum(fire):,} total)')
    axes[1, 1].legend(loc='upper right', markerscale=10, fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Scanline')

    plt.tight_layout()
    outname = f'fire_detection_{suffix}.png'
    plt.savefig(outname, dpi=150)
    print(f'Saved {outname}')
    plt.close()


def plot_map(data, result, filepath):
    """Plot georeferenced fire pixel map."""
    T4 = result['T4']
    fire = result['combined_mask']
    lat, lon = data['lat'], data['lon']
    name = os.path.basename(filepath)

    fig, ax = plt.subplots(figsize=(10, 8))
    v = np.nanpercentile(T4, [2, 98])
    sc = ax.pcolormesh(lon, lat, T4, cmap='gray', vmin=v[0], vmax=v[1], shading='auto')
    plt.colorbar(sc, ax=ax, label='T4 Brightness Temp (K)', fraction=0.046)

    fire_lat = lat[fire]
    fire_lon = lon[fire]
    if len(fire_lat) > 0:
        ax.scatter(fire_lon, fire_lat, s=1, c='red', alpha=0.8, label=f'Fire ({len(fire_lat):,} px)')
        ax.legend(loc='upper right', markerscale=5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Fire Map: {name}')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('fire_map_burn.png', dpi=150)
    print('Saved fire_map_burn.png')
    plt.close()


def plot_comparison(data_pre, result_pre, data_burn, result_burn):
    """Side-by-side pre-burn vs burn comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pre-Burn vs Burn Comparison (T4 ~3.9 um)', fontsize=13)

    for ax, data, result, label in [
        (axes[0], data_pre, result_pre, 'Pre-Burn (no fire)'),
        (axes[1], data_burn, result_burn, 'Burn flight'),
    ]:
        T4 = result['T4']
        v = np.nanpercentile(T4, [2, 98])
        ax.imshow(T4, aspect='auto', cmap='gray', vmin=v[0], vmax=v[1])

        fire = result['combined_mask']
        fy, fx = np.where(fire)
        ax.scatter(fx, fy, s=0.3, c='red', alpha=0.7)
        ax.set_title(f'{label}\n{data["filename"]}\nFire pixels: {np.sum(fire):,}')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Scanline')

    plt.tight_layout()
    plt.savefig('fire_comparison.png', dpi=150)
    print('Saved fire_comparison.png')
    plt.close()


# ── Main ───────────────────────────────────────────────────

def main():
    preburn_file = 'ignite_fire_data/MASTERL1B_2480103_05_20231018_1849_1853_V01.hdf'
    burn_file = 'ignite_fire_data/MASTERL1B_2480104_20_20231019_2031_2033_V01.hdf'

    # Pre-burn
    print('Loading pre-burn file...')
    data_pre = load_master_file(preburn_file)
    T4_pre, T11_pre = compute_fire_channels(data_pre)
    day_pre = is_daytime(data_pre['solar_zenith'])
    result_pre = detect_fire(T4_pre, T11_pre, day_pre)
    print_summary(preburn_file, result_pre)
    plot_detection(data_pre, result_pre, preburn_file, suffix='preburn')

    # Burn
    print('Loading burn file...')
    data_burn = load_master_file(burn_file)
    T4_burn, T11_burn = compute_fire_channels(data_burn)
    day_burn = is_daytime(data_burn['solar_zenith'])
    result_burn = detect_fire(T4_burn, T11_burn, day_burn)
    print_summary(burn_file, result_burn)
    plot_detection(data_burn, result_burn, burn_file, suffix='burn')
    plot_map(data_burn, result_burn, burn_file)

    # Comparison
    plot_comparison(data_pre, result_pre, data_burn, result_burn)

    print('Done.')


if __name__ == '__main__':
    main()

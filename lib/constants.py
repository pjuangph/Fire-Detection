"""Physical constants, MASTER channel indices, and grid parameters."""

# Physical constants
H_PLANCK = 6.62607015e-34   # Planck constant [J·s]
C_LIGHT  = 2.99792458e8     # speed of light [m/s]
K_BOLTZ  = 1.380649e-23     # Boltzmann constant [J/K]

# MASTER channel indices (0-based)
CH_T4   = 30   # Ch 31: effective wavelength 3.9029 μm, MWIR fire channel
CH_T11  = 47   # Ch 48: effective wavelength 11.3274 μm, TIR background channel
CH_SWIR = 21   # Ch 22: nominal wavelength 2.162 μm, SWIR solar reflection channel
CH_RED  = 4    # Ch 5:  nominal wavelength 0.654 μm, VNIR Red band (for NDVI)
CH_NIR  = 8    # Ch 9:  nominal wavelength 0.866 μm, VNIR NIR band (for NDVI)

# Grid resolution: 0.00025 degrees ≈ 28 m at 36°N latitude.
# Native MASTER pixel spacing is ~8 m; this is ~3× downsampled for speed.
GRID_RES = 0.00025  # [degrees]

# Ignition temperature of dry wood (piloted ignition ≈ 300 °C = 573.15 K).
# YAML configs store this in °C; code converts to K via + 273.15.
# Used to normalize thermal features: T_norm = T / T_IGNITION.
T_IGNITION_DRY_WOOD = 573.15  # [K]

# Feature indices for thermal vs non-thermal normalization.
THERMAL_FEATURE_INDICES = [0, 1, 2, 3]      # T4_max, T4_mean, T11_mean, dT_max
NON_THERMAL_FEATURE_INDICES = [4, 5, 6, 7, 8, 9, 10, 11]  # SWIR, Red, NIR, NDVI, obs

# Vegetation loss threshold for fire confirmation [NDVI units].
# NDVI drop >= 0.15 from baseline indicates burned vegetation.
# Healthy grassland NDVI: 0.3–0.6; burned: -0.1 to 0.1.
# 0.15 is conservative enough to avoid false positives from
# illumination angle changes (~0.05 NDVI noise).
VEG_LOSS_THRESHOLD = 0.15

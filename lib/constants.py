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

# Vegetation loss threshold for fire confirmation [NDVI units].
# NDVI drop >= 0.15 from baseline indicates burned vegetation.
# Healthy grassland NDVI: 0.3–0.6; burned: -0.1 to 0.1.
# 0.15 is conservative enough to avoid false positives from
# illumination angle changes (~0.05 NDVI noise).
VEG_LOSS_THRESHOLD = 0.15

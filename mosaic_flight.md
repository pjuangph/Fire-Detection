# Flight Mosaic Documentation

`mosaic_flight.py` assembles all flight lines from a single flight into one georeferenced composite image, giving a complete spatial view of the survey area with fire detections overlaid.

## What It Does

Each HDF4 file contains one flight line — a single pass of the aircraft over the target area. A full flight consists of many parallel passes (9 to 40 lines). This script:

1. Groups all HDF files by flight number
2. Creates a regular latitude/longitude grid covering the full flight extent
3. Processes each file in chronological order, projecting pixels onto the grid
4. Runs fire detection on each file
5. Composites everything into a single image per flight

```
Flight line 1:   ═══════════════
Flight line 2:     ═══════════════
Flight line 3:   ═══════════════
    ...                              →  Combined mosaic
Flight line N:     ═══════════════
```

## Key Variables: grid_T4 vs grid_T11

The mosaic produces two temperature grids from two different infrared channels:

- **grid_T4** — Brightness temperature at **~3.9 μm** (MASTER Channel 31) [K]. This is the **fire detection channel**. At this wavelength, fire (600–1200 K) emits enormously more radiation than the cool background (~300 K) — a fire pixel can be **23,000× brighter** than its surroundings. This extreme sensitivity is what makes 3.9 μm the primary fire detection band.

- **grid_T11** — Brightness temperature at **~11.25 μm** (MASTER Channel 48) [K]. This is the **background temperature channel**. It measures "normal" surface temperature. Fire is only ~4× brighter than background at this wavelength, so it provides the baseline against which T4 anomalies are measured.

**Why we need both:**

| Scenario | T4 (3.9 μm) | T11 (11.25 μm) | ΔT = T4 − T11 | Detected? |
|----------|-------------|-----------------|----------------|-----------|
| Cool ground | 290 K | 288 K | 2 K | No — both low, small ΔT |
| Sun-heated rock | 320 K | 315 K | 5 K | No — warm but ΔT < 10 K |
| Active fire | 600 K | 330 K | 270 K | **Yes** — T4 > 325 K AND ΔT > 10 K |

Using T4 alone would give false positives from warm ground. Using ΔT alone wouldn't catch absolute temperature. The combination of both is what makes the detection robust.

Currently only `grid_T4` is visualized in the mosaic plots. `grid_T11` is computed and stored but not plotted — it could be added as an additional panel if needed.

## How the Gridding Works

### Grid Construction

A regular grid is created in latitude/longitude coordinates:

- **Bounding box**: Determined from the `lat_LL`, `lat_UL`, `lat_LR`, `lat_UR`, `lon_LL`, `lon_UL`, `lon_LR`, `lon_UR` global attributes of all files in the flight, plus a 0.005° (~550 m) buffer on each side
- **Resolution**: 0.00025° per grid cell, which equals approximately **28 meters** at 36°N latitude (the Kaibab Plateau)
- **Orientation**: Row 0 = northernmost latitude (top), Column 0 = westernmost longitude (left)

### Pixel Mapping

For each flight line file, every pixel's latitude/longitude is converted to grid row/column:

```
row = (lat_max - pixel_lat) / GRID_RES
col = (pixel_lon - lon_min) / GRID_RES
```

Multiple source pixels may map to the same grid cell. When this happens, the **last pixel written wins** — since files are processed chronologically, overlapping areas show the most recent observation.

### Why ~28 m Instead of Native 8 m?

| Resolution | Grid size (Flight 04) | Tradeoff |
|-----------|----------------------|----------|
| 0.00007° (~8 m, native) | ~2200 × 5600 = 12M cells | Full detail, slower processing |
| **0.00025° (~28 m, default)** | **~730 × 1120 = 817K cells** | **Good detail, fast** |
| 0.001° (~111 m) | ~180 × 280 = 50K cells | Quick preview, blurry |

The default is 3× the native resolution — fine enough to resolve fire features (which span hundreds of meters) while keeping processing fast for 40-file flights.

To change the resolution, edit `GRID_RES` at the top of `mosaic_flight.py`:

```python
GRID_RES = 0.00025  # [degrees] ≈ 28 m at 36°N
```

## Fire Detection in the Mosaic

The mosaic uses a **simplified fire detection** (absolute threshold only, no contextual test) for speed when processing many files:

- A pixel is flagged as fire if **both** conditions are met:
  - **T4 > threshold**: Brightness temperature at 3.9 μm exceeds 325 K (52°C) for daytime flights, or 310 K (37°C) for nighttime flights
  - **ΔT > 10 K**: The difference T4 − T11 exceeds 10 K, confirming the spectral signature of fire rather than just warm ground

### Why Different Day/Night Thresholds?

| | Daytime | Nighttime |
|---|---------|-----------|
| **Threshold** | 325 K (52°C) | 310 K (37°C) |
| **Background T4** | 290–320 K | 260–290 K |
| **Reason** | Solar heating warms bare rock/soil to 310–320 K, so the threshold must be above that | No solar heating; background cools significantly, so a lower threshold still catches fire without false positives |

These values come from the MODIS MOD14 Collection 6 fire algorithm (Giglio et al., 2016).

### Fire Compositing Across Overlapping Lines

When multiple flight lines cover the same area:
- **T4 and T11 grids**: Later observations overwrite earlier ones (most recent state)
- **Fire mask**: Uses OR logic — if fire was detected in **any** pass, the grid cell is marked as fire

This means the fire mask shows the **cumulative fire extent** across all passes of the flight, while the temperature grids show the snapshot from the last pass.

## Assumptions and Limitations

1. **Flat-grid approximation**: The grid uses equirectangular projection (treating degrees as equal-spaced). At 36°N over a ~0.3° span, the distortion is negligible (<0.5%).

2. **Nearest-neighbor resampling**: Each source pixel maps to the single nearest grid cell. No interpolation or anti-aliasing is applied. At 28 m resolution with 8 m source pixels, this means ~3-4 source pixels may compete for each grid cell, with only the last one kept.

3. **No atmospheric correction**: Brightness temperatures are derived from top-of-atmosphere radiance. For fire detection this is acceptable because fire signals are so large that atmospheric effects are minor by comparison.

4. **Simplified fire detection**: The mosaic uses only the absolute threshold test (not the contextual anomaly test from `detect_fire.py`) for processing speed. This may miss smaller/cooler fires that would be caught by the contextual test.

5. **Temporal compositing**: The "last write wins" approach means overlapping areas show the most recent observation. For a fire that is growing, the edges of the fire in earlier passes may be overwritten by cooler background from later passes if the aircraft flew a different pattern.

6. **Day/night classification**: Applied per-flight (not per-pixel) based on the `day_night_flag` global attribute. Flight 24-801-05 spans dusk into night; all its pixels use the nighttime threshold.

## Output

One PNG per flight:

| File | Flight | Content |
|------|--------|---------|
| `mosaic_flight_2480103.png` | 24-801-03 | Pre-burn, 9 lines, no fire |
| `mosaic_flight_2480104.png` | 24-801-04 | First fire, 40 lines, ~3,900 fire cells |
| `mosaic_flight_2480105.png` | 24-801-05 | Night fire, 16 lines, ~2,200 fire cells |
| `mosaic_flight_2480106.png` | 24-801-06 | Third fire, 14 lines, ~4,500 fire cells |

Each PNG contains two panels:
- **Left**: T4 brightness temperature (3.9 μm) in the `inferno` colormap [K]
- **Right**: Grayscale T4 background with red fire pixel overlay

## Usage

```bash
python mosaic_flight.py
```

Processes all HDF4 files in `ignite_fire_data/`, grouped by flight. Console output shows progress and summary statistics for each flight.

## Units Reference

| Variable | Units | Notes |
|----------|-------|-------|
| `GRID_RES` | degrees | 0.00025° ≈ 28 m at 36°N |
| `T4`, `T11` | Kelvin (K) | Brightness temperature; subtract 273.15 for °C |
| `T4_thresh` | Kelvin (K) | 325 K day / 310 K night |
| `dT_thresh` | Kelvin (K) | Minimum T4−T11 = 10 K |
| Radiance (input) | W/m²/sr/μm | Watts per square meter per steradian per micrometer |
| Latitude/Longitude | degrees | WGS84; lat: -90 to +90, lon: -180 to +180 |
| `buf` | degrees | Grid buffer = 0.005° ≈ 550 m |

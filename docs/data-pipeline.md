# Data Processing Pipeline

End-to-end documentation of how raw MASTER HDF files become ML-ready
features for fire detection.

---

## 1. Raw Data Ingestion

**Source:** `lib/io.py` &rarr; `process_file()`

### 1.1 MASTER Channels Extracted

The system reads MASTER (Multispectral Thermal Imager) L1B HDF4 files.
Five of the 50 channels are used:

| Name | Index | Wavelength | Band | Units After Processing |
|------|-------|-----------|------|----------------------|
| T4   | 30 (Ch 31) | 3.903 &mu;m | MWIR | Brightness temperature [K] |
| T11  | 47 (Ch 48) | 11.327 &mu;m | TIR  | Brightness temperature [K] |
| SWIR | 21 (Ch 22) | 2.162 &mu;m | SWIR | Radiance [W/m&sup2;/sr/&mu;m] |
| Red  | 4  (Ch 5)  | 0.654 &mu;m | VNIR | Radiance [W/m&sup2;/sr/&mu;m] |
| NIR  | 8  (Ch 9)  | 0.866 &mu;m | VNIR | Radiance [W/m&sup2;/sr/&mu;m] |

Channel indices and physical constants are defined in `lib/constants.py`.

### 1.2 Processing Steps (`process_file`)

1. **Read calibrated data** from `CalibratedData` HDF dataset.
2. **Apply scale factors** — raw integer values &times; per-channel `scale_factor`.
3. **Mask fill values** — values &lt; 0 (including -999 fills) &rarr; NaN.
4. **Read coordinates** — `PixelLatitude`, `PixelLongitude`; -999 &rarr; NaN.
5. **Read calibration metadata** — `EffectiveCentralWavelength_IR_bands`,
   `TemperatureCorrectionSlope`, `TemperatureCorrectionIntercept`.
6. **Radiance &rarr; brightness temperature** for T4 and T11 via inverse Planck
   (see &sect;1.3), then apply temperature calibration:
   `T_cal = slope &times; T + intercept`.
7. **Keep VNIR bands as radiance** — SWIR, Red, NIR are reflected solar, not
   thermal emission, so BT conversion does not apply.
8. **Compute NDVI** — `compute_ndvi(Red, NIR)` from `lib/vegetation.py`.

**Returns** a dict:

```python
{'T4': ndarray, 'T11': ndarray, 'SWIR': ndarray,
 'Red': ndarray, 'NIR': ndarray, 'NDVI': ndarray,
 'lat': ndarray, 'lon': ndarray}
# Each array has shape (scanlines, 716)
```

### 1.3 Inverse Planck Function (`radiance_to_bt`)

Converts spectral radiance to brightness temperature:

```
Constants:
  c1 = 2 * h * c^2          (first radiation constant)
  c2 = h * c / k_B          (second radiation constant)

Steps:
  1. lambda [um] -> [m]
  2. L [W/m^2/sr/um] -> [W/m^2/sr/m]
  3. T = c2 / lambda / ln(1 + c1 / (lambda^5 * L))
```

Physical constants from `lib/constants.py`:
- `H_PLANCK = 6.62607015e-34` J&middot;s
- `C_LIGHT  = 2.99792458e8` m/s
- `K_BOLTZ  = 1.380649e-23` J/K

---

## 2. Flight Organization

**Source:** `lib/io.py`

### `group_files_by_flight(data_dir)`

1. Scans `data_dir` for `*.hdf` files (sorted alphabetically = chronological).
2. Reads HDF attributes per file: `FlightNumber`, `FlightComment`, `day_night_flag`.
3. Groups files by flight number.
4. Returns `dict[flight_num] -> {files: list, comment: str, day_night: str}`.

### `compute_grid_extent(files)`

1. Reads corner-coordinate attributes (`lat_LL`, `lat_UL`, `lat_LR`, `lat_UR`,
   `lon_UL`, `lon_UR`, `lon_LL`, `lon_LR`) from each file.
2. Computes bounding box: `(lat_min, lat_max, lon_min, lon_max)`.
3. Adds 0.005&deg; buffer (~550 m) on all sides.

---

## 3. Gridding & Mosaicking

**Source:** `lib/mosaic.py`

Grid resolution: `GRID_RES = 0.00025°` &asymp; 28 m at 36&deg;N latitude
(native MASTER ~8 m, ~3&times; downsampled for speed).

### 3.1 Grid State Initialization (`init_grid_state`)

Two modes:

- **Empty sentinel** (no args): creates 0&times;0 arrays for real-time processing.
  The grid auto-initializes on the first sweep.
- **Explicit bounds** (lat/lon min/max): pre-allocates arrays of shape
  `(nrows, ncols)` for batch processing.

Grid state (`gs`) contains:

| Array | Type | Fill | Purpose |
|-------|------|------|---------|
| T4, T11, SWIR, Red, NIR | float32 | NaN | Latest pixel values |
| fire_count | int32 | 0 | Cumulative fire detections |
| obs_count | int32 | 0 | Cumulative observations |
| NDVI_baseline | float32 | NaN | First valid daytime NDVI (write-once) |
| veg_confirmed | bool | False | Vegetation loss confirmed |
| T4_max, dT_max, SWIR_max | float32 | NaN | Running max accumulators |
| T4_sum, T11_sum, SWIR_sum | float64 | 0 | Running sum accumulators |
| Red_sum, NIR_sum | float64 | 0 | Running sum accumulators |
| NDVI_min | float32 | NaN | Running min accumulator |
| NDVI_sum | float64 | 0 | Running sum for NDVI |
| NDVI_obs | int32 | 0 | Daytime NDVI observation count |

### 3.2 Dynamic Grid Expansion (`_expand_grid`)

When a new sweep extends beyond current grid bounds:

1. Compute new dimensions from expanded lat/lon range.
2. Calculate row/col offset of old grid within new grid.
3. Allocate new (larger) arrays; copy old data at offset.
4. Update `lat_axis`, `lon_axis` metadata.

This allows real-time processing without knowing the full flight extent
in advance.

### 3.3 Per-Sweep Processing (`process_sweep`)

Called once per HDF file in chronological order. Modifies `gs` in-place.

**Steps:**

1. **Initialize or expand grid** from file corner coordinates.
2. **Read file** via `process_file()`.
3. **Auto-detect day/night** — checks median NIR radiance > 5.0 W/m&sup2;/sr/&mu;m.
4. **Fire detection** — `detect_fire_simple(T4, T11)`:
   - Daytime: `T4 > 325 K AND T4 - T11 > 10 K`
   - Nighttime: `T4 > 310 K AND T4 - T11 > 10 K`
5. **Grid indexing** — map pixel (lat, lon) to grid (row, col); discard
   out-of-bounds.
6. **Update latest values** — T4, T11, SWIR at each grid cell.
7. **Update running accumulators** — max (T4_max, dT_max, SWIR_max) and
   sums (T4_sum, T11_sum, SWIR_sum, Red_sum, NIR_sum) for ML features.
8. **NDVI processing** (daytime only) — update NDVI_min, NDVI_sum, NDVI_obs;
   set write-once NDVI_baseline on first valid daytime observation.
9. **Vegetation loss detection**:
   - Daytime: fire + NDVI drop &ge; 0.15 from baseline.
   - Nighttime: any fire with existing baseline.
   - Either confirms `veg_confirmed = True`.
10. **VNIR update strategy**:
    - Normal pixels: keep best-illuminated (highest NIR reflectance).
    - Veg-confirmed pixels: take latest observation (to show burn scar).
11. **Accumulator reset** — where vegetation is confirmed AND fire count is 0,
    reset all ML accumulators (fire has passed).

**Returns:** `(n_new_fire, day_night)`.

### 3.4 Multi-Pass Consistency Filter (`get_fire_mask`)

Applied after all sweeps are processed:

```
For each pixel:
  if obs_count >= 2:
      fire = (fire_count >= 2)        # require fire in >= 2 passes
  else:
      fire = (fire_count >= 1)        # single-pass: accept any detection

  # Vegetation loss overrides
  if veg_confirmed AND fire_count >= 1:
      fire = True                     # independent confirmation
```

This reduced pre-burn false positives from ~135 to ~65 (**52% reduction**).

---

## 4. Pixel Table Construction

**Source:** `lib/stats.py` &rarr; `build_pixel_table()`

Creates a per-pixel DataFrame from all files in a flight:

1. For each HDF file:
   - `process_file()` &rarr; pixel data.
   - `detect_fire_simple()` &rarr; per-pixel fire boolean.
   - Compute `dT = T4 - T11`.
   - Map (lat, lon) to grid indices; discard out-of-bounds.
   - Append one row per valid pixel.
2. Concatenate all rows.

**Output columns:** `flight, file, lat, lon, T4, T11, dT, SWIR, Red, NIR, NDVI, fire`.

Grid cells observed by multiple flight lines appear as **multiple rows**,
preserving all temporal observations.

---

## 5. Feature Engineering

**Source:** `lib/features.py` &rarr; `build_location_features()`

Groups pixel table by `(lat, lon)` and computes **12 aggregate features**
per grid-cell location.

| # | Feature | Formula | Physical Meaning |
|---|---------|---------|-----------------|
| 1 | T4_max | `max(T4)` | Peak fire-channel temperature (fire intensity) |
| 2 | T4_mean | `mean(T4)` | Average thermal state (normalizes peak) |
| 3 | T11_mean | `mean(T11)` | Stable background temperature reference |
| 4 | dT_max | `max(T4 - T11)` | Strongest thermal contrast (core fire signature) |
| 5 | SWIR_max | `max(SWIR)` | Peak 2.2 &mu;m radiance (definitive at night) |
| 6 | SWIR_mean | `mean(SWIR)` | Average SWIR (normalizes peak) |
| 7 | Red_mean | `mean(Red)` | Implicit day/night indicator (~0 at night) |
| 8 | NIR_mean | `mean(NIR)` | Implicit day/night indicator (~0 at night) |
| 9 | NDVI_min | `min(NDVI)` | Burn scar indicator (NaN &rarr; 0 for night) |
| 10 | NDVI_mean | `mean(NDVI)` | Average vegetation (NaN &rarr; 0 for night) |
| 11 | NDVI_drop | `first(NDVI) - min(NDVI)` | Temporal vegetation loss (0 if &lt; 2 obs) |
| 12 | obs_count | `count(T4)` | Number of observations (reliability indicator) |

**Label:** `y = 1` if **any** observation at that location detected fire; else 0.

**Output:** `X (N_locations, 12)`, `y (N_locations,)`, `lats`, `lons`.

Feature selection is based on MODIS fire detection literature (Giglio et
al. 2003, Schroeder et al. 2014).

---

## 6. Train/Test Split

**Source:** `lib/training.py` &rarr; `extract_train_test()`

### Strategy

1. **Ground truth flight** (24-801-03, pre-burn) is split 80/20 train/test.
   All GT labels are **forced to 0** (no actual fire was present).
2. **Training set** = 80% GT + specified burn flights.
3. **Test set** = 20% GT + specified burn flights.
4. **Importance weights** computed per sample (see &sect;6.1).

### 6.1 Pixel-Wise Importance Weighting

**Source:** `lib/losses.py` &rarr; `compute_pixel_weights()`

Three sample categories with configurable importance multipliers:

| Category | Condition | Default Importance | Purpose |
|----------|-----------|-------------------|---------|
| GT | `flight == '24-801-03'` | 10.0 | Penalize FP on known no-fire data |
| Fire | `y == 1 AND NOT GT` | 5.0 | Penalize FN on confirmed fire pixels |
| Other | `y == 0 AND NOT GT` | 1.0 | Baseline for uncertain non-fire |

**Formula:**

```
w_i = importance_category * (N_total / N_category)    # importance x inverse-frequency
w_normalized = w / mean(w)                             # normalize to mean=1
```

The inverse-frequency term corrects for class imbalance; the importance
term encodes domain knowledge about sample reliability.

---

## 7. Oversampling

**Source:** `lib/training.py` &rarr; `oversample_minority()`

Balances fire vs. non-fire samples in the training set.

### Algorithm

```
Input:  X (N, 12), y (N,), w (N,), ratio (default 1.0)

1. Count:  n_fire = sum(y == 1),  n_nofire = sum(y == 0)
2. If n_fire >= n_nofire * ratio: return unchanged (already balanced)
3. target_fire = floor(n_nofire * ratio)
4. n_to_add = target_fire - n_fire
5. Randomly select n_to_add indices from fire samples (with replacement, seed=42)
6. Concatenate: X_bal = [X, X[repeat_idx]]  (same for y, w)
7. Re-normalize weights: w_bal = w_bal / mean(w_bal)
8. Shuffle all rows (seed=42)
9. Return X_bal, y_bal, w_bal
```

### Key Details

- **Fixed seed (42)** for reproducibility.
- **With replacement** — fire samples may be duplicated multiple times.
- **Weights preserved** — duplicated samples carry their original weights;
  re-normalization to mean=1 keeps gradient scale stable.
- **Shuffle** — prevents the model from seeing all duplicates in sequence.

### Example

```
Before:  1000 non-fire + 50 fire,  ratio=1.0
Target:  1000 fire samples needed
Add:     950 randomly duplicated fire samples
After:   1000 non-fire + 1000 fire = 2000 total (shuffled, weights re-normalized)
```

---

## 8. Feature Normalization (Hybrid)

After oversampling, features are normalized using a **hybrid** approach
that combines physics-based and statistical normalization:

### 8.1 Thermal Features (indices 0–3)

Features: `T4_max`, `T4_mean`, `T11_mean`, `dT_max`.

Divided by the piloted ignition temperature of dry wood:

```
T_norm = T / T_ignition    where T_ignition = 573.15 K (300 °C)
```

**Rationale:** Fire detection is fundamentally about whether temperatures
approach or exceed ignition. Normalizing by T_ignition gives values
physical meaning: a value of 1.0 means the pixel is at ignition
temperature. This is more interpretable and stable than z-scoring
temperatures, which would depend on the ambient temperature distribution
of the baseline flight.

The T_ignition value is configurable in YAML configs (`T_ignition` key,
in °C). Code converts to Kelvin internally: `T_K = T_C + 273.15`.

### 8.2 Non-Thermal Features (indices 4–11)

Features: `SWIR_max`, `SWIR_mean`, `Red_mean`, `NIR_mean`, `NDVI_min`,
`NDVI_mean`, `NDVI_drop`, `obs_count`.

Normalized via `sklearn.StandardScaler`:

1. **Fit on GT flight non-thermal features only** (pre-burn baseline).
2. **Transform** non-thermal columns of all data (train and test).
3. NaN values replaced with 0.0 before scaling.

This ensures SWIR, VNIR, NDVI, and observation count features are
z-scored against the pre-burn baseline, so fire-affected values deviate
upward from normal.

### 8.3 Constants

Defined in `lib/constants.py`:

```python
T_IGNITION_DRY_WOOD = 573.15       # [K]
THERMAL_FEATURE_INDICES = [0, 1, 2, 3]
NON_THERMAL_FEATURE_INDICES = [4, 5, 6, 7, 8, 9, 10, 11]
```

---

## 9. End-to-End Flow

```
HDF Files (61 files, 4 flights)
       |
       v
 [process_file]  per-pixel: T4, T11, SWIR, Red, NIR, NDVI, lat, lon
       |
       v
 [build_pixel_table]  per-pixel DataFrame (all observations preserved)
       |
       v
 [build_location_features]  group by (lat,lon) -> 12 features + label
       |
       v
 [load_all_data]  repeat for all flights -> FlightFeatures dict
       |
       v
 [extract_train_test]  split GT 80/20, add burn flights, compute weights
       |
       v
 [oversample_minority]  duplicate fire samples to balance classes
       |
       v
 [Hybrid normalize]  thermal / T_ignition; non-thermal via StandardScaler(GT)
       |
       v
 ML-ready: X_train (N, 12), y_train (N,), w_train (N,)
```

### Real-Time Alternative Path

For real-time processing (`realtime_fire.py`), features are computed
directly from grid-state accumulators without a DataFrame:

```
HDF Files (streamed one at a time)
       |
       v
 [process_sweep]  update gs accumulators in-place
       |
       v
 [compute_aggregate_features]  extract 12 features from gs (lib/fire.py)
       |
       v
 [_MLFireDetector.predict_from_gs]  normalize + MLP inference
       |
       v
 Fire probability grid (nrows x ncols)
```

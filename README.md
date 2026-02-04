# Fire-Detection
Team Flaming Kitty's AI Fire Detection code! 🔥🐱

## Table of Contents
- [What Is This Data?](#what-is-this-data)
- [How the Instrument Works](#how-the-instrument-works)
- [Dataset Parameters](#dataset-parameters)
- [Key Equations](#key-equations)
- [How Fire Detection Works](#how-fire-detection-works)
- [Examples: Fire vs No Fire](#examples-fire-vs-no-fire)
- [Running the Code](#running-the-code)

---

## What Is This Data?

This project uses data from the **MASTER** instrument — the **MODIS/ASTER Airborne Simulator**. MASTER is a thermal imaging sensor flown on NASA aircraft that captures images of the ground in 50 different wavelengths (called "channels" or "bands"), ranging from visible light (what your eyes see) all the way to thermal infrared (heat radiation).

The data comes from the **FireSense 2023** campaign, where NASA flew over a **prescribed burn** (a controlled, intentional fire) on the Kaibab Plateau in Arizona during October 18-20, 2023.

### Why 50 channels?

A normal camera captures 3 colors (red, green, blue). MASTER captures 50 "colors" spanning far beyond what the human eye can see. This matters for fire detection because **fire emits very strongly at wavelengths around 3-4 micrometers (μm)** — invisible to the human eye, but clearly visible to MASTER's infrared detectors.

### Flights in the Dataset

| Flight | Date | Description | Day/Night | Files | Fire Present? |
|--------|------|-------------|-----------|-------|---------------|
| 24-801-03 | Oct 18, 2023 | Pre-burn data collection | Day | 9 | **No** — collected before the burn started |
| 24-801-04 | Oct 19, 2023 | First fire flight | Day | 40 | **Yes** — USFS Blowdown Prescribed Burn |
| 24-801-05 | Oct 20, 2023 | Second fire flight (nighttime) | Night | 16 | **Yes** — USFS Lakes Unit Prescribed Burn |
| 24-801-06 | Oct 20, 2023 | Third fire flight | Day | 18 | **Yes** — USFS Blowdown Prescribed Burn |

Each file covers one "flight line" — a single pass of the aircraft over the target area. The aircraft flew back and forth in parallel strips to cover the burn zone.

### File Naming Convention

```
MASTERL1B_2480104_20_20231019_2031_2033_V01.hdf
│         │       │  │        │    │    │
│         │       │  │        │    │    └─ Version
│         │       │  │        │    └────── End time (HHMM UTC)
│         │       │  │        └─────────── Start time (HHMM UTC)
│         │       │  └──────────────────── Date (YYYYMMDD)
│         │       └─────────────────────── Flight line number
│         └─────────────────────────────── Flight ID (248=year, 01=aircraft, 04=flight#)
└───────────────────────────────────────── Product: MASTER Level 1B
```

---

## How the Instrument Works

MASTER sits in the belly of a NASA B200 aircraft and points straight down. A spinning mirror sweeps the sensor's field of view from side to side (cross-track), building up an image line by line as the aircraft flies forward (along-track).

```
     Aircraft flying direction →
     ═══════════╦═══════════
                ║ MASTER sensor
                ║
         ╱─────╨─────╲       Scanning mirror sweeps
        ╱      │       ╲     left to right
       ╱       │        ╲
      ╱        │         ╲
    ─────────────────────────  Ground (716 pixels wide)
    ← pixel 0    pixel 715 →

    Each sweep = 1 "scanline"
    2736 scanlines per file (typical)
```

The result is a 2D image where:
- **Rows** = scanlines (time/along-track direction)
- **Columns** = pixels (cross-track, 716 pixels per line)
- **Depth** = 50 spectral channels

### Calibration

Before and after each scan, the mirror views two internal **blackbody references** — one cold (~10°C) and one warm (~39°C) — whose temperatures are precisely known. By comparing the signal from these references to the signal from the ground, the instrument converts raw detector counts into calibrated **radiance** (the actual amount of energy reaching the sensor). This is what is stored in the Level 1B (L1B) data.

![Blackbody Temperatures](plots/blackbody_temperatures.png)
*The two onboard blackbody references maintain stable temperatures throughout the flight, providing the calibration anchors for converting raw counts to physical radiance units.*

---

## Dataset Parameters

Each HDF4 file contains 38 datasets. Here are the important ones grouped by category:

### The Main Science Data

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `CalibratedData` | (2736, 50, 716) | W/m²/sr/μm | **The primary data product.** Calibrated spectral radiance for every pixel in every channel. This is the energy per unit area, per steradian (viewing angle), per micrometer of wavelength reaching the sensor. Has a per-channel `scale_factor` that must be multiplied in. |

### Geolocation — Where Each Pixel Is

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `PixelLatitude` | (2736, 716) | degrees | Latitude of each pixel on the ground. -90 (South Pole) to +90 (North Pole). |
| `PixelLongitude` | (2736, 716) | degrees | Longitude of each pixel. -180 (West) to +180 (East). |
| `PixelElevation` | (2736, 716) | meters MSL | Ground elevation at each pixel location. |

### Spectral Channel Information

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `Central100%ResponseWavelength` | (50,) | μm | The nominal center wavelength for each of the 50 channels. Ranges from 0.46 μm (blue visible) to 12.86 μm (thermal infrared). |
| `EffectiveCentralWavelength_IR_bands` | (50,) | μm | The **effective** center wavelength, accounting for the shape of each channel's spectral response. More accurate than the nominal value for converting radiance to temperature. Only valid for IR channels (26-50); filled with -99 for visible channels. |
| `Left50%ResponseWavelength` | (50,) | μm | The short-wavelength edge of each channel's spectral response (at 50% of peak). |
| `Right50%ResponseWavelength` | (50,) | μm | The long-wavelength edge. Together with the left edge, these define the bandwidth of each channel. |
| `SolarSpectralIrradiance` | (50,) | W/m²/μm | How much sunlight arrives at the top of the atmosphere in each channel's band. Used for reflectance calculations. |

### The 50 Channels at a Glance

| Channel Group | Channels | Wavelength Range | What They See |
|--------------|----------|-----------------|---------------|
| **VNIR** (Visible/Near-IR) | 1-11 | 0.46 – 0.95 μm | Reflected sunlight. Similar to what a camera sees, plus near-infrared. Useful for vegetation, clouds, land cover. |
| **SWIR** (Short-Wave IR) | 12-25 | 1.60 – 2.39 μm | Mix of reflected sunlight and thermal emission. Sensitive to minerals, soil moisture, and very hot fires. |
| **MWIR** (Mid-Wave IR) | 26-40 | 3.30 – 5.26 μm | **The fire detection sweet spot.** At these wavelengths, fire (600-1200 K) emits enormously more radiation than the cool background (~300 K). Channel 31 (3.915 μm) is the primary fire channel. |
| **TIR** (Thermal IR) | 41-50 | 7.83 – 12.86 μm | Pure thermal emission from the ground. Measures surface temperature regardless of sunlight. Channel 48 (11.25 μm) provides background temperature. |

### Viewing and Solar Geometry

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `SolarZenithAngle` | (2736, 716) | degrees | Angle from directly overhead to the sun. 0° = sun overhead, 90° = sun on horizon. Used to classify day (< 85°) vs night (≥ 85°). |
| `SolarAzimuthAngle` | (2736, 716) | degrees | Compass direction from each pixel to the sun. |
| `SensorZenithAngle` | (2736, 716) | degrees | Angle from straight down (nadir) to the sensor. Pixels at the edge of the swath have larger angles. |
| `SensorAzimuthAngle` | (2736, 716) | degrees | Compass direction from each pixel to the sensor. |

### Aircraft Navigation

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `AircraftLatitude` | (2736,) | degrees | Aircraft position (one value per scanline). |
| `AircraftLongitude` | (2736,) | degrees | |
| `AircraftAltitude` | (2736,) | meters MSL | Flight altitude. |
| `AircraftHeading` | (2736,) | degrees true | Compass heading. |
| `AircraftPitch` | (2736,) | degrees | Nose up/down angle. |

### Calibration Parameters

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `BlackBody1Temperature` | (2736,) | °C (×scale_factor) | Temperature of the cold calibration reference. |
| `BlackBody2Temperature` | (2736,) | °C (×scale_factor) | Temperature of the warm calibration reference. |
| `TBack` | (2736,) | Kelvin | Background/instrument temperature. |
| `CalibrationSlope` | (2736, 50) | W/m²/sr/μm per count | Gain: how much radiance per raw detector count. |
| `CalibrationIntercept` | (2736, 50) | W/m²/sr/μm | Offset in the calibration equation. |
| `TemperatureCorrectionSlope` | (50,) | unitless | Post-Planck brightness temperature correction (slope). See [equations](#temperature-correction) below. |
| `TemperatureCorrectionIntercept` | (50,) | Kelvin | Post-Planck brightness temperature correction (intercept). |

### Timing

| Dataset | Shape | Units | What It Is |
|---------|-------|-------|------------|
| `ScanlineTime` | (2736,) | hours (UTC) | Time of each scanline as decimal hours. |
| `YearMonthDay` | (2736,) | YYYYMMDD | Date stamp per scanline. |
| `GreenwichMeanTime` | (2736,) | — | GMT time reference. |

---

## Key Equations

### Blackbody Radiation and the Planck Function

Every object above absolute zero (0 K / -273.15°C) emits thermal radiation. The **Planck function** describes how much radiation a perfect emitter (a "blackbody") produces at each wavelength for a given temperature:

$$L(\lambda, T) = \frac{2hc^2}{\lambda^5} \cdot \frac{1}{e^{hc / \lambda k T} - 1}$$

Where:
| Symbol | Value | Meaning |
|--------|-------|---------|
| $L$ | — | Spectral radiance (W/m²/sr/m) |
| $\lambda$ | — | Wavelength (m) |
| $T$ | — | Temperature (K) |
| $h$ | 6.626 × 10⁻³⁴ J·s | Planck's constant |
| $c$ | 2.998 × 10⁸ m/s | Speed of light |
| $k$ | 1.381 × 10⁻²³ J/K | Boltzmann's constant |

### Why Fire Is Bright at 3.9 μm

The Planck function peaks at shorter wavelengths for hotter objects (**Wien's displacement law**: $\lambda_{peak} = 2898 / T$ μm). For a background surface at 300 K (~27°C), the peak emission is around 9.7 μm. For a fire at 800 K (~527°C), the peak shifts to 3.6 μm.

At 3.9 μm, the radiance from an 800 K fire is **orders of magnitude** greater than from a 300 K surface. At 11 μm, the difference is much smaller. This is why the 3.9 μm channel is the primary fire detection band — even a tiny sub-pixel fire produces a massive signal increase.

```
Radiance at 3.9 μm:
  300 K background:  ~0.26 W/m²/sr/μm
  800 K fire:        ~5,985 W/m²/sr/μm   ← 23,000× brighter!

Radiance at 11 μm:
  300 K background:  ~9.8 W/m²/sr/μm
  800 K fire:        ~41.2 W/m²/sr/μm    ← only 4× brighter
```

### Inverse Planck Function (Radiance → Brightness Temperature)

MASTER measures radiance. To convert it to **brightness temperature** (the temperature a blackbody would need to emit that radiance), we invert the Planck function:

$$T_B = \frac{hc}{\lambda k} \cdot \frac{1}{\ln\left(\frac{2hc^2}{\lambda^5 L} + 1\right)}$$

Or equivalently, using the radiation constants $c_1 = 2hc^2$ and $c_2 = hc/k$:

$$T_B = \frac{c_2}{\lambda \cdot \ln\left(\frac{c_1}{\lambda^5 L} + 1\right)}$$

Where:
| Constant | Value |
|----------|-------|
| $c_1$ | 1.191 × 10⁻¹⁶ W·m² |
| $c_2$ | 1.439 × 10⁻² m·K |

In Python:
```python
def radiance_to_brightness_temp(radiance_Wm2_sr_um, wavelength_um):
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 2.99792458e8    # Speed of light (m/s)
    k = 1.380649e-23    # Boltzmann constant (J/K)

    lam = wavelength_um * 1e-6            # convert μm → m
    L = radiance_Wm2_sr_um * 1e6          # convert W/m²/sr/μm → W/m²/sr/m

    c1 = 2.0 * h * c**2
    c2 = h * c / k

    T_b = c2 / lam / np.log(1.0 + c1 / (lam**5 * L))
    return T_b  # Kelvin
```

### Temperature Correction

Because each channel integrates over a range of wavelengths (not a single wavelength), the inverse Planck function introduces a small systematic error. MASTER provides correction coefficients to fix this:

$$T_{corrected} = \text{slope} \times T_{Planck} + \text{intercept}$$

The slope is very close to 1.0 (e.g., 0.9995) and the intercept is a fraction of a Kelvin (e.g., 0.30 K). These corrections are only meaningful for thermal/IR channels 26-50.

---

## How Fire Detection Works

Our algorithm is inspired by the **MODIS MOD14** active fire detection product (Giglio et al., 2016). It uses two complementary tests:

### Test 1: Absolute Threshold ("Is the temperature above X?")

A pixel is flagged as fire if:
- **Daytime** (solar zenith angle < 85°): $T_4 > 325\text{ K}$ (52°C)
- **Nighttime** (solar zenith angle ≥ 85°): $T_4 > 310\text{ K}$ (37°C)
- AND the temperature difference $\Delta T = T_4 - T_{11} > 10\text{ K}$

Where $T_4$ is brightness temperature at ~3.9 μm (Channel 31) and $T_{11}$ is at ~11.25 μm (Channel 48).

The $\Delta T$ requirement is critical — it distinguishes fire from surfaces that are simply warm (like sun-heated bare rock). Fire has a unique spectral signature: disproportionately high emission at 3.9 μm compared to 11 μm.

### Test 2: Contextual Anomaly ("Is this pixel hotter than its neighbors?")

A pixel is flagged as fire if:
- $T_4 > \bar{T_4} + 3\sigma_{T_4}$ (more than 3 standard deviations above the local mean)
- AND $\Delta T > \overline{\Delta T} + 3\sigma_{\Delta T}$ (delta-T also anomalous)
- AND $\Delta T > 10\text{ K}$ (minimum sanity check)

The local mean ($\bar{T}$) and standard deviation ($\sigma$) are computed over a 61×61 pixel window (~3 km) surrounding each pixel.

This test catches smaller or cooler fires that may not exceed the absolute threshold but are clearly anomalous compared to their surroundings.

### Combined Result

A pixel is classified as **fire** if it passes **either** test (union). This gives us two types of fire pixels:
- **Absolute detections** (red): intense fires exceeding the hard temperature threshold
- **Contextual detections** (orange): anomalously warm pixels relative to their local background

---

## Examples: Fire vs No Fire

### Pre-Burn (No Fire)

**File:** `MASTERL1B_2480103_05_20231018_1849_1853_V01.hdf`
**Flight:** 24-801-03 (October 18, 2023) — collected the day before the prescribed burn

```
T4  range: 225.1 - 445.4 K (-48.0 - 172.3°C)
T11 range: 276.7 - 358.1 K
ΔT  range: -65.8 - 138.4 K
Fire pixels: 455 (0.017% of image) — false positives from edge artifacts
```

![Pre-burn fire detection](plots/fire_detection_preburn.png)

**What you see:**
- **Top left (T4, 3.9 μm):** Warm and cool terrain visible. Bright streaks are exposed rock or bare soil reflecting sunlight. The temperature range is typical for unburned Arizona plateau in October.
- **Top right (T11, 11 μm):** Similar terrain pattern but with less contrast. This wavelength is less sensitive to reflected sunlight.
- **Bottom left (ΔT = T4 − T11):** Mostly uniform (values near 0-5 K). No strong anomalies — the 3.9 μm and 11 μm channels agree because there is no fire.
- **Bottom right (Fire overlay):** A few scattered red/orange dots (455 total, 0.017% of pixels). These are false positives caused by bright bare soil or edge-of-scan artifacts where the viewing angle is extreme. They are **not spatially clustered** — this is the key indicator that they are not real fire.

### Active Burn (Fire Present)

**File:** `MASTERL1B_2480104_20_20231019_2031_2033_V01.hdf`
**Flight:** 24-801-04 (October 19, 2023) — during the USFS Blowdown Prescribed Burn

```
T4  range: 255.9 - 800.0 K (-17.2 - 526.8°C)
T11 range: 114.9 - 416.8 K
ΔT  range: -65.2 - 472.9 K
Fire pixels: 8,650 (0.468% of image)
```

![Burn fire detection](plots/fire_detection_burn.png)

**What you see:**
- **Top left (T4, 3.9 μm):** Bright white/yellow features appear that were not present in the pre-burn image. These are fire pixels reaching up to 800 K (527°C) — the sensor may even be saturating on the hottest pixels.
- **Top right (T11, 11 μm):** The fire is visible but much less prominent. The hottest fire pixels only reach ~417 K at 11 μm vs 800 K at 3.9 μm. This wavelength difference is exactly what the fire detection algorithm exploits.
- **Bottom left (ΔT = T4 − T11):** The fire area lights up dramatically — ΔT values reaching 473 K. The background remains near 0-10 K. This is the spectral "fingerprint" of fire.
- **Bottom right (Fire overlay):** 8,650 fire pixels, clearly **clustered in a coherent spatial pattern** matching the shape of the burn area. Red = absolute threshold detections, orange = contextual anomaly detections.

### Georeferenced Fire Map

![Fire map on geographic coordinates](plots/fire_map_burn.png)

The fire pixels plotted on their actual geographic coordinates (latitude/longitude) on the Kaibab Plateau. The fire perimeter follows the terrain — you can see it tracing along ridgelines and valleys where the prescribed burn was conducted.

### Side-by-Side Comparison

![Pre-burn vs burn comparison](plots/fire_comparison.png)

The comparison makes it clear: the pre-burn scene (left) shows no coherent fire pattern, while the burn scene (right) shows a dense cluster of fire detections.

### Multi-Channel Overview

![Radiance across all spectral regions](plots/radiance_overview.png)

This shows the same scene across 6 different channels spanning the full spectrum (VNIR through TIR), demonstrating how different wavelengths reveal different surface features.

---

## Running the Code

### Requirements

```
pip install pyhdf numpy matplotlib
```

Note: `pyhdf` may require HDF4 libraries. On macOS with conda/mamba:
```
conda install -c conda-forge pyhdf
```

### Scripts

| Script | Purpose | Documentation |
|--------|---------|---------------|
| `plotdata.py` | Explore the HDF files — plots radiance across channels and a georeferenced thermal image | — |
| `detect_fire.py` | Run fire detection — compares a pre-burn file to a burn file, produces detection maps | — |
| `mosaic_flight.py` | Assemble all flight lines into a single georeferenced mosaic per flight | [mosaic_flight.md](mosaic_flight.md) |
| `plot_burn_locations.py` | Per-flight 2x2 analysis: burn locations, T4, T11, and detection space scatter | — |

```bash
python detect_fire.py
```

This produces (in `plots/`):
- `fire_detection_preburn.png` — 4-panel analysis of pre-burn scene
- `fire_detection_burn.png` — 4-panel analysis of burn scene
- `fire_map_burn.png` — georeferenced fire pixel map
- `fire_comparison.png` — side-by-side comparison

```bash
python mosaic_flight.py
```

This produces one mosaic per flight in `plots/` (see [mosaic_flight.md](mosaic_flight.md) for details):
- `mosaic_flight_2480103.png` — Pre-burn, 9 lines composited
- `mosaic_flight_2480104.png` — First fire flight, 40 lines composited
- `mosaic_flight_2480105.png` — Night fire flight, 16 lines composited
- `mosaic_flight_2480106.png` — Third fire flight, 14 lines composited

```bash
python plot_burn_locations.py
```

This produces one PNG per flight in `plots/`, each with a 2x2 layout (burn locations, T4, T11, detection space scatter):
- `burn_locations_2480103.png` — Pre-burn false positive analysis
- `burn_locations_2480104.png` — First fire flight (Blowdown)
- `burn_locations_2480105.png` — Night fire flight (Lakes Unit)
- `burn_locations_2480106.png` — Third fire flight (Blowdown)

### Data

The HDF4 files should be placed in `ignite_fire_data/`. They are MASTER Level 1B files from the FireSense 2023 campaign, publicly available from the [NASA ASAP Data Archive](https://asapdata.arc.nasa.gov/sensors/master/). The specific dataset used is the **MASTER Level 1B** product from flights 24-801-03 through 24-801-06 (October 18-20, 2023, Kaibab National Forest, Arizona).

All plots are saved to the `plots/` directory.

---

## References

- Giglio, L., Schroeder, W., & Justice, C. O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31-41.
- [MASTER Instrument Overview](https://asapdata.arc.nasa.gov/sensors/master/)
- [MODIS Active Fire Product](https://modis.gsfc.nasa.gov/data/dataprod/mod14.php)

Science and Algorithm Guide
===========================

This document describes every physical assumption, threshold, and
algorithm used in the fire detection pipeline. For running instructions,
see :doc:`operators-guide`.


Instrument Overview
-------------------

The **MASTER** (MODIS/ASTER Airborne Simulator) instrument is a
50-channel thermal imaging sensor flown on NASA's B200 aircraft. It
captures images from visible light (0.46 :math:`\mu`\ m) through thermal
infrared (12.87 :math:`\mu`\ m).

**Scanning geometry:** A spinning mirror sweeps the sensor's field of
view cross-track (716 pixels wide) while the aircraft flies forward,
building an image line by line (~2736 scanlines per file).

::

        Aircraft flying direction -->
        ============+============
                    | MASTER sensor
                    |
             /------+------\       Scanning mirror sweeps
            /       |       \      left to right
           /        |        \
          /         |         \
        -------------------------  Ground (716 pixels wide)
        <-- pixel 0   pixel 715 -->

        Each sweep = 1 "scanline"
        2736 scanlines per file (typical)

**Native resolution:** ~8 m at nadir.

**Calibration:** Dual onboard blackbody references (cold ~10 C, warm
~39 C) convert raw detector counts to calibrated **radiance**
(W/m\ :sup:`2`/sr/:math:`\mu`\ m) stored in Level 1B (L1B) data.


Channel Selection
-----------------

Of the 50 MASTER channels, this project uses 5:

.. list-table::
   :header-rows: 1
   :widths: 10 10 15 15 50

   * - Variable
     - Channel
     - Wavelength
     - Band
     - Role
   * - T4
     - 31 (index 30)
     - 3.903 :math:`\mu`\ m
     - MWIR
     - **Primary fire detection.** Fire (600--1200 K) is ~23,000x brighter
       than background (~300 K) at this wavelength.
   * - T11
     - 48 (index 47)
     - 11.327 :math:`\mu`\ m
     - TIR
     - **Background temperature.** Fire is only ~4x brighter here.
       The large contrast ratio at 3.9 :math:`\mu`\ m vs small ratio at
       11 :math:`\mu`\ m is the spectral fingerprint of fire.
   * - SWIR
     - 22 (index 21)
     - 2.162 :math:`\mu`\ m
     - SWIR
     - **Solar reflection channel.** High SWIR + high T4 suggests sun
       glint (false positive). High T4 + low SWIR suggests fire. Kept
       as radiance (not brightness temperature).
   * - Red
     - 5 (index 4)
     - 0.654 :math:`\mu`\ m
     - VNIR
     - **Visible red** for NDVI vegetation mapping. Kept as radiance.
   * - NIR
     - 9 (index 8)
     - 0.866 :math:`\mu`\ m
     - VNIR
     - **Near-infrared** for NDVI and sunlight detection. Kept as
       radiance.

**Why these channels?** The channel selection follows the MODIS MOD14
fire product heritage (Giglio et al., 2016). The T4/T11 pair exploits
the extreme radiance contrast of fire at 3.9 :math:`\mu`\ m vs the
small contrast at 11 :math:`\mu`\ m.


Radiometric Processing
----------------------

Inverse Planck Function
^^^^^^^^^^^^^^^^^^^^^^^

MASTER measures spectral radiance :math:`L` in W/m\ :sup:`2`/sr/:math:`\mu`\ m.
To convert to **brightness temperature** (the temperature a blackbody
would need to produce that radiance), we invert the Planck function:

.. math::

   T_B = \frac{c_2}{\lambda \cdot \ln\!\left(\frac{c_1}{\lambda^5 L} + 1\right)}

where:

.. list-table::
   :widths: 15 30 55

   * - :math:`c_1`
     - :math:`2hc^2 = 1.191 \times 10^{-16}` W m\ :sup:`2`
     - First radiation constant
   * - :math:`c_2`
     - :math:`hc/k = 1.439 \times 10^{-2}` m K
     - Second radiation constant
   * - :math:`\lambda`
     - wavelength in meters
     - Effective central wavelength from HDF metadata
   * - :math:`L`
     - radiance in W/m\ :sup:`2`/sr/m
     - Calibrated spectral radiance (converted from :math:`\mu`\ m units)

**Physical constants** (CODATA 2018, exact values):

- Planck constant: :math:`h = 6.62607015 \times 10^{-34}` J s
- Speed of light: :math:`c = 2.99792458 \times 10^8` m/s
- Boltzmann constant: :math:`k = 1.380649 \times 10^{-23}` J/K

**Unit conversions applied:**

- Wavelength: :math:`\mu`\ m :math:`\to` m (multiply by :math:`10^{-6}`)
- Radiance: W/m\ :sup:`2`/sr/:math:`\mu`\ m :math:`\to` W/m\ :sup:`2`/sr/m
  (multiply by :math:`10^6`)

See :func:`lib.io.radiance_to_bt`.

Temperature Correction
^^^^^^^^^^^^^^^^^^^^^^

Because each channel integrates over a range of wavelengths (not a
single wavelength), the inverse Planck function introduces a small
systematic error. MASTER provides per-channel correction coefficients:

.. math::

   T_\text{corrected} = \text{slope} \times T_\text{Planck} + \text{intercept}

The slope is very close to 1.0 (e.g., 0.9995) and the intercept is a
fraction of a Kelvin (e.g., 0.30 K). These are stored in the HDF
attributes ``TemperatureCorrectionSlope`` and
``TemperatureCorrectionIntercept``.


Fire Detection: Absolute Threshold
-----------------------------------

A pixel is flagged as fire if both conditions hold:

1. **T4 exceeds a brightness temperature threshold:**

   - Daytime: :math:`T_4 > 325` K (52 C)
   - Nighttime: :math:`T_4 > 310` K (37 C)

2. **The spectral difference exceeds a minimum:**

   .. math::

      \Delta T = T_4 - T_{11} > 10 \text{ K}

**Physical justification for** :math:`\Delta T`:

Fire has a unique spectral signature -- disproportionately high emission
at 3.9 :math:`\mu`\ m compared to 11 :math:`\mu`\ m. Surfaces that are
simply warm (sun-heated rock, bare soil) have similar brightness
temperatures at both wavelengths (:math:`\Delta T \approx 0\text{--}5` K),
while fire produces :math:`\Delta T > 100` K for intense burns.

**Why different day/night thresholds:** Solar heating warms surfaces to
310--320 K during daytime, so the daytime threshold (325 K) must be
higher to avoid flagging warm ground. At night, background cools to
260--290 K, allowing a lower threshold (310 K) to catch smaller fires.

See :func:`lib.fire.detect_fire_simple`.


Fire Detection: Contextual Anomaly
------------------------------------

A pixel is flagged as fire if it is anomalous relative to its local
background:

.. math::

   T_4 > \bar{T}_4 + n\sigma_{T_4} \quad \text{AND} \quad
   \Delta T > \overline{\Delta T} + n\sigma_{\Delta T} \quad \text{AND} \quad
   \Delta T > 10 \text{ K}

**Parameters:**

.. list-table::
   :widths: 20 15 65

   * - Parameter
     - Value
     - Rationale
   * - Window size
     - 61 x 61 pixels
     - ~1.7 km at grid resolution. Large enough to capture local
       background statistics, small enough to respond to spatial variation.
   * - :math:`n` (sigma multiplier)
     - 3.0
     - Standard 3-sigma threshold for statistical anomaly detection.
   * - :math:`\Delta T` floor
     - 10 K
     - Minimum sanity check regardless of local statistics.

**Implementation:** NaN-aware cumulative-sum box filter using 2D
summed-area tables. Reflection-padded at boundaries to avoid
window-size edge artifacts.

The **combined** fire mask is the union of absolute and contextual
detections.

See :func:`lib.fire.detect_fire`, :func:`lib.fire._contextual_stats`.


Multi-Pass Consistency Filter
-----------------------------

**Problem:** Solar reflection false positives are **angle-dependent** --
they trigger in one pass but not others. Real fire emission is
**isotropic** and triggers consistently.

**Solution:** Track per-grid-cell counts across all flight lines:

- ``obs_count``: number of valid observations
- ``fire_count``: number of fire detections

**Decision rule:**

.. math::

   \text{fire} =
   \begin{cases}
   \text{fire\_count} \geq 2 & \text{if obs\_count} \geq 2 \\
   \text{fire\_count} \geq 1 & \text{if obs\_count} = 1
   \end{cases}

Pixels observed multiple times must be detected as fire in at least 2
passes. Single-observation pixels keep their detection (no multi-pass
information available to filter).

**Results:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Flight
     - Before (OR)
     - After (filter)
     - Eliminated
     - Reduction
   * - 03 (pre-burn)
     - 135 FP
     - 65 FP
     - 70
     - **52%**
   * - 04 (day burn)
     - ~3,100
     - 3,064
     - ~36
     - ~1%
   * - 05 (night burn)
     - ~1,730
     - 1,712
     - ~18
     - ~1%
   * - 06 (day burn)
     - ~3,320
     - 3,305
     - ~15
     - <1%

Real fire detections are >99% multi-pass confirmed, validating the
physics: fire emission is isotropic.

See :func:`lib.mosaic.build_mosaic`, :func:`lib.mosaic.get_fire_mask`.


SWIR for False Positive Discrimination
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Source
     - T4 (3.9 :math:`\mu`\ m)
     - SWIR (2.16 :math:`\mu`\ m)
     - Reason
   * - **Fire**
     - Very high (thermal)
     - Low-to-moderate
     - Fire emits thermally; much less at 2.16 :math:`\mu`\ m than 3.9
   * - **Sun-heated rock**
     - Elevated (reflected)
     - **High** (reflected)
     - Rock reflects sunlight across the solar spectrum
   * - **Background**
     - Low (~300 K)
     - Moderate
     - Normal terrain

At night, SWIR is near-zero (no sunlight) so it provides no
discrimination -- but nighttime flights already have fewer false
positives because there is no solar reflection.

SWIR is extracted from Channel 22 (2.162 :math:`\mu`\ m, index 21) and
kept as radiance (not converted to brightness temperature).


Day/Night Classification
------------------------

Two methods are available:

1. **Solar Zenith Angle (SZA):** :math:`\text{SZA} < 85^\circ` = daytime.
   Matches the MODIS MOD14 convention. See :func:`lib.fire.is_daytime`.

2. **VNIR radiance-based (preferred):** Check if median NIR radiance
   exceeds a threshold:

   .. math::

      \text{sunlight} = \text{median}(\text{NIR}_\text{valid}) > 5.0 \text{ W/m}^2\text{/sr/}\mu\text{m}

   **Why this is more robust than SZA:** SZA only checks geometric sun
   position. Cloud cover blocks solar signal even when the sun is above
   the horizon. The VNIR approach detects actual illumination at the
   sensor.

   **Empirical calibration from MASTER data:**

   .. list-table::
      :widths: 25 25 25 25

      * - Condition
        - NIR median
        - NIR p95
        - NIR min
      * - Daytime
        - ~40
        - --
        - ~7 W/m\ :sup:`2`/sr/:math:`\mu`\ m
      * - Nighttime
        - ~0.2
        - ~0.5
        - --

   The threshold of 5.0 sits cleanly between nighttime noise (0.5)
   and daytime minimum (7).

See :func:`lib.vegetation.has_sunlight`.


Vegetation Mapping (NDVI)
-------------------------

NDVI (Normalized Difference Vegetation Index):

.. math::

   \text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}

**Key assumptions:**

- **Radiance, not reflectance:** The ratio cancels solar irradiance to
  first order (same sun angle, same atmospheric path), so NDVI from
  radiance closely approximates NDVI from reflectance.

- **Grid, then compute:** Red and NIR are gridded separately, then
  NDVI is computed post-gridding. This is necessary because NDVI is
  a nonlinear ratio -- averaging NDVI values directly would introduce
  bias.

- **Nighttime = NaN:** Reflected solar bands are meaningless at night
  (sensor noise only). NDVI is set to NaN for nighttime pixels.

Per-Pixel Best-Illumination Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple sweeps observe the same grid cell, the VNIR observation
with the **highest NIR radiance** is kept. This ensures:

- Daytime sweeps override nighttime sweeps (higher NIR).
- Cloud-free observations override cloudy ones (higher NIR).
- Partial cloud: per-pixel, not per-sweep, so clear pixels are
  preserved even when part of the sweep is cloudy.

Every sweep contributes VNIR data. The per-pixel NIR comparison
decides whether the update is applied -- nighttime sweeps have
near-zero NIR and will not overwrite good daytime data.

See :func:`lib.mosaic.process_sweep`, :func:`lib.vegetation.compute_ndvi`.


Mosaic Gridding
---------------

**Grid resolution:** 0.00025 degrees per cell (~28 m at 36 N).
This is ~3x downsampled from the native 8 m pixel spacing, chosen
for computational speed while preserving fire detection capability.

**Coordinate system:** Equirectangular (flat grid).
At 36 N over ~0.3 degree extent, distortion is <0.5%.

**Resampling:** Nearest-neighbor. Each source pixel maps to the single
nearest grid cell. No interpolation. "Last write wins" for thermal
channels (T4, T11, SWIR).

**Bounding box:** Computed from HDF corner attributes (``lat_UL``,
``lat_UR``, ``lat_LL``, ``lat_LR`` and equivalent longitude attributes)
plus a 0.005 degree (~550 m) buffer to account for georeferencing
uncertainty.

**Cell area computation:**

.. math::

   \Delta y = \text{GRID\_RES} \times 111{,}000 \text{ m/deg}

   \Delta x = \text{GRID\_RES} \times 111{,}000 \times \cos(\text{lat})

   A = \Delta x \times \Delta y

Uses a simple spherical model (111,000 m/degree average; WGS84 varies
110,574--111,320 m/degree). No ellipsoid correction.

See :func:`lib.mosaic.build_mosaic`, :func:`lib.mosaic.init_grid_state`,
:func:`lib.stats.compute_cell_area_m2`.


Connected Component Fire Zone Analysis
---------------------------------------

Fire zones are identified by 8-connectivity connected component labeling
(scipy.ndimage.label with a 3x3 ones structuring element). Each
connected region of fire pixels becomes a labeled zone.

Zones are sorted by pixel count (largest first) and annotated with
their centroid coordinates and estimated area
(:math:`\text{pixels} \times \text{cell area}`).

See :func:`lib.fire.detect_fire_zones`.


ML Fire Detection
-----------------

``fire_ml.py`` trains a neural network to classify fire vs non-fire
pixels.

Features
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 45

   * - Feature
     - Source
     - Units
     - Why
   * - T4
     - Channel 31 (3.9 :math:`\mu`\ m)
     - K
     - Primary fire signal
   * - T11
     - Channel 48 (11.3 :math:`\mu`\ m)
     - K
     - Background temperature
   * - :math:`\Delta T`
     - T4 - T11
     - K
     - Spectral fingerprint
   * - SWIR
     - Channel 22 (2.2 :math:`\mu`\ m)
     - W/m\ :sup:`2`/sr/:math:`\mu`\ m
     - Solar reflection discriminator

Architecture
^^^^^^^^^^^^

::

   Input (4) --> Linear(64) --> ReLU --> Linear(32) --> ReLU --> Linear(1) --> Sigmoid

2,337 trainable parameters.

Loss Function
^^^^^^^^^^^^^

Combined **Dice + BCE** loss (50/50 weight):

.. math::

   \mathcal{L}_\text{Dice} = 1 - \frac{2 \cdot TP}{2 \cdot TP + FP + FN}

True negatives (TN) **do not appear** in the Dice formula. This makes
the loss insensitive to the massive class imbalance (~99.4% non-fire).
BCE provides per-pixel gradient signals for early training convergence.

Training
^^^^^^^^

- **Train flights:** 03 (pre-burn) + 04 (day burn) + 05 (night burn)
- **Test flight:** 06 (day burn, unseen)
- **Labels:** Pseudo-labels from the threshold detector
- **Class balancing:** Minority class (fire) oversampled to 50/50


Assumptions and Limitations
----------------------------

1. **Flat-grid projection:** Equirectangular. Distortion <0.5% at 36 N
   over 0.3 degree.
2. **No atmospheric correction:** Acceptable for fire detection because
   fire signals dominate atmospheric effects by orders of magnitude.
3. **Simplified fire detection in mosaics:** Only absolute threshold
   (not contextual anomaly) for processing speed.
4. **Pseudo-labels for ML:** The ML model is trained on threshold
   detector outputs, not ground truth. It learns the threshold
   detector's behavior.
5. **FN ~ 0 assumption:** For intense prescribed burns, the threshold
   detector has near-zero false negatives. This assumption may not hold
   for small or smoldering fires.
6. **Spherical Earth model:** 111,000 m/degree approximation for area
   calculation. No ellipsoid correction.
7. **Nearest-neighbor resampling:** No sub-pixel interpolation. Each
   source pixel maps to one grid cell.


References
----------

- Giglio, L., Schroeder, W., & Justice, C. O. (2016). The collection 6
  MODIS active fire detection algorithm and fire products. *Remote
  Sensing of Environment*, 178, 31--41.
- `MASTER Instrument Overview <https://asapdata.arc.nasa.gov/sensors/master/>`_
- `MODIS Active Fire Product <https://modis.gsfc.nasa.gov/data/dataprod/mod14.php>`_

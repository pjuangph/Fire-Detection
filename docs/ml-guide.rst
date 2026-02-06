Machine Learning Fire Detection
================================

This guide documents the ML-based fire detection approach, its
assumptions, feature engineering, and integration with the real-time
detection system.


Assumptions
-----------

The ML fire detector is built on these explicit assumptions:

1. **Pseudo-labels from threshold detector** -- no ground truth fire masks
   exist for this dataset. Training labels come from
   :func:`lib.fire.detect_fire_simple` (T4 > 325 K daytime / 310 K
   nighttime, and T4 - T11 > 10 K). The ML model learns a smooth
   probability surface from these hard-threshold labels.

2. **Accumulated observations normalize the data** -- rather than
   predicting from a single observation ``[T4, T11, NDVI]``, we compute
   aggregate features across ALL observations of each pixel. These
   aggregates (max, mean, min, temporal change) capture each pixel's
   temporal context and serve as a self-normalizing representation.

3. **Nighttime fire = vegetation destroyed** -- nighttime thermal
   detections have fewer false positives than daytime (no solar
   reflection). Fire detected at night at a pixel with a known
   vegetation baseline directly implies vegetation loss. Daytime
   detections still require NDVI drop confirmation.

4. **Vegetation loss clears observation history** -- once fire has burned
   through a pixel (``veg_confirmed = True``) and no current thermal fire
   is present, all running accumulators are reset. The fire has passed;
   future observations start fresh. This prevents stale high-T4 values
   from keeping a pixel flagged indefinitely.

5. **NDVI at night is meaningless** -- reflected solar bands carry no
   signal at night. Nighttime NDVI is NaN. For the model, nighttime NDVI
   features are set to 0.0 (neutral). The model learns to rely solely on
   T4, T11, and dT for nighttime pixels.

6. **Train/test split with ground truth in both** -- flight 03 (pre-burn,
   no fire) is split 80/20 between train and test. Train: 80% of 03 + 04 + 05.
   Test: 20% of 03 + 06. This ensures the test set has ground truth "no fire"
   data for proper false positive evaluation.

7. **Fire is rare (class imbalance)** -- fire pixels are a tiny fraction
   of all observations. Training data is balanced by oversampling the
   fire class to a 1:1 ratio with the no-fire class.

8. **Equirectangular grid** -- the lat/lon grid (0.00025 deg/cell, ~28 m
   at 36 N) is sufficient at this scale. No map projection is needed.

9. **Observations are independent** -- each sweep produces one
   observation per pixel. Autocorrelation between overlapping passes is
   not explicitly modeled; it is captured implicitly through aggregate
   statistics.


Feature Engineering
-------------------

For each grid cell, we maintain running accumulators across all
observations. From these, 8 aggregate features are computed:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Feature
     - Formula
     - Physical Meaning
   * - ``T4_max``
     - max(T4) across all observations
     - Peak fire temperature -- strongest thermal signal
   * - ``T4_mean``
     - mean(T4) across all observations
     - Average thermal state -- normalizes the peak
   * - ``T11_mean``
     - mean(T11) across all observations
     - Background temperature -- stable reference
   * - ``dT_max``
     - max(T4 - T11) across all observations
     - Strongest spectral difference -- fire signature strength
   * - ``NDVI_min``
     - min(NDVI) across daytime observations
     - Lowest vegetation index -- burn scar indicator
   * - ``NDVI_mean``
     - mean(NDVI) across daytime observations
     - Average vegetation state -- normalizes the drop
   * - ``NDVI_drop``
     - NDVI_baseline - NDVI_min
     - Temporal vegetation loss -- independent fire confirmation
   * - ``obs_count``
     - Number of observations
     - Reliability indicator -- more observations = higher confidence

**Why these features work:** A fire pixel has high ``T4_max`` relative to
``T4_mean`` (thermal anomaly), large ``dT_max`` (strong spectral fire
signature), and large ``NDVI_drop`` (vegetation burned away). A false
positive from solar glint has high ``T4_max`` but NDVI remains stable
(no vegetation loss). The MLP learns these discriminating patterns.


Model Architecture
------------------

::

    Input (8 features)
      |
      Linear(8, 64) + ReLU
      |
      Linear(64, 32) + ReLU
      |
      Linear(32, 1)
      |
      sigmoid -> P(fire) in [0, 1]

- **Parameters:** 2,465
- **Loss:** Pixel-wise weighted BCEWithLogitsLoss (see below)
- **Optimizer:** Adam, lr = 0.001
- **Epochs:** 300
- **Batch size:** 4,096
- **Normalization:** Global z-score from all flights (mean/std saved
  with model checkpoint)


Pixel-Wise Weighted BCE Loss
----------------------------

Standard BCE loss treats all pixels equally. For fire detection, some errors
are worse than others:

- **False positive on ground truth (flight 03):** Definitely wrong -- there
  was no fire before the burn started.
- **False negative on fire pixel:** Missed a real detection -- we want to
  capture these.
- **Error on uncertain pixel:** Ambiguous -- threshold detector may be wrong.

We use pixel-wise weighted BCE loss to encode these priorities:

.. math::

   \mathcal{L} = \frac{1}{N} \sum_i w_i \cdot \text{BCE}(p_i, y_i)

where the weight for each pixel combines **importance** and **inverse-frequency**:

.. math::

   w_i = \text{importance}_i \times \frac{N}{\text{category\_count}_i}

Then normalized so :math:`\text{mean}(w) = 1` to keep gradient scale stable.

**Weight assignment:**

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Category
     - Importance
     - Rationale
   * - Flight 03 (ground truth no fire)
     - 10.0
     - FP here = definitely wrong, heavily penalize
   * - Fire pixels in burn flights
     - 5.0
     - Confirmed fire, want to capture (penalize FN)
   * - Non-fire in burn flights
     - 1.0
     - Uncertain, baseline importance

**Why this works:**

1. **Importance multiplier** (10, 5, 1): Encodes error severity
2. **Inverse-frequency**: Small categories (rare fire pixels) contribute
   proportionally despite being outnumbered
3. **Normalization** (mean=1): Keeps gradient magnitudes stable during training

**Expected outcome:**

- FP on flight 03 approaches 0 (heavily penalized)
- Recall on burn flights maintained (fire pixels have high weight)
- Model learns: "If thermal signature is ambiguous, don't call it fire"


Training Data
-------------

- **Ground truth split:** Flight 03 (pre-burn, no fire) is split 80/20
  between train and test. This ensures the test set has ground truth
  "no fire" data for proper false positive evaluation.
- **Train:** 80% of flight 03 + flights 04 (burn, day), 05 (burn, night)
- **Test:** 20% of flight 03 + flight 06 (burn, day)
- **Labels:** Pseudo-labels from threshold detector, aggregated per
  location (1 if any observation at that grid cell was fire)
- **Balancing:** Fire class oversampled to 1:1 ratio

**Data saved to:** ``dataset/fire_features.pkl.gz`` (compressed pickle)

**Model saved to:** ``checkpoint/fire_detector.pt`` containing:

- ``model_state``: PyTorch state dict
- ``mean``, ``std``: Training normalization statistics (8-element arrays)
- ``n_features``: 8
- ``threshold``: 0.5
- ``feature_names``: List of feature names


Integration with Real-Time System
----------------------------------

``realtime_fire.py`` auto-detects the saved model:

1. On startup, calls :func:`lib.fire.load_fire_model`
2. If ``checkpoint/fire_detector.pt`` exists, loads the MLP
3. After each sweep, ``process_sweep()`` updates running accumulators in
   grid state (``T4_max``, ``T4_sum``, ``T11_sum``, ``dT_max``,
   ``NDVI_min``, ``NDVI_sum``, ``NDVI_obs``)
4. The ML model computes aggregate features from accumulators and
   predicts P(fire) per pixel, replacing the threshold-based
   ``get_fire_mask()``
5. If no model is found, the system falls back to threshold detection
   (no behavior change)

**Running accumulators** are maintained in grid state alongside existing
arrays. They are expanded automatically when the grid grows (dynamic
grid expansion) and cleared when vegetation loss confirms fire has passed.


Comparison: Threshold vs ML
----------------------------

The threshold detector uses hard cutoffs:

- T4 > 325 K (day) or 310 K (night)
- T4 - T11 > 10 K

The ML detector learns a continuous decision surface:

- Outputs probability P(fire) instead of binary yes/no
- Uses NDVI to distinguish solar glint (high NDVI) from fire (low NDVI)
- Leverages temporal patterns across multiple observations
- Nighttime detections directly imply vegetation loss

**Expected improvements:**

- Fewer false positives (NDVI context eliminates solar glint)
- Probability output enables confidence-weighted decisions
- Temporal accumulation catches fire even when single observations are
  below threshold


Usage
-----

::

    # Train the model (requires HDF data in ignite_fire_data/)
    python tune_fire_prediction.py

    # Run real-time simulation with ML model
    python realtime_fire.py
    # (auto-detects checkpoint/fire_detector.pt)

See :doc:`operators-guide` for output interpretation and
:doc:`science-guide` for the underlying physics.

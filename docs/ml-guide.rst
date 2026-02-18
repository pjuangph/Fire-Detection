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
observations. From these, 12 aggregate features are computed:

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
   * - ``SWIR_max``
     - max(SWIR) across all observations
     - Peak 2.2 μm radiance -- fire Planck emission at short wavelength
   * - ``SWIR_mean``
     - mean(SWIR) across all observations
     - Average SWIR radiance -- normalizes peak, high at night = fire
   * - ``Red_mean``
     - mean(Red) across all observations
     - Average 0.654 μm radiance -- implicit day/night indicator (~0 at night)
   * - ``NIR_mean``
     - mean(NIR) across all observations
     - Average 0.866 μm radiance -- implicit day/night indicator (~0 at night)
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
signature), high ``SWIR_max`` (Planck emission from hot sources), and
large ``NDVI_drop`` (vegetation burned away). A false positive from solar
glint has high ``T4_max`` but NDVI remains stable (no vegetation loss)
and SWIR is low at night. The MLP learns these discriminating patterns.


Model Architecture
------------------

The architecture is **variable via YAML grid search**
(``configs/grid_search.yaml``). The network is a fully connected MLP
with configurable hidden layers:

::

    Input (12 features)
      |
      [Hidden layers: variable width and depth]
      |
      Linear(last_hidden, 1)
      |
      sigmoid -> P(fire) in [0, 1]

**Best architecture** (from grid search run 37): ``12 -> 64 -> 32 -> 1``

- **Parameters:** ~2,721 (varies by architecture)
- **Loss:** Weighted BCE or SoftErrorRateLoss (see below)
- **Optimizer:** Adam (learning rate selected via grid search)
- **Epochs:** 300
- **Batch size:** 4,096
- **Normalization:** Global z-score from all flights (mean/std saved
  with model checkpoint)


Loss Functions
--------------

Two loss functions are compared via grid search:

1. Weighted BCE Loss
^^^^^^^^^^^^^^^^^^^^

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

2. SoftErrorRateLoss
^^^^^^^^^^^^^^^^^^^^

Directly minimizes the error rate instead of per-pixel cross-entropy:

.. math::

   \mathcal{L} = \frac{\text{soft\_FN} + \text{soft\_FP}}{P}

where :math:`P` is the total number of actual fire pixels (positives).
Soft counts are computed with differentiable approximations:

- :math:`\text{soft\_FN} = \sum_{i: y_i=1} (1 - p_i)` -- predicted
  probability mass missed on true fire pixels
- :math:`\text{soft\_FP} = \sum_{i: y_i=0} p_i` -- predicted probability
  mass assigned to non-fire pixels

**Key properties:**

- True negatives (TN) do not appear in the loss -- correctly ignoring
  the dominant class.
- Uses **uniform weights** (all 1.0) with minority class oversampling
  instead of per-pixel importance weighting.
- Directly optimizes the evaluation metric (error rate).

Evaluation Metric
^^^^^^^^^^^^^^^^^

All models are evaluated using error rate:

.. math::

   \text{error\_rate} = \frac{FN + FP}{P}

where :math:`P` is the number of actual fire pixels. This metric counts
absolute errors (false negatives + false positives) relative to the number
of real fire detections. Lower is better.


Grid Search
-----------

The grid search explores 49 configurations across the hyperparameter space:

- **2 loss functions:** Weighted BCE, SoftErrorRateLoss
- **4 architectures:** e.g., ``[64, 32]``, ``[64, 64, 32]``,
  ``[64, 64, 64, 32]``, ``[128, 64, 32]``
- **3 learning rates:** e.g., 0.001, 0.0005, 0.0001
- **Multiple importance weight configurations** (for Weighted BCE only)

Configuration is defined in ``configs/grid_search.yaml``. Run with::

   python tune_fire_prediction.py --config configs/grid_search.yaml

Results are saved to ``results/grid_search_results.json``.


Best Model Results
^^^^^^^^^^^^^^^^^^

**Run 37** achieved the best error rate:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Value
   * - Loss function
     - SoftErrorRateLoss
   * - Architecture
     - 12 -> 64 -> 32 -> 1
   * - Error rate
     - **0.031**
   * - True Positives (TP)
     - 9,009
   * - False Positives (FP)
     - 221
   * - False Negatives (FN)
     - 59

The best model is automatically saved to ``checkpoint/fire_detector_best.pt``.


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
- ``mean``, ``std``: Training normalization statistics (12-element arrays)
- ``n_features``: 12
- ``threshold``: 0.5
- ``feature_names``: List of feature names


Integration with Real-Time System
----------------------------------

``realtime_fire.py`` auto-detects the saved model:

1. On startup, calls :func:`lib.fire.load_fire_model`
2. If ``checkpoint/fire_detector.pt`` exists, loads the MLP
3. After each sweep, ``process_sweep()`` updates running accumulators in
   grid state (``T4_max``, ``T4_sum``, ``T11_sum``, ``dT_max``,
   ``SWIR_max``, ``SWIR_sum``, ``Red_sum``, ``NIR_sum``,
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

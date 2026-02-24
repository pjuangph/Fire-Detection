Operator's Guide
================

This guide is for pilots, flight operators, and anyone running the fire
detection software without needing to understand the underlying physics.
For algorithm details, see :doc:`science-guide`.

Quick Start
-----------

1. **Install dependencies**::

      pip install pyhdf numpy matplotlib torch scikit-learn pandas scipy earthaccess

   ``pyhdf`` requires HDF4 C libraries. On macOS with conda::

      conda install -c conda-forge pyhdf

2. **Download the data** (requires a free
   `NASA Earthdata account <https://urs.earthdata.nasa.gov/>`_)::

      python download_data.py                # all 4 flights (~9 GB)
      python download_data.py --flight 04    # just one flight (~4 GB)
      python download_data.py --list         # list files without downloading

   The script will prompt for your Earthdata credentials on first run
   and cache them for future use. Alternatively, place MASTER L1B HDF4
   files manually in ``ignite_fire_data/``.

3. **Run a script**::

      python mosaic_flight.py          # standard mosaic + fire overlay
      python realtime_fire.py          # real-time sweep simulation
      python plot_burn_locations.py    # per-flight burn analysis
      python plot_vegetation.py        # NDVI vegetation maps (daytime)
      python detect_fire.py            # single-file fire detection
      python tune_fire_prediction.py --config configs/grid_search.yaml  # grid search
      python compare_fire_detectors.py  # compare ML vs threshold

All output goes to the ``plots/`` directory.


Flights in the Dataset
----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 30 10 10 20

   * - Flight
     - Date
     - Description
     - Day/Night
     - Files
     - Fire Present?
   * - 24-801-03
     - Oct 18, 2023
     - Pre-burn data collection
     - Day
     - 9
     - **No** -- collected before the burn
   * - 24-801-04
     - Oct 19, 2023
     - First fire flight
     - Day
     - 40
     - **Yes** -- USFS Blowdown Prescribed Burn
   * - 24-801-05
     - Oct 20, 2023
     - Second fire flight (nighttime)
     - Night
     - 16
     - **Yes** -- USFS Lakes Unit Prescribed Burn
   * - 24-801-06
     - Oct 20, 2023
     - Third fire flight
     - Day
     - 18
     - **Yes** -- USFS Blowdown Prescribed Burn


Project Structure
-----------------

Scripts
^^^^^^^

Scripts are organized by workflow stage. Run them from the project root.

**Data Acquisition & Inspection**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Script
     - Purpose
     - Output
   * - ``download_data.py``
     - Download MASTER L1B data from NASA Earthdata
     - ``ignite_fire_data/*.hdf``
   * - ``plotdata.py``
     - Inspect HDF structure, plot sample channels across spectrum
     - ``plots/radiance_overview.png``

**Fire Detection & Mosaics**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Script
     - Purpose
     - Output
   * - ``detect_fire.py``
     - Single-file contextual fire detection (pre-burn vs burn)
     - ``plots/fire_detection_*.png``
   * - ``mosaic_flight.py``
     - Assemble flight lines into georeferenced mosaic with fire overlay
     - ``plots/mosaic_flight_*.png``

**Analysis & Visualization**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Script
     - Purpose
     - Output
   * - ``plot_burn_locations.py``
     - 2x2 per-flight analysis: fire map, T4, SWIR, scatter
     - ``plots/burn_locations_*.png``
   * - ``plot_vegetation.py``
     - 2x2 NDVI vegetation maps with fire overlay (daytime only)
     - ``plots/vegetation_*.png``
   * - ``plot_grid_resolution.py``
     - Native (~8 m) vs gridded (28 m) resolution comparison
     - ``plots/grid_resolution_*.png``

**Model Training**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Script
     - Purpose
     - Output
   * - ``train_mlp.py``
     - MLP grid search (``--config configs/grid_search_mlp.yaml``)
     - ``checkpoint/fire_detector_mlp_best.pt``
   * - ``train_tabpfn_classification.py``
     - TabPFN classification grid search
     - ``checkpoint/fire_detector_tabpfn_best.pt``
   * - ``train_tabpfn_regression.py``
     - TabPFN regression grid search
     - ``checkpoint/fire_detector_tabpfn_regression_best.pt``
   * - ``tune_fire_prediction-MLP.py``
     - Legacy MLP grid search (superseded by ``train_mlp.py``)
     - ``checkpoint/fire_detector_best.pt``

**Real-Time Simulation**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Script
     - Purpose
     - Output
   * - ``realtime_fire.py``
     - Sweep-by-sweep simulation; threshold (default) or ML via ``--config``
     - ``plots/realtime/simple-*.png`` or ``ml-*.png``
   * - ``realtime_mlp.py``
     - MLP-specific realtime wrapper (``--config configs/best_model.yaml``)
     - ``plots/realtime/ml-*.png``
   * - ``realtime_tabpfn_classification.py``
     - TabPFN classifier realtime simulation
     - ``plots/realtime/tabpfn_classification-*.png``
   * - ``realtime_tabpfn_regression.py``
     - TabPFN regressor realtime simulation
     - ``plots/realtime/tabpfn_regression-*.png``

**Comparison & Reporting**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Script
     - Purpose
     - Output
   * - ``compare_fire_detectors.py``
     - Per-flight ML vs threshold comparison table
     - stdout
   * - ``make_gifs.py``
     - Animate realtime frames; pairwise detector comparison GIFs
     - ``plots/gifs/*.gif``
   * - ``create_presentation.py``
     - Generate PowerPoint presentation from all results
     - ``Fire_Detection_Presentation.pptx``
   * - ``plot_presentation_diagrams.py``
     - Architecture and flow diagrams for presentation
     - ``plots/diagram_*.png``

**Shell Orchestration Scripts**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - Purpose
   * - ``generate_plots.sh``
     - Run all visualization, realtime simulations, and GIF creation
   * - ``train-all-models.sh``
     - Train MLP + TabPFN classification + TabPFN regression sequentially
   * - ``reset_project.sh``
     - Clean up generated artifacts (``--plots``, ``--model``, ``--stale``)


Library Modules (``lib/``)
^^^^^^^^^^^^^^^^^^^^^^^^^^

All shared functions live in ``lib/`` and are re-exported through
``lib/__init__.py`` for convenient imports (``from lib import ...``).

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Module
     - Purpose
     - Key Functions / Classes
   * - ``constants.py``
     - Physical constants, MASTER channel indices, grid parameters
     - ``CH_T4``, ``CH_T11``, ``CH_SWIR``, ``CH_RED``, ``CH_NIR``, ``GRID_RES``
   * - ``io.py``
     - HDF file I/O, radiance to brightness temperature conversion
     - ``radiance_to_bt()``, ``process_file()``, ``group_files_by_flight()``
   * - ``fire.py``
     - Threshold + contextual fire detection, zone labeling
     - ``detect_fire_simple()``, ``detect_fire()``, ``detect_fire_zones()``
   * - ``vegetation.py``
     - NDVI computation, sunlight detection, vegetation loss
     - ``compute_ndvi()``, ``has_sunlight()``, ``detect_vegetation_loss()``
   * - ``mosaic.py``
     - Incremental gridding, dynamic grid expansion
     - ``build_mosaic()``, ``process_sweep()``, ``get_fire_mask()``
   * - ``features.py``
     - 12 aggregate features per grid cell for ML models
     - ``build_location_features()``
   * - ``firemlp.py``
     - Variable-depth MLP neural network architecture
     - ``FireMLP(nn.Module)``
   * - ``inference.py``
     - Model loading (MLP, TabPFN) and prediction
     - ``load_fire_model()``, ``predict()``
   * - ``losses.py``
     - Loss functions: weighted BCE, SoftErrorRateLoss
     - ``SoftErrorRateLoss``, ``compute_pixel_weights()``
   * - ``training.py``
     - Data pipeline: load, split, oversample, grid search helpers
     - ``load_all_data()``, ``extract_train_test()``, ``oversample_minority()``
   * - ``evaluation.py``
     - TP/FP/FN metrics, device selection (CUDA/MPS/CPU)
     - ``evaluate()``, ``print_metrics()``, ``get_device()``
   * - ``plotting.py``
     - Training diagnostic plots (loss curves, convergence)
     - ``plot_training_loss()``, ``plot_convergence_curves()``
   * - ``realtime.py``
     - Shared rendering and simulation for all realtime scripts
     - ``render_frame()``, ``simulate_flight()``
   * - ``stats.py``
     - Pixel tables, grid cell area computation
     - ``build_pixel_table()``, ``compute_cell_area_m2()``
   * - ``context_sampling.py``
     - TabPFN context sampling to control memory usage
     - ``EvalBatching``, ``sample_train_context_indices()``


Configuration Files (``configs/``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - What It Configures
   * - ``grid_search_mlp.yaml``
     - MLP grid search: 4 architectures x 3 LRs x 2 losses
   * - ``grid_search_tabpfn_classification.yaml``
     - TabPFN classification: LR, batch size, estimators, weight decay, grad clip
   * - ``grid_search_tabpfn_regression.yaml``
     - TabPFN regression: same as above + CRPS/MSE loss weights
   * - ``best_model.yaml``
     - Best MLP model config (auto-generated by training)
   * - ``best_model_mlp.yaml``
     - Best MLP from ``train_mlp.py`` (auto-generated)
   * - ``best_model_tabpfn_classification.yaml``
     - Best TabPFN classifier (auto-generated)
   * - ``best_model_tabpfn_regression.yaml``
     - Best TabPFN regressor (auto-generated)


Typical Workflow
^^^^^^^^^^^^^^^^

::

   python download_data.py                     # 1. Fetch MASTER data (~9 GB)
   bash train-all-models.sh                    # 2. Train MLP + both TabPFN models
   bash generate_plots.sh                      # 3. Generate all plots + GIFs
   python create_presentation.py               # 4. Build PowerPoint presentation


Reading the Outputs
-------------------

Fire Overlay Colors
^^^^^^^^^^^^^^^^^^^

- **Red dots with black edge**: Fire detections that passed the absolute
  temperature threshold (T4 > 325 K daytime, 310 K nighttime).
- **Magenta dots** (#FF00FF, in ``realtime_fire.py``):
  **Vegetation-confirmed fire** -- pixels where thermal fire is
  independently confirmed by NDVI drop (vegetation loss). Higher
  confidence than red-only.
- **Magenta dots** (#FF00FF, in ``detect_fire.py``): Contextual anomaly
  detections -- pixels that are anomalously warm relative to their
  neighbors but below the absolute threshold.
- **Black bounding boxes**: Bounding boxes around detected fire zones.

Background Layer
^^^^^^^^^^^^^^^^

The display automatically adapts based on available sunlight data:

- **Daytime (NDVI background)**: Green tones show healthy vegetation,
  brown/red tones show bare soil or stressed vegetation. Fire appears
  as red dots overlaid on the vegetation map.
- **Nighttime (T4 thermal background)**: Warm colors show terrain
  temperature. Fire appears as bright spots against the cooler
  background.

The system detects day/night **per pixel** from actual NIR radiance
levels, not from clock time. This handles cloud cover correctly --
if clouds block sunlight, those pixels show thermal background even
during geometric daytime.

Plot Title (Real-Time Simulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The real-time simulation plot title now shows the **detector type** being
used (e.g., "Threshold Detector" or "MLP Detector"). This makes it
immediately clear which detection method produced the displayed results.

Stats Box (Real-Time Simulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``realtime_fire.py`` output includes a stats box showing:

- **Sweep N/M [NDVI or T4]**: Current sweep number and background type.
- **Coverage**: Percentage of grid cells with data so far.
- **Fire pixels**: Total confirmed fire pixels (after multi-pass filter).
- **Veg-confirmed**: Fire pixels independently confirmed by vegetation
  loss (NDVI drop from baseline). These are shown as magenta dots.
- **Total fire area**: Estimated area in m\ :sup:`2` or hectares.
- **Fire zones**: Number of spatially connected fire regions.
- **Zone breakdown**: Top 5 zones by size with individual areas.

What "Multi-Pass Confirmed" Means
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When a pixel is observed by multiple flight lines (overlapping passes),
the system requires fire to be detected in **at least 2 passes** before
confirming it as fire. This eliminates false positives from solar
reflection, which are angle-dependent and only trigger from one
direction. Real fire emits in all directions and triggers consistently.


Real-Time Simulation
--------------------

``realtime_fire.py`` processes one sweep at a time, building the mosaic
incrementally. Each sweep produces a PNG frame showing the current state.

To create an animated GIF from the frames::

   convert -delay 50 -loop 0 plots/realtime_2480104/frame_*.png animation.gif

This requires `ImageMagick <https://imagemagick.org/>`_.


Data Requirements
-----------------

The HDF4 files should be placed in ``ignite_fire_data/``. They are
MASTER Level 1B files from the FireSense 2023 campaign, publicly
available from the `NASA ASAP Data Archive
<https://asapdata.arc.nasa.gov/sensors/master/>`_.

File Naming Convention
^^^^^^^^^^^^^^^^^^^^^^

::

   MASTERL1B_2480104_20_20231019_2031_2033_V01.hdf
   |         |       |  |        |    |    |
   |         |       |  |        |    |    +-- Version
   |         |       |  |        |    +------- End time (HHMM UTC)
   |         |       |  |        +------------ Start time (HHMM UTC)
   |         |       |  +--------------------- Date (YYYYMMDD)
   |         |       +------------------------ Flight line number
   |         +-------------------------------- Flight ID
   +------------------------------------------ Product: MASTER Level 1B


Troubleshooting
---------------

**"No module named 'pyhdf'"**
   Install HDF4 C libraries first, then ``pip install pyhdf``.
   On macOS: ``conda install -c conda-forge pyhdf``.

**"No module named 'lib'"**
   Ensure the ``lib/`` directory is present in the project root.
   If you cloned from git and ``lib/`` is missing, check that
   ``.gitignore`` does not contain a bare ``lib/`` entry.

**No HDF files found**
   Place MASTER L1B files in ``ignite_fire_data/`` relative to the
   project root. Files must match ``MASTERL1B_*.hdf``.

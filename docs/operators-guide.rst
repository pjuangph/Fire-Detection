Operator's Guide
================

This guide is for pilots, flight operators, and anyone running the fire
detection software without needing to understand the underlying physics.
For algorithm details, see :doc:`science-guide`.

Quick Start
-----------

1. **Install dependencies**::

      pip install pyhdf numpy matplotlib torch scikit-learn pandas scipy

   ``pyhdf`` requires HDF4 C libraries. On macOS with conda::

      conda install -c conda-forge pyhdf

2. **Place data files** in ``ignite_fire_data/``. Files must be MASTER
   Level 1B HDF4 files (``MASTERL1B_*.hdf``).

3. **Run a script**::

      python mosaic_flight.py          # standard mosaic + fire overlay
      python realtime_fire.py          # real-time sweep simulation
      python plot_burn_locations.py    # per-flight burn analysis
      python plot_vegetation.py        # NDVI vegetation maps (daytime)
      python detect_fire.py            # single-file fire detection
      python fire_ml.py                # train ML fire detector

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


What Each Script Produces
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Script
     - Purpose
     - Output Files
   * - ``mosaic_flight.py``
     - Assemble flight lines into georeferenced mosaic with fire overlay
     - ``plots/mosaic_flight_*.png``
   * - ``realtime_fire.py``
     - Simulate real-time sweep-by-sweep fire detection
     - ``plots/realtime_*/frame_*.png``
   * - ``plot_burn_locations.py``
     - Per-flight 2x2 analysis (fire map, T4, SWIR, scatter)
     - ``plots/burn_locations_*.png``
   * - ``plot_vegetation.py``
     - 2x2 NDVI vegetation maps with fire overlay (daytime only)
     - ``plots/vegetation_*.png``
   * - ``detect_fire.py``
     - Single-file fire detection comparing pre-burn vs burn
     - ``plots/fire_detection_*.png``, ``plots/fire_map_burn.png``
   * - ``fire_ml.py``
     - Train MLP fire detector, produce prediction maps
     - ``plots/ml_*.png``


Reading the Outputs
-------------------

Fire Overlay Colors
^^^^^^^^^^^^^^^^^^^

- **Red dots**: Fire detections that passed the absolute temperature
  threshold (T4 > 325 K daytime, 310 K nighttime).
- **Orange dots** (in ``detect_fire.py`` only): Contextual anomaly
  detections -- pixels that are anomalously warm relative to their
  neighbors but below the absolute threshold.

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

Stats Box (Real-Time Simulation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``realtime_fire.py`` output includes a stats box showing:

- **Sweep N/M [NDVI or T4]**: Current sweep number and background type.
- **Coverage**: Percentage of grid cells with data so far.
- **Fire pixels**: Total confirmed fire pixels (after multi-pass filter).
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

MASTER Fire Detection Documentation
====================================

Detect active fire from NASA MASTER airborne thermal imaging data.

This project processes Level 1B (L1B) calibrated radiance from the
MODIS/ASTER Airborne Simulator (MASTER) and identifies fire pixels using
physics-based threshold detection, contextual anomaly analysis,
multi-pass consistency filtering, and machine learning.

The data comes from the **FireSense 2023** campaign -- prescribed burns
on the Kaibab Plateau in Arizona (October 18--20, 2023).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   operators-guide
   science-guide
   api/index
   scripts/index

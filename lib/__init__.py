"""Fire-Detection library — shared functions for MASTER L1B fire analysis."""

from lib.constants import (
    H_PLANCK, C_LIGHT, K_BOLTZ,
    CH_T4, CH_T11, CH_SWIR, CH_RED, CH_NIR,
    GRID_RES,
)
from lib.io import (
    radiance_to_bt, process_file, group_files_by_flight, compute_grid_extent,
)
from lib.fire import (
    detect_fire_simple, detect_fire, is_daytime, detect_fire_zones,
    compute_aggregate_features, MLFireDetector, load_fire_model,
)
from lib.vegetation import compute_ndvi, has_sunlight, detect_vegetation_loss
from lib.mosaic import (
    build_mosaic, init_grid_state, process_sweep, get_fire_mask,
)
from lib.stats import (
    build_pixel_table, compute_location_stats, compute_cell_area_m2,
    format_area,
)

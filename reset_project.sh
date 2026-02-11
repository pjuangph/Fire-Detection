#!/usr/bin/env bash
# reset_project.sh — Delete trained model, cached dataset, and generated plots.
#
# Usage:
#   ./reset_project.sh          # delete everything (model + dataset + all plots)
#   ./reset_project.sh --plots  # delete only plots
#   ./reset_project.sh --model  # delete only model checkpoint + dataset
#   ./reset_project.sh --stale  # delete only stale/orphaned plots
#
# Regenerable plots (have a generating script):
#   tune_fire_prediction.py  → plots/tune_*.png
#   realtime_fire.py         → plots/realtime/{detector}-{flight}-{frame}.png
#   mosaic_flight.py         → plots/mosaic_flight_*.png
#   plot_burn_locations.py   → plots/burn_locations_*.png
#   plot_vegetation.py       → plots/vegetation_*.png
#   detect_fire.py           → plots/fire_detection_*.png, fire_map_burn.png, fire_comparison.png
#   plotdata.py              → plots/radiance_overview.png, georeferenced_thermal.png
#
# Stale/orphaned plots (old model or no generating script):
#   (deleted fire_ml.py)     → plots/ml_*.png
#   (no script)              → plots/blackbody_temperatures.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Helpers ──────────────────────────────────────────────────

removed=0

rm_if_exists() {
    local path="$1"
    if [ -e "$path" ]; then
        rm -rf "$path"
        echo "  Removed $path"
        removed=$((removed + 1))
    fi
}

delete_model() {
    echo "Cleaning model checkpoint and dataset..."
    rm_if_exists checkpoint/fire_detector.pt
    rm_if_exists dataset/fire_features.pkl.gz
}

delete_stale_plots() {
    echo "Cleaning stale/orphaned plots (old fire_ml.py outputs + blackbody)..."
    rm_if_exists plots/ml_decision_boundary.png
    rm_if_exists plots/ml_training_loss.png
    for f in plots/ml_prediction_map_*.png; do
        rm_if_exists "$f"
    done
    for f in plots/ml_fpfn_comparison_*.png; do
        rm_if_exists "$f"
    done
    rm_if_exists plots/blackbody_temperatures.png
}

delete_all_plots() {
    echo "Cleaning all generated plots..."

    # tune_fire_prediction.py outputs
    rm_if_exists plots/tune_training_loss.png
    rm_if_exists plots/tune_probability_hist.png
    for f in plots/tune_prediction_map_*.png; do
        rm_if_exists "$f"
    done

    # realtime_fire.py outputs (new flat dir + old per-flight dirs)
    rm_if_exists plots/realtime
    for d in plots/realtime_*/; do
        rm_if_exists "$d"
    done

    # mosaic_flight.py outputs
    for f in plots/mosaic_flight_*.png; do
        rm_if_exists "$f"
    done

    # plot_burn_locations.py outputs
    for f in plots/burn_locations_*.png; do
        rm_if_exists "$f"
    done

    # plot_vegetation.py outputs
    for f in plots/vegetation_*.png; do
        rm_if_exists "$f"
    done

    # detect_fire.py outputs
    rm_if_exists plots/fire_detection_preburn.png
    rm_if_exists plots/fire_detection_burn.png
    rm_if_exists plots/fire_map_burn.png
    rm_if_exists plots/fire_comparison.png

    # plotdata.py outputs
    rm_if_exists plots/radiance_overview.png
    rm_if_exists plots/georeferenced_thermal.png

    # Stale/orphaned
    delete_stale_plots
}

# ── Main ─────────────────────────────────────────────────────

mode="${1:-all}"

case "$mode" in
    --plots)
        delete_all_plots
        ;;
    --model)
        delete_model
        ;;
    --stale)
        delete_stale_plots
        ;;
    --help|-h)
        head -16 "$0" | tail -15
        exit 0
        ;;
    *)
        delete_model
        delete_all_plots
        ;;
esac

echo ""
echo "Done. Removed $removed items."

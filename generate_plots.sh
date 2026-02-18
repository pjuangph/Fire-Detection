#!/usr/bin/env bash
# Generate all plots, diagrams, realtime simulations, and GIFs.
# Usage: bash generate_plots.sh

set -e

echo "=== Presentation diagrams ==="
python plot_presentation_diagrams.py

echo ""
echo "=== Burn location plots ==="
python plot_burn_locations.py

echo ""
echo "=== Vegetation maps ==="
python plot_vegetation.py

echo ""
echo "=== Grid resolution comparison ==="
python plot_grid_resolution.py

echo ""
echo "=== Simple threshold detector (realtime) ==="
python realtime_fire.py

echo ""
echo "=== MLP detector (realtime) ==="
python realtime_fire.py --config configs/best_model.yaml

echo ""
echo "=== Creating GIFs ==="
python make_gifs.py

echo ""
echo "Done. All plots in plots/, GIFs in plots/gifs/"

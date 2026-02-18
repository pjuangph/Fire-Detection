#!/usr/bin/env bash
# Run realtime fire detection simulations for both detectors.
# Usage: bash run_realtime.sh

set -e

echo "=== Simple threshold detector ==="
python realtime_fire.py

echo ""
echo "=== MLP detector ==="
python realtime_fire.py --config configs/best_model.yaml

echo ""
echo "=== Creating GIFs ==="
python make_gifs.py

echo ""
echo "Done. Frames in plots/realtime/, GIFs in plots/gifs/"

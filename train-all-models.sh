#!/usr/bin/env bash
# Train all fire detection models (MLP + TabPFN classification + TabPFN regression).
# Stops on first failure.
set -euo pipefail

echo "=========================================="
echo "  Training all fire detection models"
echo "=========================================="

echo ""
echo "--- 1/3  MLP ---"
python train_mlp.py --config configs/grid_search_mlp.yaml

echo ""
echo "--- 2/3  TabPFN Classification ---"
python train_tabpfn_classification.py --config configs/grid_search_tabpfn_classification.yaml

echo ""
echo "--- 3/3  TabPFN Regression ---"
python train_tabpfn_regression.py --config configs/grid_search_tabpfn_regression.yaml

echo ""
echo "=========================================="
echo "  All models trained successfully"
echo "=========================================="

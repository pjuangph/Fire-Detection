#!/usr/bin/env bash
# Train all fire detection models (MLP + TabPFN classification + TabPFN regression).
# After training, compare hybrid vs StandardScaler normalization on the best model.
# Stops on first failure.
set -euo pipefail

echo "=========================================="
echo "  Training all fire detection models"
echo "=========================================="

mkdir -p results

echo ""
echo "--- 1/3  MLP ---"
python train_mlp.py --config configs/grid_search_mlp.yaml

echo ""
echo "--- 2/3  TabPFN Classification ---"
# python train_tabpfn_classification.py --config configs/grid_search_tabpfn_classification.yaml

echo ""
echo "--- 3/3  TabPFN Regression ---"
# python train_tabpfn_regression.py --config configs/grid_search_tabpfn_regression.yaml

echo ""
echo "=========================================="
echo "  All models trained successfully"
echo "=========================================="

echo ""
echo "--- Normalization comparison (best MLP: hybrid vs StandardScaler) ---"
python compare_normalization.py --config configs/best_model_mlp.yaml

echo ""
echo "=========================================="
echo "  All done"
echo "=========================================="

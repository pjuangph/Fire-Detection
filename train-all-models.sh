#!/usr/bin/env bash
# Train all fire detection models (MLP + TabPFN classification + TabPFN regression).
# After training, compare hybrid vs StandardScaler normalization on the best model.
# Stops on first failure.
set -euo pipefail

# Number of parallel MLP workers (set to 1 for sequential).
NUM_WORKERS=2

echo "=========================================="
echo "  Training all fire detection models"
echo "=========================================="

echo ""
echo "--- 1/3  MLP (${NUM_WORKERS} parallel workers) ---"
if [ "$NUM_WORKERS" -gt 1 ]; then
  mkdir -p results
  pids=()
  for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "  Launching worker $i..."
    python -u train_mlp.py --config configs/grid_search_mlp.yaml \
      --worker-id "$i" --num-workers "$NUM_WORKERS" \
      > "results/mlp_worker_${i}.log" 2>&1 &
    pids+=($!)
  done
  echo "  Waiting for ${NUM_WORKERS} workers (PIDs: ${pids[*]})..."
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "  ERROR: worker PID $pid failed"
      fail=1
    fi
  done
  if [ "$fail" -ne 0 ]; then
    echo "  Some workers failed. Check results/mlp_worker_*.log"
    exit 1
  fi
  echo "  All workers finished. Merging results..."
  python train_mlp.py --config configs/grid_search_mlp.yaml --merge-results
else
  python train_mlp.py --config configs/grid_search_mlp.yaml
fi

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

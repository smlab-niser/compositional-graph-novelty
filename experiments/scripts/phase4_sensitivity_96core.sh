#!/bin/bash

# Phase 4: Sensitivity Analysis (96-core optimized)
# Test robustness to hyperparameter choices
# - Motif size k (parallelized)
# - Component weights
# - Corpus size
# - Grid search

set -e

echo "Starting Phase 4: Sensitivity Analysis (96-core optimized)"

# Create output directory
mkdir -p results/sensitivity

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"

# Launch all tasks in parallel
PIDS=()

# Motif size sensitivity - run k=2,3,4,5 in PARALLEL (not sequential!)
echo "Launching motif size k=2..."
$PYTHON experiments/sensitivity/motif_size.py \
    --k 2 \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/k2.json &
PIDS+=($!)

echo "Launching motif size k=3..."
$PYTHON experiments/sensitivity/motif_size.py \
    --k 3 \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/k3.json &
PIDS+=($!)

echo "Launching motif size k=4..."
$PYTHON experiments/sensitivity/motif_size.py \
    --k 4 \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/k4.json &
PIDS+=($!)

echo "Launching motif size k=5..."
$PYTHON experiments/sensitivity/motif_size.py \
    --k 5 \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/k5.json &
PIDS+=($!)

# Weight sensitivity (100 random combinations)
echo "Launching weight sensitivity..."
$PYTHON experiments/sensitivity/weight_sensitivity.py \
    --n_trials 100 \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/weights.json &
PIDS+=($!)

# Corpus size sensitivity
echo "Launching corpus size sensitivity..."
$PYTHON experiments/sensitivity/corpus_size.py \
    --sizes 100,500,1000,5000,10000,50000 \
    --datasets qm9,arxiv \
    --output results/sensitivity/corpus_size.json &
PIDS+=($!)

# Hyperparameter grid search
echo "Launching grid search..."
$PYTHON experiments/sensitivity/grid_search.py \
    --param_grid configs/param_grid.yaml \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/grid_search.json &
PIDS+=($!)

# Wait for all with progress tracking
echo ""
echo "Launched ${#PIDS[@]} parallel sensitivity analyses..."
echo "Waiting for completion..."
echo ""

COMPLETED=0
TOTAL=${#PIDS[@]}

for pid in "${PIDS[@]}"; do
    wait $pid
    COMPLETED=$((COMPLETED + 1))
    echo "  ✓ Completed $COMPLETED/$TOTAL analyses"
done

echo ""
echo "Phase 4 complete!"
echo "Results saved in results/sensitivity/"
echo ""

# List all results
echo "Generated files:"
ls -lh results/sensitivity/*.json

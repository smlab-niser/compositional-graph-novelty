#!/bin/bash

# Phase 4: Sensitivity Analysis
# Test robustness to hyperparameter choices
# - Motif size k
# - Component weights
# - Corpus size
# - Grid search

set -e

echo "Starting Phase 4: Sensitivity Analysis"

# Create output directory
mkdir -p results/sensitivity

echo "GPU 0: Motif size sensitivity (k=2,3,4,5)..."
(
    for k in 2 3 4 5; do
        CUDA_VISIBLE_DEVICES=0 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/sensitivity/motif_size.py \
            --k $k \
            --datasets qm9,arxiv,reddit,protein \
            --output results/sensitivity/k${k}.json
        echo "  Completed k=$k"
    done
    echo "GPU 0: Motif size sensitivity complete"
) &
PID1=$!

echo "GPU 1: Weight sensitivity (100 random combinations)..."
CUDA_VISIBLE_DEVICES=1 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/sensitivity/weight_sensitivity.py \
    --n_trials 100 \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/weights.json &
PID2=$!

echo "GPU 2: Corpus size sensitivity..."
CUDA_VISIBLE_DEVICES=2 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/sensitivity/corpus_size.py \
    --sizes 100,500,1000,5000,10000,50000 \
    --datasets qm9,arxiv \
    --output results/sensitivity/corpus_size.json &
PID3=$!

echo "GPU 3: Hyperparameter grid search..."
CUDA_VISIBLE_DEVICES=3 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/sensitivity/grid_search.py \
    --param_grid configs/param_grid.yaml \
    --datasets qm9,arxiv,reddit,protein \
    --output results/sensitivity/grid_search.json &
PID4=$!

# Wait for all
echo "Waiting for all sensitivity analyses to complete..."

wait $PID1
echo "✓ GPU 0 (motif size) complete"

wait $PID2
echo "✓ GPU 1 (weight sensitivity) complete"

wait $PID3
echo "✓ GPU 2 (corpus size) complete"

wait $PID4
echo "✓ GPU 3 (grid search) complete"

echo "Phase 4 complete!"
echo "Results saved in results/sensitivity/"

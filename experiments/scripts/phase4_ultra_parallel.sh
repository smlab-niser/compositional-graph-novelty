#!/bin/bash

# Phase 4: Sensitivity Analysis (ULTRA parallelized - 96 cores)
# Each experiment uses multiprocessing internally

set -e

echo "=" * 80
echo " Phase 4: Sensitivity Analysis (Ultra Parallel - 96 cores)"
echo "=" * 80

# Create output directory
mkdir -p results/sensitivity logs/phase4

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"
N_CORES=96

echo ""
echo "Configuration:"
echo "  Total CPUs: $N_CORES"
echo "  Strategy: Run experiments sequentially, each using all CPUs"
echo ""

# Run motif size experiments sequentially, each using all 96 cores
echo "=" * 80
echo "Motif Size Sensitivity (k=2,3,4,5)"
echo "=" * 80

for k in 2 3 4 5; do
    echo ""
    echo "Running k=$k (using $N_CORES cores)..."

    if [ -f "results/sensitivity/k${k}.json" ]; then
        echo "  ✓ k=$k already exists, skipping"
        continue
    fi

    $PYTHON experiments/sensitivity/motif_size_parallel.py \
        --k $k \
        --datasets qm9,arxiv,reddit,protein \
        --n_cores $N_CORES \
        --output results/sensitivity/k${k}.json \
        2>&1 | tee logs/phase4/k${k}.log

    echo "  ✓ Completed k=$k"
done

echo ""
echo "=" * 80
echo "Weight Sensitivity"
echo "=" * 80

if [ -f "results/sensitivity/weights.json" ]; then
    echo "  ✓ Weights already exist, skipping"
else
    echo "Running weight sensitivity (100 trials, $N_CORES cores)..."
    $PYTHON experiments/sensitivity/weight_sensitivity.py \
        --n_trials 100 \
        --datasets qm9,arxiv,reddit,protein \
        --output results/sensitivity/weights.json \
        2>&1 | tee logs/phase4/weights.log
    echo "  ✓ Completed weight sensitivity"
fi

echo ""
echo "=" * 80
echo "Corpus Size Sensitivity"
echo "=" * 80

if [ -f "results/sensitivity/corpus_size.json" ]; then
    echo "  ✓ Corpus size already exists, skipping"
else
    echo "Running corpus size sensitivity ($N_CORES cores)..."
    $PYTHON experiments/sensitivity/corpus_size.py \
        --sizes 100,500,1000,5000,10000,50000 \
        --datasets qm9,arxiv \
        --output results/sensitivity/corpus_size.json \
        2>&1 | tee logs/phase4/corpus_size.log
    echo "  ✓ Completed corpus size sensitivity"
fi

echo ""
echo "=" * 80
echo "Hyperparameter Grid Search"
echo "=" * 80

if [ -f "results/sensitivity/grid_search.json" ]; then
    echo "  ✓ Grid search already exists, skipping"
else
    echo "Running grid search ($N_CORES cores)..."
    $PYTHON experiments/sensitivity/grid_search.py \
        --param_grid configs/param_grid.yaml \
        --datasets qm9,arxiv,reddit,protein \
        --output results/sensitivity/grid_search.json \
        2>&1 | tee logs/phase4/grid_search.log
    echo "  ✓ Completed grid search"
fi

echo ""
echo "=" * 80
echo "Phase 4 Complete!"
echo "=" * 80
echo ""
echo "Generated files:"
ls -lh results/sensitivity/*.json

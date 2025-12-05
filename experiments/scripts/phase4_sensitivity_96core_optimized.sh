#!/bin/bash

# Phase 4: Sensitivity Analysis (FULLY OPTIMIZED for 96 cores)
# Strategy: Break down motif size experiments into individual (k, dataset) jobs
# This creates 16 jobs just for motif size (4 k-values × 4 datasets)
# Plus other sensitivity analyses = maximum parallelization

set -e

echo "Starting Phase 4: Sensitivity Analysis (96-core FULLY OPTIMIZED)"

# Create output directory
mkdir -p results/sensitivity
mkdir -p results/sensitivity/motif_size

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"

# Launch all tasks in parallel
PIDS=()

# ============================================================================
# MOTIF SIZE SENSITIVITY - 16 PARALLEL JOBS (k × dataset combinations)
# ============================================================================
# Instead of running 4 jobs (one per k), we run 16 jobs (k × dataset)
# This allows us to use up to 16 cores just for motif size analysis

echo "Launching motif size sensitivity (16 parallel jobs)..."

# k=2 (4 datasets in parallel)
for dataset in qm9 arxiv reddit protein; do
    echo "  Launching k=2, dataset=${dataset}..."
    $PYTHON experiments/sensitivity/motif_size.py \
        --k 2 \
        --datasets ${dataset} \
        --output results/sensitivity/motif_size/k2_${dataset}.json &
    PIDS+=($!)
done

# k=3 (4 datasets in parallel)
for dataset in qm9 arxiv reddit protein; do
    echo "  Launching k=3, dataset=${dataset}..."
    $PYTHON experiments/sensitivity/motif_size.py \
        --k 3 \
        --datasets ${dataset} \
        --output results/sensitivity/motif_size/k3_${dataset}.json &
    PIDS+=($!)
done

# k=4 (4 datasets in parallel)
for dataset in qm9 arxiv reddit protein; do
    echo "  Launching k=4, dataset=${dataset}..."
    $PYTHON experiments/sensitivity/motif_size.py \
        --k 4 \
        --datasets ${dataset} \
        --output results/sensitivity/motif_size/k4_${dataset}.json &
    PIDS+=($!)
done

# k=5 (4 datasets in parallel)
for dataset in qm9 arxiv reddit protein; do
    echo "  Launching k=5, dataset=${dataset}..."
    $PYTHON experiments/sensitivity/motif_size.py \
        --k 5 \
        --datasets ${dataset} \
        --output results/sensitivity/motif_size/k5_${dataset}.json &
    PIDS+=($!)
done

# ============================================================================
# WEIGHT SENSITIVITY - Break into parallel jobs per dataset
# ============================================================================
echo "Launching weight sensitivity (4 parallel jobs)..."

for dataset in qm9 arxiv reddit protein; do
    echo "  Launching weight sensitivity for ${dataset}..."
    $PYTHON experiments/sensitivity/weight_sensitivity.py \
        --n_trials 100 \
        --datasets ${dataset} \
        --output results/sensitivity/weights_${dataset}.json &
    PIDS+=($!)
done

# ============================================================================
# CORPUS SIZE SENSITIVITY - Already parallelized
# ============================================================================
echo "Launching corpus size sensitivity (2 parallel jobs)..."

for dataset in qm9 arxiv; do
    echo "  Launching corpus size sensitivity for ${dataset}..."
    $PYTHON experiments/sensitivity/corpus_size.py \
        --sizes 100,500,1000,5000,10000,50000 \
        --datasets ${dataset} \
        --output results/sensitivity/corpus_size_${dataset}.json &
    PIDS+=($!)
done

# ============================================================================
# GRID SEARCH - Break into parallel jobs per dataset
# ============================================================================
echo "Launching grid search (4 parallel jobs)..."

for dataset in qm9 arxiv reddit protein; do
    echo "  Launching grid search for ${dataset}..."
    $PYTHON experiments/sensitivity/grid_search.py \
        --param_grid configs/param_grid.yaml \
        --datasets ${dataset} \
        --output results/sensitivity/grid_search_${dataset}.json &
    PIDS+=($!)
done

# ============================================================================
# Wait for all with progress tracking
# ============================================================================
echo ""
echo "Launched ${#PIDS[@]} parallel sensitivity analyses..."
echo "Expected CPU cores utilized: ${#PIDS[@]} (out of 96 available)"
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

# ============================================================================
# Merge results per k-value for motif size analysis
# ============================================================================
echo "Merging motif size results..."

for k in 2 3 4 5; do
    $PYTHON -c "
import json
from pathlib import Path

results = {}
for dataset in ['qm9', 'arxiv', 'reddit', 'protein']:
    path = Path('results/sensitivity/motif_size/k${k}_' + dataset + '.json')
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            results[dataset] = data[dataset]

with open('results/sensitivity/k${k}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'  ✓ Merged k=${k} results')
"
done

echo ""
echo "Generated files:"
ls -lh results/sensitivity/*.json
echo ""
echo "Detailed motif size results:"
ls -lh results/sensitivity/motif_size/*.json

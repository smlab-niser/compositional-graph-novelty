#!/bin/bash

# Phase 4: Sensitivity Analysis - SELECTIVE K-VALUES with MULTIPLE SEEDS
# Allows running specific k-values with full parallelization and statistical robustness
# Usage: ./phase4_sensitivity_selective_multiseed.sh [k_values]
# Example: ./phase4_sensitivity_selective_multiseed.sh 4 5
# Example: ./phase4_sensitivity_selective_multiseed.sh 2 3 4 5

set -e

# Parse k-values from command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <k_values...>"
    echo "Example: $0 4 5        (run k=4 and k=5 with 4 seeds each)"
    echo "Example: $0 2 3 4 5    (run all k-values with 4 seeds each)"
    exit 1
fi

K_VALUES=("$@")
SEEDS=(0 1 2 3)

echo "=========================================================================="
echo "Starting Phase 4: Sensitivity Analysis (Selective K-values + Multi-Seed)"
echo "K-values to test: ${K_VALUES[@]}"
echo "Random seeds: ${SEEDS[@]}"
echo "=========================================================================="

# Create output directory
mkdir -p results/sensitivity
mkdir -p results/sensitivity/motif_size

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"

# Launch all tasks in parallel
PIDS=()

# ============================================================================
# MOTIF SIZE SENSITIVITY - SELECTIVE K-VALUES with MULTIPLE SEEDS
# ============================================================================
echo ""
echo "Launching motif size sensitivity..."
echo "  K-values: ${K_VALUES[@]}"
echo "  Datasets: qm9, arxiv, reddit, protein"
echo "  Seeds: ${SEEDS[@]}"
echo ""

JOB_COUNT=0

for k in "${K_VALUES[@]}"; do
    for dataset in qm9 arxiv reddit protein; do
        for seed in "${SEEDS[@]}"; do
            echo "  Launching k=${k}, dataset=${dataset}, seed=${seed}..."
            $PYTHON experiments/sensitivity/motif_size.py \
                --k $k \
                --datasets ${dataset} \
                --output results/sensitivity/motif_size/k${k}_${dataset}_seed${seed}.json &
            PIDS+=($!)
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

# ============================================================================
# Wait for all with progress tracking
# ============================================================================
echo ""
echo "=========================================================================="
echo "Launched ${JOB_COUNT} parallel jobs"
echo "  (${#K_VALUES[@]} k-values × 4 datasets × ${#SEEDS[@]} seeds)"
echo "Expected CPU cores utilized: ${JOB_COUNT} (out of 96 available)"
echo "=========================================================================="
echo "Waiting for completion..."
echo ""

COMPLETED=0
TOTAL=${#PIDS[@]}

for pid in "${PIDS[@]}"; do
    wait $pid
    COMPLETED=$((COMPLETED + 1))

    # Progress indicator every 10%
    if [ $((COMPLETED % 10)) -eq 0 ] || [ $COMPLETED -eq $TOTAL ]; then
        PERCENT=$((COMPLETED * 100 / TOTAL))
        echo "  ✓ Completed $COMPLETED/$TOTAL analyses (${PERCENT}%)"
    fi
done

echo ""
echo "=========================================================================="
echo "All analyses complete!"
echo "=========================================================================="
echo ""

# ============================================================================
# Aggregate results across seeds
# ============================================================================
echo "Aggregating results across random seeds..."

for k in "${K_VALUES[@]}"; do
    echo "  Processing k=${k}..."
    $PYTHON -c "
import json
import numpy as np
from pathlib import Path

results = {}

for dataset in ['qm9', 'arxiv', 'reddit', 'protein']:
    # Collect results from all seeds
    seed_results = []
    for seed in [0, 1, 2, 3]:
        path = Path(f'results/sensitivity/motif_size/k${k}_{dataset}_seed{seed}.json')
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                if dataset in data:
                    seed_results.append(data[dataset])

    if seed_results:
        # Aggregate statistics across seeds
        novelties = [r['mean_novelty'] for r in seed_results]
        init_times = [r['init_time'] for r in seed_results]
        eval_times = [r['eval_time'] for r in seed_results]
        n_motifs = [r['n_unique_motifs'] for r in seed_results]

        results[dataset] = {
            'k': ${k},
            'mean_novelty': float(np.mean(novelties)),
            'std_novelty': float(np.std(novelties)),
            'median_novelty': float(np.median(novelties)),
            'min_novelty': float(np.min(novelties)),
            'max_novelty': float(np.max(novelties)),
            'mean_init_time': float(np.mean(init_times)),
            'mean_eval_time': float(np.mean(eval_times)),
            'mean_n_motifs': float(np.mean(n_motifs)),
            'n_seeds': len(seed_results),
            'individual_seeds': seed_results
        }

# Save aggregated results
if results:
    with open('results/sensitivity/k${k}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'    ✓ Aggregated k=${k} results across {len(seed_results)} seeds')
else:
    print(f'    ✗ No results found for k=${k}')
"
done

echo ""
echo "=========================================================================="
echo "Results Summary:"
echo "=========================================================================="
echo ""
echo "Aggregated results (across seeds):"
for k in "${K_VALUES[@]}"; do
    if [ -f "results/sensitivity/k${k}.json" ]; then
        SIZE=$(ls -lh results/sensitivity/k${k}.json | awk '{print $5}')
        echo "  k=${k}: $SIZE"
    fi
done

echo ""
echo "Individual seed results:"
N_FILES=$(ls results/sensitivity/motif_size/*.json 2>/dev/null | wc -l)
echo "  Total files: ${N_FILES}"
for k in "${K_VALUES[@]}"; do
    N_K=$(ls results/sensitivity/motif_size/k${k}_*.json 2>/dev/null | wc -l)
    echo "  k=${k}: ${N_K} files ($(($N_K / 4)) datasets × 4 seeds)"
done

echo ""
echo "=========================================================================="
echo "Done! Results saved in results/sensitivity/"
echo "  - Aggregated: results/sensitivity/k*.json"
echo "  - Individual: results/sensitivity/motif_size/k*_*_seed*.json"
echo "=========================================================================="

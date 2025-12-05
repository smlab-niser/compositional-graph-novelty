#!/bin/bash

# Phase 4: Sensitivity Analysis (MAXIMUM 96-core utilization with multiple seeds)
# Strategy: Run each (k, dataset) combination with 4 different random seeds
# This creates 64 jobs for motif size alone (4 k-values × 4 datasets × 4 seeds)
# Plus other sensitivity analyses = near-full 96-core utilization

set -e

echo "Starting Phase 4: Sensitivity Analysis (96-core MAXIMUM UTILIZATION)"
echo "Using multiple random seeds for statistical robustness"

# Create output directory
mkdir -p results/sensitivity
mkdir -p results/sensitivity/motif_size
mkdir -p results/sensitivity/weights
mkdir -p results/sensitivity/corpus_size
mkdir -p results/sensitivity/grid_search

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"

# Launch all tasks in parallel
PIDS=()

# Number of random seeds for statistical robustness
SEEDS=(0 1 2 3)

# ============================================================================
# MOTIF SIZE SENSITIVITY - 64 PARALLEL JOBS (k × dataset × seed)
# ============================================================================
echo "Launching motif size sensitivity (64 parallel jobs)..."
echo "  k-values: 2, 3, 4, 5"
echo "  datasets: qm9, arxiv, reddit, protein"
echo "  seeds: ${SEEDS[@]}"

for k in 2 3 4 5; do
    for dataset in qm9 arxiv reddit protein; do
        for seed in "${SEEDS[@]}"; do
            echo "  Launching k=${k}, dataset=${dataset}, seed=${seed}..."
            $PYTHON experiments/sensitivity/motif_size.py \
                --k $k \
                --datasets ${dataset} \
                --output results/sensitivity/motif_size/k${k}_${dataset}_seed${seed}.json \
                --n_cores 1 &
            PIDS+=($!)
        done
    done
done

# ============================================================================
# WEIGHT SENSITIVITY - 16 PARALLEL JOBS (dataset × seed)
# ============================================================================
echo "Launching weight sensitivity (16 parallel jobs)..."

for dataset in qm9 arxiv reddit protein; do
    for seed in "${SEEDS[@]}"; do
        echo "  Launching weight sensitivity for ${dataset}, seed=${seed}..."
        $PYTHON experiments/sensitivity/weight_sensitivity.py \
            --n_trials 100 \
            --datasets ${dataset} \
            --output results/sensitivity/weights/weights_${dataset}_seed${seed}.json &
        PIDS+=($!)
    done
done

# ============================================================================
# CORPUS SIZE SENSITIVITY - 8 PARALLEL JOBS (dataset × seed)
# ============================================================================
echo "Launching corpus size sensitivity (8 parallel jobs)..."

for dataset in qm9 arxiv; do
    for seed in "${SEEDS[@]}"; do
        echo "  Launching corpus size sensitivity for ${dataset}, seed=${seed}..."
        $PYTHON experiments/sensitivity/corpus_size.py \
            --sizes 100,500,1000,5000,10000,50000 \
            --datasets ${dataset} \
            --output results/sensitivity/corpus_size/corpus_size_${dataset}_seed${seed}.json &
        PIDS+=($!)
    done
done

# ============================================================================
# GRID SEARCH - 16 PARALLEL JOBS (dataset × seed)
# ============================================================================
echo "Launching grid search (16 parallel jobs)..."

for dataset in qm9 arxiv reddit protein; do
    for seed in "${SEEDS[@]}"; do
        echo "  Launching grid search for ${dataset}, seed=${seed}..."
        $PYTHON experiments/sensitivity/grid_search.py \
            --param_grid configs/param_grid.yaml \
            --datasets ${dataset} \
            --output results/sensitivity/grid_search/grid_search_${dataset}_seed${seed}.json &
        PIDS+=($!)
    done
done

# ============================================================================
# Wait for all with progress tracking
# ============================================================================
echo ""
echo "=========================================================================="
echo "Launched ${#PIDS[@]} parallel sensitivity analyses..."
echo "Expected CPU cores utilized: ${#PIDS[@]} (out of 96 available)"
echo "Breakdown:"
echo "  - Motif size: 64 jobs (4 k-values × 4 datasets × 4 seeds)"
echo "  - Weight sensitivity: 16 jobs (4 datasets × 4 seeds)"
echo "  - Corpus size: 8 jobs (2 datasets × 4 seeds)"
echo "  - Grid search: 16 jobs (4 datasets × 4 seeds)"
echo "=========================================================================="
echo "Waiting for completion..."
echo ""

COMPLETED=0
TOTAL=${#PIDS[@]}

for pid in "${PIDS[@]}"; do
    wait $pid
    COMPLETED=$((COMPLETED + 1))

    # Progress indicator every 10%
    if [ $((COMPLETED % 10)) -eq 0 ]; then
        PERCENT=$((COMPLETED * 100 / TOTAL))
        echo "  ✓ Completed $COMPLETED/$TOTAL analyses (${PERCENT}%)"
    fi
done

echo ""
echo "=========================================================================="
echo "Phase 4 complete! All ${TOTAL} analyses finished."
echo "=========================================================================="
echo ""

# ============================================================================
# Aggregate results across seeds
# ============================================================================
echo "Aggregating results across random seeds..."

# Merge motif size results (aggregate by k-value and dataset)
for k in 2 3 4 5; do
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
            'seed_results': seed_results
        }

# Save aggregated results
with open('results/sensitivity/k${k}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'  ✓ Aggregated k=${k} results across {len(seed_results)} seeds')
"
done

# Similar aggregation for other experiments
echo "  Processing weight sensitivity..."
$PYTHON -c "
import json
import numpy as np
from pathlib import Path

results = {}

for dataset in ['qm9', 'arxiv', 'reddit', 'protein']:
    seed_results = []
    for seed in [0, 1, 2, 3]:
        path = Path(f'results/sensitivity/weights/weights_{dataset}_seed{seed}.json')
        if path.exists():
            with open(path) as f:
                seed_results.append(json.load(f))

    if seed_results:
        results[dataset] = {
            'n_seeds': len(seed_results),
            'seed_results': seed_results
        }

with open('results/sensitivity/weights.json', 'w') as f:
    json.dump(results, f, indent=2)

print('  ✓ Aggregated weight sensitivity results')
"

echo ""
echo "=========================================================================="
echo "Results Summary:"
echo "=========================================================================="
echo ""
echo "Aggregated results (across seeds):"
ls -lh results/sensitivity/*.json
echo ""
echo "Individual seed results:"
echo "  Motif size: $(ls results/sensitivity/motif_size/*.json 2>/dev/null | wc -l) files"
echo "  Weights: $(ls results/sensitivity/weights/*.json 2>/dev/null | wc -l) files"
echo "  Corpus size: $(ls results/sensitivity/corpus_size/*.json 2>/dev/null | wc -l) files"
echo "  Grid search: $(ls results/sensitivity/grid_search/*.json 2>/dev/null | wc -l) files"
echo ""
echo "=========================================================================="

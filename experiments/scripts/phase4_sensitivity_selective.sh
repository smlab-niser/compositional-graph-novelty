#!/bin/bash

# Phase 4: Sensitivity Analysis - SELECTIVE K-VALUES
# Allows running specific k-values with full parallelization
# Usage: ./phase4_sensitivity_selective.sh [k_values]
# Example: ./phase4_sensitivity_selective.sh 4 5
# Example: ./phase4_sensitivity_selective.sh 2 3 4 5

set -e

# Parse k-values from command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <k_values...>"
    echo "Example: $0 4 5"
    echo "Example: $0 2 3 4 5"
    exit 1
fi

K_VALUES=("$@")

echo "=========================================================================="
echo "Starting Phase 4: Sensitivity Analysis (Selective K-values)"
echo "K-values to test: ${K_VALUES[@]}"
echo "=========================================================================="

# Create output directory
mkdir -p results/sensitivity
mkdir -p results/sensitivity/motif_size

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"

# Launch all tasks in parallel
PIDS=()

# ============================================================================
# MOTIF SIZE SENSITIVITY - SELECTIVE K-VALUES
# ============================================================================
echo ""
echo "Launching motif size sensitivity..."
echo "  K-values: ${K_VALUES[@]}"
echo "  Datasets: qm9, arxiv, reddit, protein"
echo ""

JOB_COUNT=0

for k in "${K_VALUES[@]}"; do
    for dataset in qm9 arxiv reddit protein; do
        echo "  Launching k=${k}, dataset=${dataset}..."
        $PYTHON experiments/sensitivity/motif_size.py \
            --k $k \
            --datasets ${dataset} \
            --output results/sensitivity/motif_size/k${k}_${dataset}.json &
        PIDS+=($!)
        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

# ============================================================================
# Wait for all with progress tracking
# ============================================================================
echo ""
echo "=========================================================================="
echo "Launched ${JOB_COUNT} parallel jobs (${#K_VALUES[@]} k-values × 4 datasets)"
echo "Expected CPU cores utilized: ${JOB_COUNT} (out of 96 available)"
echo "=========================================================================="
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
echo "=========================================================================="
echo "All analyses complete!"
echo "=========================================================================="
echo ""

# ============================================================================
# Merge results per k-value
# ============================================================================
echo "Merging results by k-value..."

for k in "${K_VALUES[@]}"; do
    $PYTHON -c "
import json
from pathlib import Path

results = {}
for dataset in ['qm9', 'arxiv', 'reddit', 'protein']:
    path = Path('results/sensitivity/motif_size/k${k}_' + dataset + '.json')
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            if dataset in data:
                results[dataset] = data[dataset]

if results:
    with open('results/sensitivity/k${k}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  ✓ Merged k=${k} results ({len(results)} datasets)')
else:
    print(f'  ✗ No results found for k=${k}')
"
done

echo ""
echo "=========================================================================="
echo "Results Summary:"
echo "=========================================================================="
echo ""
echo "Aggregated results by k-value:"
for k in "${K_VALUES[@]}"; do
    if [ -f "results/sensitivity/k${k}.json" ]; then
        echo "  k=${k}: $(ls -lh results/sensitivity/k${k}.json | awk '{print $5}')"
    fi
done

echo ""
echo "Individual dataset results:"
ls -lh results/sensitivity/motif_size/*.json 2>/dev/null || echo "  (none)"

echo ""
echo "=========================================================================="
echo "Done! Results saved in results/sensitivity/"
echo "=========================================================================="

#!/bin/bash

# Phase 5: Predictive Validation (96-core optimized)
# Validate metric through correlation with ground-truth signals:
# - Molecular synthesis difficulty (SA-scores)
# - Citation impact (future citations)
# - Downstream classification performance
# - Expert ranking agreement

set -e

echo "Starting Phase 5: Predictive Validation (96-core optimized)"

# Create output directory
mkdir -p results/validation

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"

# Launch all validation tasks in parallel
PIDS=()

echo "Launching molecular synthesis correlation..."
$PYTHON experiments/validation/molecular_synthesis.py \
    --dataset qm9 \
    --compute_sa_scores \
    --n_molecules 50000 \
    --output results/validation/molecular_synthesis.json &
PIDS+=($!)

echo "Launching citation impact correlation..."
$PYTHON experiments/validation/citation_impact.py \
    --dataset arxiv \
    --citation_window_years 2 \
    --control_variables author_h_index,venue,year \
    --output results/validation/citation_impact.json &
PIDS+=($!)

echo "Launching synthetic graph downstream performance..."
$PYTHON experiments/validation/synthetic_downstream.py \
    --datasets reddit,protein \
    --task node_classification \
    --n_trials 10 \
    --novelty_ranges 0.0-0.3,0.3-0.5,0.5-0.7,0.7-1.0 \
    --output results/validation/synthetic_downstream.json &
PIDS+=($!)

echo "Launching expert ranking collection..."
$PYTHON experiments/validation/expert_ranking.py \
    --n_experts 5 \
    --n_pairs 30 \
    --datasets qm9,arxiv \
    --output results/validation/expert_ranking.json &
PIDS+=($!)

# Wait for all with progress tracking
echo ""
echo "Launched ${#PIDS[@]} parallel validation tasks..."
echo "Waiting for completion..."
echo ""

COMPLETED=0
TOTAL=${#PIDS[@]}

for pid in "${PIDS[@]}"; do
    wait $pid
    COMPLETED=$((COMPLETED + 1))
    echo "  ✓ Completed $COMPLETED/$TOTAL validation tasks"
done

echo ""
echo "Phase 5 complete!"
echo "Results saved in results/validation/"
echo ""

# Compute summary statistics
echo "Computing validation summary..."
$PYTHON experiments/validation/compute_correlations.py \
    --results_dir results/validation/ \
    --output results/validation/summary.json

echo ""
echo "Validation summary saved to results/validation/summary.json"
echo ""

# List all results
echo "Generated files:"
ls -lh results/validation/*.json

#!/bin/bash

# Phase 5: Predictive Validation
# Validate metric through correlation with ground-truth signals:
# - Molecular synthesis difficulty (SA-scores)
# - Citation impact (future citations)
# - Downstream classification performance
# - Expert ranking agreement

set -e

echo "Starting Phase 5: Predictive Validation"

# Create output directory
mkdir -p results/validation

echo "GPU 0: Molecular synthesis correlation..."
CUDA_VISIBLE_DEVICES=0 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/validation/molecular_synthesis.py \
    --dataset qm9 \
    --compute_sa_scores \
    --n_molecules 50000 \
    --output results/validation/molecular_synthesis.json &
PID1=$!

echo "GPU 1: Citation impact correlation..."
CUDA_VISIBLE_DEVICES=1 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/validation/citation_impact.py \
    --dataset arxiv \
    --citation_window_years 2 \
    --control_variables author_h_index,venue,year \
    --output results/validation/citation_impact.json &
PID2=$!

echo "GPU 2: Synthetic graph downstream performance..."
CUDA_VISIBLE_DEVICES=2 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/validation/synthetic_downstream.py \
    --datasets reddit,protein \
    --task node_classification \
    --n_trials 10 \
    --novelty_ranges 0.0-0.3,0.3-0.5,0.5-0.7,0.7-1.0 \
    --output results/validation/synthetic_downstream.json &
PID3=$!

echo "GPU 3: Expert ranking collection..."
CUDA_VISIBLE_DEVICES=3 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/validation/expert_ranking.py \
    --n_experts 5 \
    --n_pairs 30 \
    --datasets qm9,arxiv \
    --output results/validation/expert_ranking.json &
PID4=$!

# Wait for all
echo "Waiting for all validation tasks to complete..."

wait $PID1
echo "✓ GPU 0 (molecular synthesis) complete"

wait $PID2
echo "✓ GPU 1 (citation impact) complete"

wait $PID3
echo "✓ GPU 2 (downstream performance) complete"

wait $PID4
echo "✓ GPU 3 (expert ranking) complete"

echo "Phase 5 complete!"
echo "Results saved in results/validation/"

# Compute summary statistics
echo "Computing validation summary..."
/home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/validation/compute_correlations.py \
    --results_dir results/validation/ \
    --output results/validation/summary.json

echo "Validation summary saved to results/validation/summary.json"

#!/bin/bash

# Phase 3: Main Experiments
# Run GCN evaluation with 10 random seeds on all datasets
# Utilize all 4 GPUs in parallel

set -e

echo "Starting Phase 3: Main Experiments"
echo "Running 10 seeds across 4 datasets on 4 GPUs..."

# Create output directory
mkdir -p results/gcn

# Function to run experiments for a dataset on a specific GPU
run_dataset_seeds() {
    local GPU=$1
    local DATASET=$2

    echo "GPU $GPU: Starting $DATASET experiments"

    for seed in {0..9}; do
        CUDA_VISIBLE_DEVICES=$GPU /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/evaluate_gcn.py \
            --dataset $DATASET \
            --seed $seed \
            --k 3 \
            --w_structural 0.4 \
            --w_edge 0.3 \
            --w_bridging 0.3 \
            --output results/gcn/${DATASET}_seed${seed}.json

        echo "GPU $GPU: Completed $DATASET seed $seed"
    done

    echo "GPU $GPU: Completed all $DATASET experiments"
}

# Launch parallel experiments (one dataset per GPU)
run_dataset_seeds 0 qm9 &
PID1=$!

run_dataset_seeds 1 arxiv &
PID2=$!

run_dataset_seeds 2 reddit &
PID3=$!

run_dataset_seeds 3 protein &
PID4=$!

# Wait for all to complete
echo "Waiting for all experiments to complete..."
wait $PID1
echo "✓ GPU 0 (qm9) complete"

wait $PID2
echo "✓ GPU 1 (arxiv) complete"

wait $PID3
echo "✓ GPU 2 (reddit) complete"

wait $PID4
echo "✓ GPU 3 (protein) complete"

echo "Phase 3 complete! All 40 experiments finished."
echo "Results saved in results/gcn/"

# Generate summary
/home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/analysis/summarize_seeds.py \
    --results_dir results/gcn/ \
    --output results/gcn/summary.json

echo "Summary saved to results/gcn/summary.json"

#!/bin/bash

# Phase 2: Baseline Computation
# Compute all baseline metrics (MMD, GED, KERNEL, EMB, SET)
# GPU 0: Baselines (most compute-intensive)
# GPUs 1-3: Start application experiments

set -e

echo "Starting Phase 2: Baseline Computation"

# Create output directories
mkdir -p results/baselines/{qm9,arxiv,reddit,protein}
mkdir -p results/molecular
mkdir -p results/explainability
mkdir -p results/synthetic

echo "GPU 0: Computing all baselines..."
CUDA_VISIBLE_DEVICES=0 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/baselines/run_all_baselines.py \
    --datasets qm9,arxiv,reddit,protein \
    --baselines MMD,GED,KERNEL,EMB,SET \
    --corpus_size 90 \
    --eval_size 10 \
    --output results/baselines/ &
BASELINE_PID=$!

echo "GPU 1: Starting molecular generation..."
CUDA_VISIBLE_DEVICES=1 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/molecular/generate_molecules.py \
    --n_samples 10000 \
    --novelty_constraint 0.5,0.7 \
    --output results/molecular/generation.pkl &
MOLECULAR_PID=$!

echo "GPU 2: Starting GNNExplainer experiments..."
CUDA_VISIBLE_DEVICES=2 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/explainability/run_gnnexplainer.py \
    --dataset arxiv \
    --model gcn \
    --n_explanations 1000 \
    --output results/explainability/gnnexplainer.pkl &
EXPLAINER_PID=$!

echo "GPU 3: Starting GraphRNN generation..."
CUDA_VISIBLE_DEVICES=3 /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/synthetic/run_graphrnn.py \
    --dataset qm9 \
    --n_samples 5000 \
    --output results/synthetic/graphrnn.pkl &
SYNTHETIC_PID=$!

# Wait for all to complete
echo "Waiting for all Phase 2 tasks to complete..."

wait $BASELINE_PID
echo "✓ GPU 0 (baselines) complete"

wait $MOLECULAR_PID
echo "✓ GPU 1 (molecular generation) complete"

wait $EXPLAINER_PID
echo "✓ GPU 2 (GNNExplainer) complete"

wait $SYNTHETIC_PID
echo "✓ GPU 3 (GraphRNN) complete"

echo "Phase 2 complete!"

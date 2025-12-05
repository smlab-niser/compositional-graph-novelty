#!/bin/bash

# Evaluate all available generation models
# This script evaluates existing pre-generated graphs

set -e

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"
BASE_DIR="/home/smlab/projects/graph-novelty"

cd $BASE_DIR

echo "=========================================================================="
echo "Evaluating All Available Generation Models"
echo "=========================================================================="
echo ""

# Create output directory
mkdir -p results/generation

# ============================================================================
# Model 1: GraphRNN (from synthetic directory)
# ============================================================================
if [ -f "results/synthetic/graphrnn.pkl" ]; then
    echo "Model 1: Evaluating GraphRNN..."
    $PYTHON experiments/generation/evaluate_generated.py \
        --model "GraphRNN" \
        --dataset qm9 \
        --input results/synthetic/graphrnn.pkl \
        --output results/generation/graphrnn_qm9.json \
        --corpus_size 9000 \
        --k 3
    echo "✓ GraphRNN completed"
    echo ""
else
    echo "⚠ GraphRNN file not found"
fi

# ============================================================================
# Model 2: Molecular Generation (from molecular directory)
# ============================================================================
if [ -f "results/molecular/generation.pkl" ]; then
    echo "Model 2: Evaluating Molecular Generation..."
    $PYTHON experiments/generation/evaluate_generated.py \
        --model "MolecularGeneration" \
        --dataset qm9 \
        --input results/molecular/generation.pkl \
        --output results/generation/molecular_gen_qm9.json \
        --corpus_size 9000 \
        --k 3
    echo "✓ Molecular Generation completed"
    echo ""
else
    echo "⚠ Molecular generation file not found"
fi

# ============================================================================
# Model 3: Held-Out Test Set (baseline - already done)
# ============================================================================
if [ -f "results/generation/testset_qm9.json" ]; then
    echo "✓ Test set baseline already evaluated"
else
    echo "Model 3: Evaluating Test Set Baseline..."
    # Create test set
    $PYTHON -c "
import sys
from pathlib import Path
import pickle

sys.path.insert(0, 'src')
from utils.data_loader import load_dataset

print('  Loading test set...')
_, test_graphs = load_dataset('qm9', corpus_size=9000, eval_size=1000)
print(f'  Test set: {len(test_graphs)} graphs')

with open('results/generation/test_graphs_qm9.pkl', 'wb') as f:
    pickle.dump(test_graphs, f)
print('  ✓ Saved test graphs')
"

    $PYTHON experiments/generation/evaluate_generated.py \
        --model "HeldOutTestSet" \
        --dataset qm9 \
        --input results/generation/test_graphs_qm9.pkl \
        --output results/generation/testset_qm9.json \
        --corpus_size 9000 \
        --k 3
    echo "✓ Test set completed"
fi
echo ""

# ============================================================================
# Generate comparison summary
# ============================================================================
echo "=========================================================================="
echo "Generating Comparison Summary"
echo "=========================================================================="
echo ""

$PYTHON experiments/generation/compare_generation_models.py \
    --results_dir results/generation \
    --output_dir results/generation/comparison

echo ""
echo "=========================================================================="
echo "✓ All evaluations completed!"
echo "=========================================================================="
echo ""
echo "Results available in:"
echo "  - results/generation/*.json (individual model results)"
echo "  - results/generation/comparison/ (plots and tables)"
echo ""

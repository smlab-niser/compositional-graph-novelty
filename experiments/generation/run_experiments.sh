#!/bin/bash

# Run graph generation model evaluation experiments
# This script evaluates novelty on graphs from different generation approaches

set -e

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"
BASE_DIR="/home/smlab/projects/graph-novelty"

cd $BASE_DIR

echo "=========================================================================="
echo "Graph Generation Model Evaluation"
echo "=========================================================================="
echo ""

# Create output directory
mkdir -p results/generation

# ============================================================================
# Experiment 1: Evaluate existing generated molecules
# ============================================================================
echo "Experiment 1: Evaluating existing generated molecules..."
if [ -f "results/molecular/generation.pkl" ]; then
    $PYTHON experiments/generation/evaluate_generated.py \
        --model "MolecularGeneration" \
        --dataset qm9 \
        --input results/molecular/generation.pkl \
        --output results/generation/molecular_gen_qm9.json \
        --corpus_size 900 \
        --k 3
    echo "✓ Completed"
else
    echo "⚠ File not found: results/molecular/generation.pkl"
fi
echo ""

# ============================================================================
# Experiment 2: Evaluate test set (upper bound)
# ============================================================================
echo "Experiment 2: Generating test set baseline..."
$PYTHON -c "
import sys
from pathlib import Path
import pickle

sys.path.insert(0, 'src')
from utils.data_loader import load_dataset

# Load test graphs
print('  Loading test set...')
_, test_graphs = load_dataset('qm9', corpus_size=900, eval_size=100)
print(f'  Test set: {len(test_graphs)} graphs')

# Save
with open('results/generation/test_graphs_qm9.pkl', 'wb') as f:
    pickle.dump(test_graphs, f)
print('  ✓ Saved test graphs')
"

echo "Evaluating test set..."
$PYTHON experiments/generation/evaluate_generated.py \
    --model "TestSet" \
    --dataset qm9 \
    --input results/generation/test_graphs_qm9.pkl \
    --output results/generation/testset_qm9.json \
    --corpus_size 900 \
    --k 3
echo "✓ Completed"
echo ""

# ============================================================================
# Experiment 3: Random generation (lower bound)
# ============================================================================
echo "Experiment 3: Generating random molecular graphs..."
$PYTHON -c "
import networkx as nx
import pickle
import random

def generate_random_molecular_graph(n_atoms=15):
    '''Generate random molecular-like graph'''
    G = nx.Graph()
    atoms = ['C', 'N', 'O', 'F', 'S', 'Cl']

    # Add atoms
    for i in range(n_atoms):
        G.add_node(i, atom_type=random.choice(atoms))

    # Add bonds (ensure connected)
    # First create a path to ensure connectivity
    for i in range(n_atoms - 1):
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']
        G.add_edge(i, i+1, bond_type=random.choice(bond_types))

    # Add random additional bonds
    for i in range(n_atoms):
        for j in range(i+2, n_atoms):
            if random.random() < 0.1:  # 10% additional edges
                bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']
                G.add_edge(i, j, bond_type=random.choice(bond_types))

    return G

print('  Generating 1000 random molecular graphs...')
random.seed(42)
random_graphs = [generate_random_molecular_graph(n_atoms=random.randint(10, 20))
                 for _ in range(1000)]
print(f'  Generated: {len(random_graphs)} graphs')

with open('results/generation/random_graphs_qm9.pkl', 'wb') as f:
    pickle.dump(random_graphs, f)
print('  ✓ Saved random graphs')
"

echo "Evaluating random graphs..."
$PYTHON experiments/generation/evaluate_generated.py \
    --model "RandomGeneration" \
    --dataset qm9 \
    --input results/generation/random_graphs_qm9.pkl \
    --output results/generation/random_qm9.json \
    --corpus_size 900 \
    --k 3
echo "✓ Completed"
echo ""

# ============================================================================
# Generate comparison summary
# ============================================================================
echo "=========================================================================="
echo "Generating comparison summary..."
echo "=========================================================================="

$PYTHON -c "
import json
from pathlib import Path

# Load all results
results = {}
result_files = {
    'Molecular Generation': 'results/generation/molecular_gen_qm9.json',
    'Test Set': 'results/generation/testset_qm9.json',
    'Random Generation': 'results/generation/random_qm9.json'
}

for model, filepath in result_files.items():
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            results[model] = json.load(f)

# Print comparison table
print()
print('='*80)
print('COMPARISON SUMMARY')
print('='*80)
print()
print(f'{'Model':<25} {'Overall':<12} {'Structural':<12} {'Edge-Type':<12} {'Bridging':<12}')
print('-'*80)

for model, data in results.items():
    overall = data['overall_novelty']['mean']
    overall_std = data['overall_novelty']['std']
    struct = data['structural_novelty']['mean']
    edge = data['edge_type_novelty']['mean']
    bridge = data['bridging_novelty']['mean']

    print(f'{model:<25} {overall:.3f}±{overall_std:.3f}  {struct:.3f}        {edge:.3f}        {bridge:.3f}')

print()
print('='*80)
print('Results saved in results/generation/')
print('='*80)
"

echo ""
echo "✓ All experiments completed!"
echo ""
echo "Results files:"
ls -lh results/generation/*.json

echo ""
echo "=========================================================================="
echo "Next steps:"
echo "1. Review results in results/generation/"
echo "2. Add subsection to paper Results section"
echo "3. Optional: Generate comparison figure"
echo "=========================================================================="

#!/bin/bash

# Simplified graph generation evaluation
# Uses test sets and synthetic generation for comparison

set -e

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"
BASE_DIR="/home/smlab/projects/graph-novelty"

cd $BASE_DIR

echo "=========================================================================="
echo "Graph Generation Model Evaluation (Simplified)"
echo "=========================================================================="
echo ""

# Create output directory
mkdir -p results/generation

# ============================================================================
# Experiment 1: Evaluate test set (held-out real graphs)
# ============================================================================
echo "Experiment 1: Evaluating held-out test graphs (baseline)..."
$PYTHON experiments/generation/evaluate_generated.py \
    --model "HeldOutTestSet" \
    --dataset qm9 \
    --input results/generation/test_graphs_qm9.pkl \
    --output results/generation/testset_qm9.json \
    --corpus_size 900 \
    --k 3 2>&1 || {
    echo "Creating test set first..."
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

    echo "Evaluating test set..."
    $PYTHON experiments/generation/evaluate_generated.py \
        --model "HeldOutTestSet" \
        --dataset qm9 \
        --input results/generation/test_graphs_qm9.pkl \
        --output results/generation/testset_qm9.json \
        --corpus_size 9000 \
        --k 3
}
echo "✓ Completed"
echo ""

# ============================================================================
# Experiment 2: Random molecular generation (lower bound)
# ============================================================================
echo "Experiment 2: Generating random molecular graphs..."
$PYTHON -c "
import networkx as nx
import pickle
import random

def generate_random_molecular_graph(n_atoms=15):
    '''Generate random molecular-like graph'''
    G = nx.Graph()
    atoms = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br']

    # Add atoms
    for i in range(n_atoms):
        G.add_node(i, atom_type=random.choice(atoms))

    # Create spanning tree first to ensure connectivity
    import random
    nodes = list(range(n_atoms))
    random.shuffle(nodes)
    for i in range(n_atoms - 1):
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        G.add_edge(nodes[i], nodes[i+1], bond_type=random.choice(bond_types))

    # Add some random additional edges
    for _ in range(n_atoms // 3):
        i, j = random.sample(range(n_atoms), 2)
        if not G.has_edge(i, j):
            bond_types = ['SINGLE', 'DOUBLE']
            G.add_edge(i, j, bond_type=random.choice(bond_types))

    return G

print('  Generating 1000 random molecular graphs...')
random.seed(42)
random_graphs = [generate_random_molecular_graph(n_atoms=random.randint(8, 25))
                 for _ in range(1000)]
print(f'  Generated: {len(random_graphs)} graphs')

# Save
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
    --corpus_size 9000 \
    --k 3
echo "✓ Completed"
echo ""

# ============================================================================
# Experiment 3: Perturbed molecules (controlled novelty)
# ============================================================================
echo "Experiment 3: Generating perturbed molecular graphs..."
$PYTHON -c "
import sys
from pathlib import Path
import pickle
import random
import networkx as nx

sys.path.insert(0, 'src')
from utils.data_loader import load_dataset

print('  Loading base graphs...')
_, base_graphs = load_dataset('qm9', corpus_size=9000, eval_size=100)

def perturb_graph(G, perturbation_rate=0.2):
    '''Create a perturbed version of a graph'''
    G_new = G.copy()
    n_nodes = G_new.number_of_nodes()

    # Perturb some node types
    nodes_to_perturb = random.sample(list(G_new.nodes()),
                                    max(1, int(n_nodes * perturbation_rate)))
    atoms = ['C', 'N', 'O', 'F', 'S', 'Cl']
    for node in nodes_to_perturb:
        G_new.nodes[node]['atom_type'] = random.choice(atoms)

    # Perturb some edge types
    if G_new.number_of_edges() > 0:
        edges = list(G_new.edges())
        edges_to_perturb = random.sample(edges,
                                        max(1, int(len(edges) * perturbation_rate)))
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']
        for u, v in edges_to_perturb:
            G_new.edges[u, v]['bond_type'] = random.choice(bond_types)

    return G_new

print('  Generating 1000 perturbed molecular graphs...')
random.seed(42)
perturbed_graphs = [perturb_graph(random.choice(base_graphs))
                    for _ in range(1000)]
print(f'  Generated: {len(perturbed_graphs)} graphs')

with open('results/generation/perturbed_graphs_qm9.pkl', 'wb') as f:
    pickle.dump(perturbed_graphs, f)
print('  ✓ Saved perturbed graphs')
"

echo "Evaluating perturbed graphs..."
$PYTHON experiments/generation/evaluate_generated.py \
    --model "PerturbedGeneration" \
    --dataset qm9 \
    --input results/generation/perturbed_graphs_qm9.pkl \
    --output results/generation/perturbed_qm9.json \
    --corpus_size 9000 \
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
    'Held-Out Test Set': 'results/generation/testset_qm9.json',
    'Perturbed Generation': 'results/generation/perturbed_qm9.json',
    'Random Generation': 'results/generation/random_qm9.json'
}

for model, filepath in result_files.items():
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            results[model] = json.load(f)

# Print comparison table
print()
print('='*90)
print('COMPARISON SUMMARY - Graph Generation Novelty Evaluation')
print('='*90)
print()
print(f'{'Model':<25} {'Overall':<15} {'Structural':<15} {'Edge-Type':<15} {'Bridging':<12}')
print('-'*90)

for model in ['Held-Out Test Set', 'Perturbed Generation', 'Random Generation']:
    if model in results:
        data = results[model]
        overall = data['overall_novelty']['mean']
        overall_std = data['overall_novelty']['std']
        struct = data['structural_novelty']['mean']
        struct_std = data['structural_novelty']['std']
        edge = data['edge_type_novelty']['mean']
        edge_std = data['edge_type_novelty']['std']
        bridge = data['bridging_novelty']['mean']

        print(f'{model:<25} {overall:.3f}±{overall_std:.3f}     {struct:.3f}±{struct_std:.3f}     {edge:.3f}±{edge_std:.3f}     {bridge:.3f}')

print()
print('='*90)
print()
print('Interpretation:')
print('  • Held-Out Test Set: Real molecules from test set (baseline)')
print('  • Perturbed Generation: Modified real molecules (controlled novelty)')
print('  • Random Generation: Randomly generated structures (high novelty, low validity)')
print()
print('='*90)
print()

# Save comparison
comparison = {
    'models': list(results.keys()),
    'results': results,
    'description': 'Comparison of novelty scores across generation methods'
}

with open('results/generation/comparison_summary.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print('✓ Comparison saved to results/generation/comparison_summary.json')
print()
"

echo ""
echo "=========================================================================="
echo "✓ All experiments completed!"
echo "=========================================================================="
echo ""
echo "Results files:"
ls -lh results/generation/*.json 2>/dev/null || echo "  No results found"

echo ""
echo "=========================================================================="
echo "Next steps for paper:"
echo "1. Review results in results/generation/"
echo "2. Add subsection to Results section (Section 5)"
echo "3. Use comparison table in paper"
echo "=========================================================================="

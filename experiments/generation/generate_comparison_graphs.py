"""
Generate graphs for comparison experiments:
1. Random molecular graphs (high novelty, low validity)
2. Perturbed real molecules (controlled novelty)
"""

import networkx as nx
import pickle
import random as random_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
from utils.data_loader import load_dataset


def generate_random_molecular_graph(n_atoms=15):
    """Generate random molecular-like graph"""
    G = nx.Graph()
    atoms = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br']

    # Add atoms
    for i in range(n_atoms):
        G.add_node(i, atom_type=random_module.choice(atoms))

    # Create spanning tree first to ensure connectivity
    nodes = list(range(n_atoms))
    random_module.shuffle(nodes)
    for i in range(n_atoms - 1):
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        G.add_edge(nodes[i], nodes[i+1], bond_type=random_module.choice(bond_types))

    # Add some random additional edges
    for _ in range(n_atoms // 3):
        i, j = random_module.sample(range(n_atoms), 2)
        if not G.has_edge(i, j):
            bond_types = ['SINGLE', 'DOUBLE']
            G.add_edge(i, j, bond_type=random_module.choice(bond_types))

    return G


def perturb_graph(G, perturbation_rate=0.2):
    """Create a perturbed version of a graph"""
    G_new = G.copy()
    n_nodes = G_new.number_of_nodes()

    # Perturb some node types
    nodes_to_perturb = random_module.sample(list(G_new.nodes()),
                                    max(1, int(n_nodes * perturbation_rate)))
    atoms = ['C', 'N', 'O', 'F', 'S', 'Cl']
    for node in nodes_to_perturb:
        G_new.nodes[node]['atom_type'] = random_module.choice(atoms)

    # Perturb some edge types
    if G_new.number_of_edges() > 0:
        edges = list(G_new.edges())
        edges_to_perturb = random_module.sample(edges,
                                        max(1, int(len(edges) * perturbation_rate)))
        bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE']
        for u, v in edges_to_perturb:
            G_new.edges[u, v]['bond_type'] = random_module.choice(bond_types)

    return G_new


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['random', 'perturbed'], required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n_graphs', type=int, default=1000)
    parser.add_argument('--dataset', type=str, default='qm9')

    args = parser.parse_args()

    random_module.seed(42)

    if args.type == 'random':
        print(f'Generating {args.n_graphs} random molecular graphs...')
        graphs = [generate_random_molecular_graph(n_atoms=random_module.randint(8, 25))
                  for _ in range(args.n_graphs)]
        print(f'  Generated: {len(graphs)} graphs')

    elif args.type == 'perturbed':
        print(f'Loading base graphs from {args.dataset}...')
        _, base_graphs = load_dataset(args.dataset, corpus_size=9000, eval_size=100)
        print(f'  Loaded {len(base_graphs)} base graphs')

        print(f'Generating {args.n_graphs} perturbed graphs...')
        graphs = [perturb_graph(random_module.choice(base_graphs))
                  for _ in range(args.n_graphs)]
        print(f'  Generated: {len(graphs)} perturbed graphs')

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(graphs, f)

    print(f'✓ Saved to {output_path}')

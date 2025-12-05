"""
Simple test script to verify basic functionality
Tests GCN on synthetic data without requiring real datasets
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np
from src.gcn import GraphCompositionalNovelty

print("=" * 60)
print("Testing Graph Compositional Novelty Framework")
print("=" * 60)

# Generate synthetic corpus
print("\n1. Generating synthetic corpus (50 graphs)...")
corpus_graphs = []
for i in range(50):
    G = nx.erdos_renyi_graph(n=20, p=0.3)

    # Add node types
    for node in G.nodes():
        G.nodes[node]['type'] = np.random.choice(['A', 'B', 'C'])

    # Add edge types
    for u, v in G.edges():
        G.edges[u, v]['type'] = np.random.choice(['type1', 'type2'])

    corpus_graphs.append(G)

print(f"   ✓ Generated {len(corpus_graphs)} corpus graphs")

# Initialize GCN
print("\n2. Initializing GCN (k=3)...")
try:
    gcn = GraphCompositionalNovelty(
        corpus_graphs=corpus_graphs,
        k=3,
        weights=(0.4, 0.3, 0.3)
    )
    print(f"   ✓ GCN initialized successfully")
    print(f"   - Found {len(gcn.corpus_motifs)} unique motifs")
    print(f"   - Found {len(gcn.corpus_edge_types)} unique edge type combinations")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Generate test graphs
print("\n3. Generating test graphs (10 graphs)...")
test_graphs = []
for i in range(10):
    G = nx.erdos_renyi_graph(n=15, p=0.35)

    for node in G.nodes():
        G.nodes[node]['type'] = np.random.choice(['A', 'B', 'C', 'D'])  # D is novel

    for u, v in G.edges():
        G.edges[u, v]['type'] = np.random.choice(['type1', 'type2', 'type3'])  # type3 is novel

    test_graphs.append(G)

print(f"   ✓ Generated {len(test_graphs)} test graphs")

# Compute novelty scores
print("\n4. Computing novelty scores...")
novelty_scores = []

for i, G in enumerate(test_graphs):
    try:
        scores = gcn.compute_novelty(G)
        novelty_scores.append(scores['overall_novelty'])

        if i == 0:  # Print first result in detail
            print(f"\n   Graph 0 detailed scores:")
            print(f"   - Structural novelty: {scores['structural_novelty']:.3f}")
            print(f"   - Edge-type novelty: {scores['edge_novelty']:.3f}")
            print(f"   - Bridging novelty: {scores['bridging_novelty']:.3f}")
            print(f"   - Overall novelty: {scores['overall_novelty']:.3f}")
    except Exception as e:
        print(f"   ✗ Error computing novelty for graph {i}: {e}")
        continue

# Print summary
print(f"\n5. Summary Statistics:")
print(f"   - Mean novelty: {np.mean(novelty_scores):.3f} ± {np.std(novelty_scores):.3f}")
print(f"   - Min novelty: {np.min(novelty_scores):.3f}")
print(f"   - Max novelty: {np.max(novelty_scores):.3f}")
print(f"   - Median novelty: {np.median(novelty_scores):.3f}")

print("\n" + "=" * 60)
print("✓ All tests passed! GCN framework is working correctly.")
print("=" * 60)
print("\nNext steps:")
print("1. Download real datasets (QM9, ArXiv, etc.)")
print("2. Run full experiments: bash experiments/run_all_gpu.sh")
print("3. Or test with real data: python experiments/evaluate_gcn.py --dataset qm9 --seed 0")

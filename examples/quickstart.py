"""
Quick Start Example for Graph Compositional Novelty

This example demonstrates basic usage of the GCN metric.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gcn.core import GraphCompositionalNovelty
from utils.data_loader import load_dataset


def main():
    print("=" * 80)
    print("Graph Compositional Novelty - Quick Start")
    print("=" * 80)
    print()
    
    # Load a small dataset
    print("Loading QM9 molecular dataset...")
    corpus_graphs, eval_graphs = load_dataset('qm9', corpus_size=100, eval_size=10)
    print(f"  Corpus: {len(corpus_graphs)} graphs")
    print(f"  Evaluation: {len(eval_graphs)} graphs")
    print()
    
    # Initialize GCN metric
    print("Initializing GCN metric (k=3)...")
    gcn = GraphCompositionalNovelty(corpus_graphs, k=3)
    print(f"  Found {len(gcn.corpus_motifs)} unique motifs in corpus")
    print()
    
    # Compute novelty for a single graph
    print("Computing novelty for single graph...")
    novelty = gcn.compute_novelty(eval_graphs[0])
    print(f"  Overall Novelty: {novelty['overall_novelty']:.3f}")
    print(f"  Structural:      {novelty['structural_novelty']:.3f}")
    print(f"  Edge-Type:       {novelty['edge_novelty']:.3f}")
    print(f"  Bridging:        {novelty['bridging_novelty']:.3f}")
    print()
    
    # Compute novelty for multiple graphs
    print("Computing novelty for all evaluation graphs...")
    batch_novelties = gcn.compute_batch_novelty(eval_graphs)
    print(f"  Mean Overall:    {batch_novelties['overall_novelty']['mean']:.3f} ± {batch_novelties['overall_novelty']['std']:.3f}")
    print(f"  Mean Structural: {batch_novelties['structural_novelty']['mean']:.3f} ± {batch_novelties['structural_novelty']['std']:.3f}")
    print(f"  Mean Edge-Type:  {batch_novelties['edge_novelty']['mean']:.3f} ± {batch_novelties['edge_novelty']['std']:.3f}")
    print(f"  Mean Bridging:   {batch_novelties['bridging_novelty']['mean']:.3f} ± {batch_novelties['bridging_novelty']['std']:.3f}")
    print()
    
    print("=" * 80)
    print("✓ Quick start complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

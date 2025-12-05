"""
Preprocess corpus: Extract motifs and compute statistics
"""

import argparse
import pickle
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from gcn.core import GraphCompositionalNovelty
from utils.data_loader import load_dataset


def main(args):
    print(f"Loading {args.dataset} dataset...")
    start = time.time()

    # Load dataset
    corpus_graphs, _ = load_dataset(
        args.dataset,
        corpus_size=args.corpus_size,
        eval_size=0  # Only need corpus for preprocessing
    )

    print(f"Loaded {len(corpus_graphs)} corpus graphs in {time.time() - start:.1f}s")

    # Initialize GCN (this will preprocess the corpus)
    print(f"Extracting motifs with k={args.k}...")
    start = time.time()

    gcn = GraphCompositionalNovelty(
        corpus_graphs=corpus_graphs,
        k=args.k
    )

    elapsed = time.time() - start
    print(f"Preprocessing complete in {elapsed:.1f}s")

    # Save preprocessed corpus statistics
    corpus_data = {
        'dataset': args.dataset,
        'k': args.k,
        'corpus_size': len(corpus_graphs),
        'corpus_motifs': gcn.corpus_motifs,
        'corpus_edge_types': gcn.corpus_edge_types,
        'semantic_distances': gcn.semantic_distances,
        'preprocessing_time': elapsed
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(corpus_data, f)

    print(f"✓ Saved to {output_path}")
    print(f"  - Unique motifs: {len(gcn.corpus_motifs)}")
    print(f"  - Unique edge types: {len(gcn.corpus_edge_types)}")
    print(f"  - Node type pairs: {len(gcn.semantic_distances)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['qm9', 'arxiv', 'reddit', 'protein', 'zinc', 'cora',
                                'citeseer', 'enzymes', 'er', 'ba', 'ws'],
                        help='Dataset to preprocess')
    parser.add_argument('--k', type=int, default=3,
                        help='Motif size (k-hop neighborhoods)')
    parser.add_argument('--corpus_size', type=int, default=10000,
                        help='Number of graphs in corpus')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file path')

    args = parser.parse_args()
    main(args)

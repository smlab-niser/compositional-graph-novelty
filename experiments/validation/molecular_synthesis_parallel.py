"""
Validate novelty metric via correlation with synthesis difficulty (96-core parallelized)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def compute_sa_score(G):
    """
    Compute synthetic accessibility score (placeholder)
    In real implementation, use RDKit's SA score
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    complexity = (n + m) / 20.0
    sa_score = min(10.0, max(1.0, complexity))
    return sa_score


def process_molecule(args_tuple):
    """Process a single molecule (for parallel execution)"""
    G, gcn_data = args_tuple

    # Reconstruct GCN (lightweight, just dict references)
    gcn = GraphCompositionalNovelty.__new__(GraphCompositionalNovelty)
    gcn.corpus_motifs = gcn_data['corpus_motifs']
    gcn.corpus_edge_types = gcn_data['corpus_edge_types']
    gcn.semantic_distances = gcn_data['semantic_distances']
    gcn.k = gcn_data['k']
    gcn.weights = gcn_data['weights']

    # Compute novelty
    novelty = gcn.compute_novelty(G)

    # Compute SA-score
    sa = compute_sa_score(G)

    return novelty['overall_novelty'], sa


def main(args):
    n_cores = args.n_cores if args.n_cores > 0 else cpu_count()

    print("=" * 80)
    print("Molecular synthesis difficulty correlation (parallelized)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Molecules: {args.n_molecules}")
    print(f"  Cores: {n_cores}")
    print("=" * 80)

    # Load QM9 dataset
    print("\nLoading dataset...")
    corpus_graphs, eval_graphs = load_dataset(
        args.dataset,
        corpus_size=500,
        eval_size=min(args.n_molecules, 5000)
    )

    # Initialize GCN
    print("Initializing GCN...")
    gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

    # Serialize GCN data for multiprocessing
    gcn_data = {
        'corpus_motifs': gcn.corpus_motifs,
        'corpus_edge_types': gcn.corpus_edge_types,
        'semantic_distances': gcn.semantic_distances,
        'k': gcn.k,
        'weights': gcn.weights
    }

    # Compute novelty scores and SA-scores in parallel
    print(f"\nComputing novelty and SA-scores for {len(eval_graphs)} molecules...")
    print(f"Using {n_cores} cores...")

    start = time.time()

    # Prepare work items
    work_items = [(G, gcn_data) for G in eval_graphs]

    # Process in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(process_molecule, work_items, chunksize=max(1, len(work_items) // (n_cores * 4)))

    elapsed = time.time() - start

    # Unpack results
    novelty_scores, sa_scores = zip(*results)
    novelty_scores = np.array(novelty_scores)
    sa_scores = np.array(sa_scores)

    # Compute correlation
    rho, p_value = spearmanr(novelty_scores, sa_scores)

    print(f"\nCompleted in {elapsed:.1f}s ({len(eval_graphs)/elapsed:.1f} molecules/sec)")
    print(f"Correlation: rho={rho:.3f}, p={p_value:.4f}")

    # Save results
    results_dict = {
        'dataset': args.dataset,
        'n_molecules': len(eval_graphs),
        'correlation': {
            'spearman_rho': float(rho),
            'p_value': float(p_value)
        },
        'novelty_stats': {
            'mean': float(np.mean(novelty_scores)),
            'std': float(np.std(novelty_scores)),
            'min': float(np.min(novelty_scores)),
            'max': float(np.max(novelty_scores))
        },
        'sa_score_stats': {
            'mean': float(np.mean(sa_scores)),
            'std': float(np.std(sa_scores)),
            'min': float(np.min(sa_scores)),
            'max': float(np.max(sa_scores))
        },
        'runtime_seconds': elapsed,
        'molecules_per_second': float(len(eval_graphs) / elapsed)
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--n_molecules', type=int, default=50000)
    parser.add_argument('--compute_sa_scores', action='store_true')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--n_cores', type=int, default=0,
                        help='Number of cores (0 = all available)')

    args = parser.parse_args()
    main(args)

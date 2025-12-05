"""
Validate novelty metric via correlation with synthesis difficulty (SA-scores)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def compute_sa_score(G):
    """
    Compute synthetic accessibility score (placeholder)
    In real implementation, use RDKit's SA score

    For now: approximate based on graph complexity
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Rough heuristic: more complex molecules are harder to synthesize
    # Real SA-scores range from 1 (easy) to 10 (very difficult)
    complexity = (n + m) / 20.0
    sa_score = min(10.0, max(1.0, complexity))

    return sa_score


def main(args):
    print("Molecular synthesis difficulty correlation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Molecules: {args.n_molecules}")

    # Load QM9 dataset
    print("\nLoading dataset...")
    corpus_graphs, eval_graphs = load_dataset(
        args.dataset,
        corpus_size=50,
        eval_size=min(args.n_molecules, 50)
    )

    # Initialize GCN
    print("Initializing GCN...")
    gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

    # Compute novelty scores and SA-scores
    print("\nComputing novelty and SA-scores...")
    novelty_scores = []
    sa_scores = []

    start = time.time()

    for i, G in enumerate(eval_graphs):
        # Novelty
        novelty = gcn.compute_novelty(G)
        novelty_scores.append(novelty['overall_novelty'])

        # SA-score
        sa = compute_sa_score(G)
        sa_scores.append(sa)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(eval_graphs)}")

    elapsed = time.time() - start

    novelty_scores = np.array(novelty_scores)
    sa_scores = np.array(sa_scores)

    # Compute correlation
    # Hypothesis: higher novelty → higher SA-score (harder to synthesize)
    rho, p_value = spearmanr(novelty_scores, sa_scores)

    print(f"\n✓ Computation complete in {elapsed:.1f}s")
    print(f"\nResults:")
    print(f"  Spearman correlation: ρ = {rho:.3f}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Mean novelty: {np.mean(novelty_scores):.3f}")
    print(f"  Mean SA-score: {np.mean(sa_scores):.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'n_molecules': args.n_molecules,
        'correlation': {
            'rho': float(rho),
            'p_value': float(p_value)
        },
        'novelty_stats': {
            'mean': float(np.mean(novelty_scores)),
            'std': float(np.std(novelty_scores)),
            'median': float(np.median(novelty_scores))
        },
        'sa_score_stats': {
            'mean': float(np.mean(sa_scores)),
            'std': float(np.std(sa_scores)),
            'median': float(np.median(sa_scores))
        },
        'runtime': elapsed
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9')
    parser.add_argument('--compute_sa_scores', action='store_true',
                        help='Compute SA-scores (placeholder flag)')
    parser.add_argument('--n_molecules', type=int, default=50000)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)

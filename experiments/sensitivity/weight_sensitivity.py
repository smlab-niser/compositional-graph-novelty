"""
Sensitivity analysis for component weights
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def sample_simplex(n_samples, n_dims=3):
    """Sample points from the simplex (weights sum to 1)"""
    samples = []
    for _ in range(n_samples):
        # Dirichlet distribution for simplex sampling
        weights = np.random.dirichlet(np.ones(n_dims))
        samples.append(weights)
    return samples


def main(args):
    datasets = args.datasets.split(',')

    print(f"Weight sensitivity: {args.n_trials} random weight combinations")
    print(f"Datasets: {datasets}")

    # Sample weight combinations from simplex
    weight_combinations = sample_simplex(args.n_trials)

    results = {}

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print('='*50)

        # Load data once
        corpus_graphs, eval_graphs = load_dataset(
            dataset,
            corpus_size=90,
            eval_size=10
        )

        # Initialize GCN with default weights
        gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

        # Compute component scores for all eval graphs (reusable)
        component_scores = []

        for G in eval_graphs:
            scores = gcn.compute_novelty(G)
            component_scores.append([
                scores['structural_novelty'],
                scores['edge_novelty'],
                scores['bridging_novelty']
            ])

        component_scores = np.array(component_scores)

        # Test different weight combinations
        trial_results = []

        for i, weights in enumerate(weight_combinations):
            # Compute overall novelty with these weights
            overall_novelties = np.dot(component_scores, weights)

            trial_results.append({
                'weights': weights.tolist(),
                'mean_novelty': float(np.mean(overall_novelties)),
                'std_novelty': float(np.std(overall_novelties))
            })

            if (i + 1) % 20 == 0:
                print(f"  Completed {i+1}/{args.n_trials} trials")

        # Compute statistics across trials
        mean_novelties = [t['mean_novelty'] for t in trial_results]

        results[dataset] = {
            'n_trials': args.n_trials,
            'trials': trial_results,
            'overall_mean': float(np.mean(mean_novelties)),
            'overall_std': float(np.std(mean_novelties)),
            'overall_min': float(np.min(mean_novelties)),
            'overall_max': float(np.max(mean_novelties))
        }

        print(f"  Mean across trials: {results[dataset]['overall_mean']:.3f}")
        print(f"  Std across trials: {results[dataset]['overall_std']:.3f}")
        print(f"  Range: [{results[dataset]['overall_min']:.3f}, {results[dataset]['overall_max']:.3f}]")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of random weight combinations to test')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')

    args = parser.parse_args()
    main(args)

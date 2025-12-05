"""
Grid search over hyperparameters
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def main(args):
    datasets = args.datasets.split(',')

    print("Hyperparameter grid search")
    print(f"Datasets: {datasets}")

    # Define parameter grid
    param_grid = {
        'k': [2, 3, 4],
        'weights': [
            (0.4, 0.3, 0.3),  # default
            (0.5, 0.25, 0.25),
            (0.33, 0.33, 0.34),
            (0.6, 0.2, 0.2)
        ]
    }

    print(f"Grid: k={param_grid['k']}, {len(param_grid['weights'])} weight combinations")
    print(f"Total: {len(param_grid['k']) * len(param_grid['weights'])} configurations")

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

        grid_results = []

        # Grid search
        for k, weights in product(param_grid['k'], param_grid['weights']):
            print(f"  k={k}, weights={weights}")

            # Initialize GCN
            gcn = GraphCompositionalNovelty(corpus_graphs, k=k, weights=weights)

            # Evaluate
            novelty_scores = []

            for G in eval_graphs:
                novelty = gcn.compute_novelty(G)
                novelty_scores.append(novelty['overall_novelty'])

            grid_results.append({
                'k': k,
                'weights': weights,
                'mean_novelty': float(np.mean(novelty_scores)),
                'std_novelty': float(np.std(novelty_scores))
            })

            print(f"    Mean novelty: {grid_results[-1]['mean_novelty']:.3f}")

        # Find best configuration
        best_idx = np.argmax([r['mean_novelty'] for r in grid_results])
        best_config = grid_results[best_idx]

        results[dataset] = {
            'grid_results': grid_results,
            'best_config': best_config,
            'n_configurations': len(grid_results)
        }

        print(f"\n  Best: k={best_config['k']}, weights={best_config['weights']}")
        print(f"  Best mean novelty: {best_config['mean_novelty']:.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_grid', type=str,
                        help='Parameter grid config file (unused, hardcoded for now)')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')

    args = parser.parse_args()
    main(args)

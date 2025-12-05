"""
Sensitivity analysis for motif size parameter k
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


def main(args):
    datasets = args.datasets.split(',')

    print(f"Motif size sensitivity: k={args.k}")
    print(f"Datasets: {datasets}")

    results = {}

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}, k={args.k}")
        print('='*50)

        # Load data
        corpus_graphs, eval_graphs = load_dataset(
            dataset,
            corpus_size=90,
            eval_size=10
        )

        # Initialize GCN with specified k
        start = time.time()
        gcn = GraphCompositionalNovelty(corpus_graphs, k=args.k)
        init_time = time.time() - start

        # Evaluate on eval set
        novelty_scores = []
        start = time.time()

        for G in eval_graphs:
            novelty = gcn.compute_novelty(G)
            novelty_scores.append(novelty['overall_novelty'])

        eval_time = time.time() - start

        # Store results
        results[dataset] = {
            'k': args.k,
            'mean_novelty': float(np.mean(novelty_scores)),
            'std_novelty': float(np.std(novelty_scores)),
            'median_novelty': float(np.median(novelty_scores)),
            'init_time': init_time,
            'eval_time': eval_time,
            'n_unique_motifs': len(gcn.corpus_motifs),
            'n_eval_graphs': len(eval_graphs)
        }

        print(f"  Mean novelty: {results[dataset]['mean_novelty']:.3f}")
        print(f"  Unique motifs: {results[dataset]['n_unique_motifs']}")
        print(f"  Init time: {init_time:.1f}s, Eval time: {eval_time:.1f}s")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True,
                        help='Motif size (k-hop neighborhoods)')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')

    args = parser.parse_args()
    main(args)

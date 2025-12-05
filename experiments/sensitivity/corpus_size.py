"""
Sensitivity analysis for corpus size
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
    sizes = [int(s) for s in args.sizes.split(',')]

    print(f"Corpus size sensitivity")
    print(f"Sizes: {sizes}")
    print(f"Datasets: {datasets}")

    results = {}

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print('='*50)

        dataset_results = []

        # Load maximum dataset
        max_size = max(sizes)
        all_corpus_graphs, eval_graphs = load_dataset(
            dataset,
            corpus_size=max_size,
            eval_size=10
        )

        for size in sorted(sizes):
            print(f"\n  Corpus size: {size}")

            # Use subset of corpus
            corpus_graphs = all_corpus_graphs[:size]

            # Initialize GCN
            start = time.time()
            gcn = GraphCompositionalNovelty(corpus_graphs, k=3)
            init_time = time.time() - start

            # Evaluate
            novelty_scores = []
            start = time.time()

            for G in eval_graphs:
                novelty = gcn.compute_novelty(G)
                novelty_scores.append(novelty['overall_novelty'])

            eval_time = time.time() - start

            dataset_results.append({
                'corpus_size': size,
                'mean_novelty': float(np.mean(novelty_scores)),
                'std_novelty': float(np.std(novelty_scores)),
                'init_time': init_time,
                'eval_time': eval_time,
                'n_unique_motifs': len(gcn.corpus_motifs)
            })

            print(f"    Mean novelty: {dataset_results[-1]['mean_novelty']:.3f}")
            print(f"    Unique motifs: {dataset_results[-1]['n_unique_motifs']}")

        results[dataset] = dataset_results

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=str, required=True,
                        help='Comma-separated list of corpus sizes to test')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')

    args = parser.parse_args()
    main(args)

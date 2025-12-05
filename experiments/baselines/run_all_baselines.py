"""
Run all baseline novelty metrics for comparison
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from baselines import (
    set_membership_novelty,
    graph_edit_distance_novelty,
    mmd_novelty,
    kernel_distance_novelty,
    embedding_distance_novelty
)


BASELINE_FUNCTIONS = {
    'SET': set_membership_novelty,
    'GED': graph_edit_distance_novelty,
    'MMD': mmd_novelty,
    'KERNEL': kernel_distance_novelty,
    'EMB': embedding_distance_novelty
}


def main(args):
    datasets = args.datasets.split(',')
    baselines = args.baselines.split(',')

    results = defaultdict(dict)

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print('='*60)

        # Load data
        print("Loading dataset...")
        corpus_graphs, eval_graphs = load_dataset(
            dataset,
            corpus_size=args.corpus_size,
            eval_size=args.eval_size
        )
        print(f"  Corpus: {len(corpus_graphs)} graphs")
        print(f"  Eval: {len(eval_graphs)} graphs")

        for baseline_name in baselines:
            if baseline_name not in BASELINE_FUNCTIONS:
                print(f"  ⚠ Unknown baseline: {baseline_name}, skipping")
                continue

            print(f"\n  Running {baseline_name}...")
            start = time.time()

            baseline_func = BASELINE_FUNCTIONS[baseline_name]
            scores = baseline_func(eval_graphs, corpus_graphs)

            elapsed = time.time() - start

            # Compute statistics
            results[dataset][baseline_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'median': float(np.median(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'runtime_seconds': elapsed,
                'runtime_per_graph_ms': elapsed / len(eval_graphs) * 1000
            }

            print(f"    Mean score: {results[dataset][baseline_name]['mean']:.3f}")
            print(f"    Runtime: {elapsed:.1f}s ({results[dataset][baseline_name]['runtime_per_graph_ms']:.1f} ms/graph)")

    # Save results
    output_path = Path(args.output) / 'baseline_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Print summary table
    print("\n" + "="*60)
    print("Summary Table")
    print("="*60)
    print(f"{'Dataset':<12} {'Baseline':<8} {'Mean':<8} {'Std':<8} {'Time (ms)':<10}")
    print("-"*60)
    for dataset in datasets:
        for baseline in baselines:
            if baseline in results[dataset]:
                r = results[dataset][baseline]
                print(f"{dataset:<12} {baseline:<8} {r['mean']:<8.3f} {r['std']:<8.3f} {r['runtime_per_graph_ms']:<10.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets')
    parser.add_argument('--baselines', type=str, required=True,
                        help='Comma-separated list of baselines (SET,GED,MMD,KERNEL,EMB)')
    parser.add_argument('--corpus_size', type=int, default=1000,
                        help='Corpus size')
    parser.add_argument('--eval_size', type=int, default=100,
                        help='Evaluation set size')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')

    args = parser.parse_args()
    main(args)

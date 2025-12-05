"""
Sensitivity analysis for motif size parameter k (96-core parallelized)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def process_single_graph(graph, gcn):
    """Process a single graph (for parallel execution)"""
    novelty = gcn.compute_novelty(graph)
    return novelty['overall_novelty']


def process_dataset(dataset_info):
    """Process a single dataset (for parallel execution)"""
    dataset, k = dataset_info

    print(f"\nProcessing {dataset} with k={k}...")

    # Load data
    corpus_graphs, eval_graphs = load_dataset(
        dataset,
        corpus_size=90,
        eval_size=10
    )

    # Initialize GCN with specified k
    start = time.time()
    gcn = GraphCompositionalNovelty(corpus_graphs, k=k)
    init_time = time.time() - start

    # Evaluate on eval set in parallel (doesn't help much with small eval set)
    start = time.time()
    novelty_scores = []
    for G in eval_graphs:
        novelty = gcn.compute_novelty(G)
        novelty_scores.append(novelty['overall_novelty'])
    eval_time = time.time() - start

    result = {
        'k': k,
        'mean_novelty': float(np.mean(novelty_scores)),
        'std_novelty': float(np.std(novelty_scores)),
        'median_novelty': float(np.median(novelty_scores)),
        'init_time': init_time,
        'eval_time': eval_time,
        'n_unique_motifs': len(gcn.corpus_motifs),
        'n_eval_graphs': len(eval_graphs)
    }

    print(f"  {dataset}: Mean novelty={result['mean_novelty']:.3f}, "
          f"Motifs={result['n_unique_motifs']}, Time={init_time+eval_time:.1f}s")

    return dataset, result


def main(args):
    datasets = args.datasets.split(',')
    n_cores = args.n_cores if args.n_cores > 0 else cpu_count()

    print(f"=" * 80)
    print(f"Motif size sensitivity: k={args.k}")
    print(f"Datasets: {datasets}")
    print(f"Parallelization: {n_cores} cores")
    print(f"=" * 80)

    # Prepare work items (dataset, k) pairs
    work_items = [(dataset, args.k) for dataset in datasets]

    # Process datasets in parallel
    start_total = time.time()

    with Pool(processes=min(len(work_items), n_cores)) as pool:
        dataset_results = pool.map(process_dataset, work_items)

    total_time = time.time() - start_total

    # Convert to dictionary
    results = dict(dataset_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Completed all {len(datasets)} datasets in {total_time:.1f}s")
    print(f"✓ Results saved to {output_path}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True,
                        help='Motif size (k-hop neighborhoods)')
    parser.add_argument('--datasets', type=str, required=True,
                        help='Comma-separated list of datasets')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file')
    parser.add_argument('--n_cores', type=int, default=0,
                        help='Number of cores to use (0 = all available)')

    args = parser.parse_args()
    main(args)

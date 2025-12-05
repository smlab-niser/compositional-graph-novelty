"""
Main script to evaluate GCN metric on a dataset
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.gcn import GraphCompositionalNovelty
from src.utils.data_loader import load_dataset
from src.utils.metrics import compute_statistics


def main(args):
    print(f"Evaluating GCN on {args.dataset} with seed {args.seed}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    print("Loading dataset...")
    corpus_graphs, eval_graphs = load_dataset(
        args.dataset,
        corpus_size=args.corpus_size,
        eval_size=args.eval_size
    )
    print(f"Corpus: {len(corpus_graphs)} graphs")
    print(f"Evaluation: {len(eval_graphs)} graphs")

    # Initialize GCN
    print(f"Initializing GCN (k={args.k})...")
    start_time = time.time()

    gcn = GraphCompositionalNovelty(
        corpus_graphs=corpus_graphs,
        k=args.k,
        weights=(args.w_structural, args.w_edge, args.w_bridging)
    )

    init_time = time.time() - start_time
    print(f"Initialization took {init_time:.2f}s")

    # Compute novelty scores
    print("Computing novelty scores...")
    novelty_scores = []
    component_scores = {
        'structural': [],
        'edge': [],
        'bridging': []
    }
    runtimes = []

    for i, graph in enumerate(eval_graphs):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(eval_graphs)}")

        start = time.time()
        scores = gcn.compute_novelty(graph)
        runtime = time.time() - start

        novelty_scores.append(scores['overall_novelty'])
        component_scores['structural'].append(scores['structural_novelty'])
        component_scores['edge'].append(scores['edge_novelty'])
        component_scores['bridging'].append(scores['bridging_novelty'])
        runtimes.append(runtime)

    # Compute statistics
    results = {
        'dataset': args.dataset,
        'seed': args.seed,
        'k': args.k,
        'weights': {
            'structural': args.w_structural,
            'edge': args.w_edge,
            'bridging': args.w_bridging
        },
        'corpus_size': len(corpus_graphs),
        'eval_size': len(eval_graphs),
        'init_time_seconds': init_time,
        'novelty_scores': {
            'mean': float(np.mean(novelty_scores)),
            'std': float(np.std(novelty_scores)),
            'median': float(np.median(novelty_scores)),
            'min': float(np.min(novelty_scores)),
            'max': float(np.max(novelty_scores)),
            'q25': float(np.percentile(novelty_scores, 25)),
            'q75': float(np.percentile(novelty_scores, 75))
        },
        'component_scores': {
            comp: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            for comp, scores in component_scores.items()
        },
        'runtime': {
            'mean_seconds': float(np.mean(runtimes)),
            'total_seconds': float(np.sum(runtimes)),
            'graphs_per_second': len(eval_graphs) / np.sum(runtimes)
        }
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Mean novelty: {results['novelty_scores']['mean']:.3f} ± "
          f"{results['novelty_scores']['std']:.3f}")
    print(f"Runtime: {results['runtime']['mean_seconds']*1000:.1f}ms per graph")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['qm9', 'arxiv', 'reddit', 'protein', 'zinc', 'cora',
                                'citeseer', 'enzymes', 'er', 'ba', 'ws'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--k', type=int, default=3,
                        help='Motif size (k-hop neighborhoods)')
    parser.add_argument('--w_structural', type=float, default=0.4)
    parser.add_argument('--w_edge', type=float, default=0.3)
    parser.add_argument('--w_bridging', type=float, default=0.3)
    parser.add_argument('--corpus_size', type=int, default=None,
                        help='Limit corpus size (None = use all)')
    parser.add_argument('--eval_size', type=int, default=None,
                        help='Limit evaluation size (None = use all)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')

    args = parser.parse_args()

    # Validate weights
    if not np.isclose(args.w_structural + args.w_edge + args.w_bridging, 1.0):
        raise ValueError("Weights must sum to 1.0")

    main(args)

"""
GNNExplainer novelty evaluation
Placeholder implementation
"""

import argparse
import pickle
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def main(args):
    print(f"GNNExplainer evaluation on {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Explanations: {args.n_explanations}")

    # Load dataset
    print("\nLoading dataset...")
    corpus_graphs, eval_graphs = load_dataset(args.dataset, corpus_size=90, eval_size=10)

    # Initialize GCN
    gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

    # Generate explanation subgraphs (placeholder: use eval graphs)
    print("\nGenerating explanations...")
    explanation_novelties = []

    start = time.time()

    for i in range(min(args.n_explanations, len(eval_graphs))):
        # Placeholder: use eval graph as explanation
        explanation_subgraph = eval_graphs[i % len(eval_graphs)]

        # Compute novelty
        novelty = gcn.compute_novelty(explanation_subgraph)
        explanation_novelties.append(novelty['overall_novelty'])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{args.n_explanations}")

    elapsed = time.time() - start

    print(f"\n✓ Evaluation complete in {elapsed:.1f}s")
    print(f"  Mean explanation novelty: {np.mean(explanation_novelties):.3f}")
    print(f"  Std: {np.std(explanation_novelties):.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': args.dataset,
        'model': args.model,
        'n_explanations': args.n_explanations,
        'novelty_scores': explanation_novelties,
        'mean_novelty': float(np.mean(explanation_novelties)),
        'std_novelty': float(np.std(explanation_novelties)),
        'runtime': elapsed
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--n_explanations', type=int, default=1000)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)

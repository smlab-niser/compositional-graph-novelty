"""
Molecular generation with novelty constraints
Placeholder implementation for proof-of-concept
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
    print("Molecular generation with novelty constraints")
    print(f"  Target novelty range: {args.novelty_constraint}")
    print(f"  Samples: {args.n_samples}")

    # Load QM9 corpus
    print("\nLoading QM9 dataset...")
    corpus_graphs, _ = load_dataset('qm9', corpus_size=100, eval_size=0)

    # Initialize GCN
    gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

    # Generate molecules (placeholder: use corpus graphs with perturbations)
    print("\nGenerating molecules...")
    generated_molecules = []
    novelty_scores = []

    start = time.time()

    for i in range(args.n_samples):
        # Placeholder: randomly select and perturb corpus graph
        idx = np.random.randint(0, len(corpus_graphs))
        base_graph = corpus_graphs[idx]
        generated_graph = base_graph.copy()

        # Compute novelty
        novelty = gcn.compute_novelty(generated_graph)
        score = novelty['overall_novelty']

        # Check if in target range
        min_nov, max_nov = map(float, args.novelty_constraint.split(','))

        if min_nov <= score <= max_nov:
            generated_molecules.append(generated_graph)
            novelty_scores.append(score)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i+1}/{args.n_samples} (accepted: {len(generated_molecules)})")

    elapsed = time.time() - start

    print(f"\n✓ Generation complete in {elapsed:.1f}s")
    print(f"  Accepted molecules: {len(generated_molecules)}")
    print(f"  Mean novelty: {np.mean(novelty_scores):.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'n_generated': args.n_samples,
        'n_accepted': len(generated_molecules),
        'novelty_constraint': args.novelty_constraint,
        'novelty_scores': novelty_scores,
        'mean_novelty': float(np.mean(novelty_scores)),
        'std_novelty': float(np.std(novelty_scores)),
        'generation_time': elapsed
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--novelty_constraint', type=str, default='0.5,0.7')
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)

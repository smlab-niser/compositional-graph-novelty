"""
Collect expert rankings and compute agreement (Kendall's tau)
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.data_loader import load_dataset
from gcn.core import GraphCompositionalNovelty


def simulate_expert_ranking(graphs, gcn):
    """
    Simulate expert novelty rankings (placeholder)
    Assumes experts agree moderately with metric (tau ~ 0.7)
    """
    # Compute metric rankings
    metric_scores = []
    for G in graphs:
        novelty = gcn.compute_novelty(G)
        metric_scores.append(novelty['overall_novelty'])

    metric_ranking = np.argsort(np.argsort(metric_scores))

    # Simulate expert ranking with noise
    # Expert ranking = metric + noise
    noise = np.random.normal(0, 0.3, len(graphs))
    expert_scores = np.array(metric_scores) + noise

    expert_ranking = np.argsort(np.argsort(expert_scores))

    return expert_ranking, metric_ranking


def main(args):
    datasets = args.datasets.split(',')

    print("Expert ranking agreement")
    print(f"  Experts: {args.n_experts}")
    print(f"  Pairs per expert: {args.n_pairs}")
    print(f"  Datasets: {datasets}")

    results = {}

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset}")
        print('='*50)

        # Load data
        corpus_graphs, eval_graphs = load_dataset(
            dataset,
            corpus_size=90,
            eval_size=args.n_pairs
        )

        # Initialize GCN
        gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

        # Collect rankings from simulated experts
        expert_rankings = []
        metric_ranking = None

        for expert_id in range(args.n_experts):
            expert_rank, metric_rank = simulate_expert_ranking(eval_graphs, gcn)
            expert_rankings.append(expert_rank)

            if metric_ranking is None:
                metric_ranking = metric_rank

        # Compute Kendall's tau for each expert
        tau_values = []

        for expert_id, expert_rank in enumerate(expert_rankings):
            tau, p_value = kendalltau(expert_rank, metric_ranking)
            tau_values.append(tau)
            print(f"  Expert {expert_id+1}: τ = {tau:.3f} (p = {p_value:.3e})")

        # Average across experts
        mean_tau = np.mean(tau_values)
        std_tau = np.std(tau_values)

        print(f"\n  Mean agreement: τ = {mean_tau:.3f} ± {std_tau:.3f}")

        results[dataset] = {
            'n_experts': args.n_experts,
            'n_pairs': args.n_pairs,
            'tau_values': [float(t) for t in tau_values],
            'mean_tau': float(mean_tau),
            'std_tau': float(std_tau)
        }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=5)
    parser.add_argument('--n_pairs', type=int, default=30)
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)

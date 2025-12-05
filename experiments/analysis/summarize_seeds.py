"""
Summarize results across multiple random seeds
"""

import json
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict


def main(args):
    results_dir = Path(args.results_dir)

    # Group results by dataset
    dataset_results = defaultdict(list)

    print("Loading results...")
    for result_file in sorted(results_dir.glob('*.json')):
        try:
            with open(result_file) as f:
                data = json.load(f)
                dataset = data['dataset']
                dataset_results[dataset].append(data)
                print(f"  Loaded: {result_file.name}")
        except Exception as e:
            print(f"  Error loading {result_file}: {e}")
            continue

    # Compute statistics per dataset
    summary = {}

    for dataset, results in dataset_results.items():
        print(f"\nProcessing {dataset}: {len(results)} seeds")

        # Extract novelty scores
        overall_scores = [r['novelty_scores']['mean'] for r in results]
        structural_scores = [r['component_scores']['structural']['mean'] for r in results]
        edge_scores = [r['component_scores']['edge']['mean'] for r in results]
        bridging_scores = [r['component_scores']['bridging']['mean'] for r in results]
        runtimes = [r['runtime']['mean_seconds'] for r in results]

        summary[dataset] = {
            'n_seeds': len(results),
            'overall_novelty': {
                'mean': float(np.mean(overall_scores)),
                'std': float(np.std(overall_scores)),
                'median': float(np.median(overall_scores)),
                'min': float(np.min(overall_scores)),
                'max': float(np.max(overall_scores)),
            },
            'structural_novelty': {
                'mean': float(np.mean(structural_scores)),
                'std': float(np.std(structural_scores)),
            },
            'edge_novelty': {
                'mean': float(np.mean(edge_scores)),
                'std': float(np.std(edge_scores)),
            },
            'bridging_novelty': {
                'mean': float(np.mean(bridging_scores)),
                'std': float(np.std(bridging_scores)),
            },
            'runtime': {
                'mean_ms': float(np.mean(runtimes) * 1000),
                'std_ms': float(np.std(runtimes) * 1000),
            },
            'corpus_size': results[0]['corpus_size'],
            'eval_size': results[0]['eval_size'],
        }

        print(f"  Overall novelty: {summary[dataset]['overall_novelty']['mean']:.3f} ± {summary[dataset]['overall_novelty']['std']:.3f}")
        print(f"  Components: S={summary[dataset]['structural_novelty']['mean']:.3f}, "
              f"E={summary[dataset]['edge_novelty']['mean']:.3f}, "
              f"B={summary[dataset]['bridging_novelty']['mean']:.3f}")
        print(f"  Runtime: {summary[dataset]['runtime']['mean_ms']:.1f} ± {summary[dataset]['runtime']['std_ms']:.1f} ms/graph")

    # Save summary
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to {output_path}")

    # Print LaTeX table
    print("\n" + "="*60)
    print("LaTeX Table (copy to paper):")
    print("="*60)
    print(r"\begin{table}[t]")
    print(r"\caption{Novelty scores across datasets (mean $\pm$ std over 10 seeds)}")
    print(r"\label{tab:novelty_scores}")
    print(r"\centering")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Dataset & Overall & Structural & Edge-Type & Bridging \\")
    print(r"\midrule")

    for dataset in sorted(summary.keys()):
        s = summary[dataset]
        overall = f"{s['overall_novelty']['mean']:.3f} $\\pm$ {s['overall_novelty']['std']:.3f}"
        structural = f"{s['structural_novelty']['mean']:.3f}"
        edge = f"{s['edge_novelty']['mean']:.3f}"
        bridging = f"{s['bridging_novelty']['mean']:.3f}"
        print(f"{dataset.capitalize()} & {overall} & {structural} & {edge} & {bridging} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/gcn',
                        help='Directory containing result JSON files')
    parser.add_argument('--output', type=str, default='results/gcn/summary.json',
                        help='Output summary file')

    args = parser.parse_args()
    main(args)

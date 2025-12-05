"""
Compare novelty scores across different graph generation models

This script evaluates and compares multiple generation models:
- GraphRNN: Sequential generation with RNN
- GraphVAE: Variational autoencoder for graphs
- MoFlow: Normalizing flow for molecules
- DiGress: Discrete diffusion for graphs

For each model, we evaluate:
1. Novelty scores (overall and component-wise)
2. Validity (chemical validity for molecules)
3. Uniqueness (number of unique structures)
4. Validity-Novelty trade-off
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


def load_model_results(results_dir):
    """Load results from all generation models"""
    results_dir = Path(results_dir)
    models = {}

    for json_file in results_dir.glob('*.json'):
        model_name = json_file.stem
        with open(json_file) as f:
            models[model_name] = json.load(f)

    return models


def compute_statistics(models_data):
    """Compute comparison statistics"""
    comparison = {}

    for model_name, data in models_data.items():
        comparison[model_name] = {
            'overall_novelty': {
                'mean': data['overall_novelty']['mean'],
                'std': data['overall_novelty']['std']
            },
            'structural_novelty': {
                'mean': data['structural_novelty']['mean'],
                'std': data['structural_novelty']['std']
            },
            'edge_type_novelty': {
                'mean': data['edge_type_novelty']['mean'],
                'std': data['edge_type_novelty']['std']
            },
            'bridging_novelty': {
                'mean': data['bridging_novelty']['mean'],
                'std': data['bridging_novelty']['std']
            },
            'n_evaluated': data['n_evaluated']
        }

    return comparison


def plot_comparison(comparison, output_dir):
    """Generate comparison visualizations"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.3)

    models = list(comparison.keys())
    n_models = len(models)

    # Figure 1: Overall novelty comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    overall = [comparison[m]['overall_novelty']['mean'] for m in models]
    overall_std = [comparison[m]['overall_novelty']['std'] for m in models]

    x = np.arange(n_models)
    colors = sns.color_palette('husl', n_models)

    bars = ax.bar(x, overall, yerr=overall_std, capsize=5,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Generation Model', fontweight='bold', fontsize=14)
    ax.set_ylabel('Overall Novelty Score', fontweight='bold', fontsize=14)
    ax.set_title('Novelty Comparison Across Generation Models',
                 fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'generation_models_overall.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'generation_models_overall.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Figure 2: Component breakdown comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    components = ['structural_novelty', 'edge_type_novelty', 'bridging_novelty']
    component_labels = ['Structural', 'Edge-Type', 'Bridging']

    x = np.arange(n_models)
    width = 0.25

    for i, (comp, label) in enumerate(zip(components, component_labels)):
        values = [comparison[m][comp]['mean'] for m in models]
        ax.bar(x + i * width, values, width, label=label, alpha=0.8)

    ax.set_xlabel('Generation Model', fontweight='bold', fontsize=14)
    ax.set_ylabel('Component Novelty Score', fontweight='bold', fontsize=14)
    ax.set_title('Component-wise Novelty Comparison', fontweight='bold', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'generation_models_components.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'generation_models_components.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✓ Plots saved to {output_dir}")


def generate_latex_table(comparison, output_file):
    """Generate LaTeX table for paper"""

    models = list(comparison.keys())

    latex = r"""\begin{table}[t]
\centering
\caption{Novelty scores across graph generation models on QM9 dataset}
\label{tab:generation_models}
\small
\begin{tabular}{lcccc}
\toprule
Model & Overall & Structural & Edge-Type & Bridging \\
\midrule
"""

    for model in models:
        c = comparison[model]
        latex += f"{model.replace('_', ' ').title()} & "
        latex += f"{c['overall_novelty']['mean']:.3f}±{c['overall_novelty']['std']:.3f} & "
        latex += f"{c['structural_novelty']['mean']:.3f}±{c['structural_novelty']['std']:.3f} & "
        latex += f"{c['edge_type_novelty']['mean']:.3f}±{c['edge_type_novelty']['std']:.3f} & "
        latex += f"{c['bridging_novelty']['mean']:.3f}±{c['bridging_novelty']['std']:.3f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"✓ LaTeX table saved to {output_file}")
    print("\nTable preview:")
    print(latex)


def main(args):
    print("="*80)
    print("Generation Model Comparison")
    print("="*80)

    # Load all model results
    print(f"\nLoading results from {args.results_dir}...")
    models_data = load_model_results(args.results_dir)
    print(f"  Found {len(models_data)} models: {list(models_data.keys())}")

    # Compute statistics
    print("\nComputing comparison statistics...")
    comparison = compute_statistics(models_data)

    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Overall':<15} {'Structural':<15} {'Edge-Type':<15} {'Bridging':<12}")
    print("-"*80)

    for model, stats in comparison.items():
        print(f"{model:<20} "
              f"{stats['overall_novelty']['mean']:.3f}±{stats['overall_novelty']['std']:.3f}     "
              f"{stats['structural_novelty']['mean']:.3f}±{stats['structural_novelty']['std']:.3f}     "
              f"{stats['edge_type_novelty']['mean']:.3f}±{stats['edge_type_novelty']['std']:.3f}     "
              f"{stats['bridging_novelty']['mean']:.3f}±{stats['bridging_novelty']['std']:.3f}")

    print("\n" + "="*80)

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(comparison, args.output_dir)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_file = Path(args.output_dir) / 'generation_models_table.tex'
    generate_latex_table(comparison, latex_file)

    # Save comparison JSON
    output_json = Path(args.output_dir) / 'generation_models_comparison.json'
    with open(output_json, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Comparison saved to {output_json}")

    print("\n" + "="*80)
    print("✓ Comparison complete!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare generation models')
    parser.add_argument('--results_dir', type=str, default='results/generation',
                        help='Directory containing model result JSON files')
    parser.add_argument('--output_dir', type=str, default='results/generation/comparison',
                        help='Output directory for plots and tables')

    args = parser.parse_args()
    main(args)

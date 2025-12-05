"""
Generate all figures for the GCN paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style with larger fonts
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 13

# Results directory
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results'
FIGURES_DIR = Path(__file__).parent.parent.parent / 'paper' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color palette - expanded for 11 datasets
COLORS = {
    # Chemistry
    'qm9': '#2ecc71',      # green
    'zinc': '#27ae60',     # dark green
    # Citations
    'arxiv': '#3498db',    # blue
    'cora': '#2980b9',     # dark blue
    'citeseer': '#5dade2', # light blue
    # Social
    'reddit': '#f39c12',   # orange
    # Biology
    'protein': '#e74c3c',  # red
    'enzymes': '#c0392b',  # dark red
    # Benchmarks
    'er': '#9b59b6',       # purple
    'ba': '#8e44ad',       # dark purple
    'ws': '#bb8fce',       # light purple
}

COMPONENT_COLORS = {
    'structural': '#3498db',  # blue
    'edge_type': '#2ecc71',   # green
    'bridging': '#e74c3c',    # red
}


def load_summary():
    """Load main experimental summary"""
    with open(RESULTS_DIR / 'gcn' / 'summary.json', 'r') as f:
        return json.load(f)


def load_baselines():
    """Load baseline results"""
    baseline_path = RESULTS_DIR / 'baselines' / 'baseline_results.json'
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            return json.load(f)
    return None


def load_sensitivity():
    """Load sensitivity analysis results"""
    sensitivity = {}
    for filename in ['k2.json', 'k3.json', 'k4.json', 'k5.json',
                     'weights.json', 'corpus_size.json']:
        filepath = RESULTS_DIR / 'sensitivity' / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                key = filename.replace('.json', '')
                sensitivity[key] = json.load(f)
    return sensitivity


def load_validation():
    """Load validation results"""
    validation_path = RESULTS_DIR / 'validation' / 'summary.json'
    if validation_path.exists():
        with open(validation_path, 'r') as f:
            return json.load(f)
    return None


def figure1_overall_novelty(summary):
    """
    Figure 1: Bar plot of overall novelty by dataset
    """
    print("Generating Figure 1: Overall novelty by dataset...")

    datasets = []
    means = []
    stds = []

    for dataset, data in summary.items():
        datasets.append(dataset.upper())
        means.append(data['overall_novelty']['mean'])
        stds.append(data['overall_novelty']['std'])

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(datasets))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                   color=[COLORS[d.lower()] for d in datasets],
                   alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Dataset', fontweight='bold', fontsize=16)
    ax.set_ylabel('Overall Novelty Score', fontweight='bold', fontsize=16)
    ax.set_title('Graph Compositional Novelty by Dataset', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure1_overall_novelty.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure1_overall_novelty.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {FIGURES_DIR / 'figure1_overall_novelty.pdf'}")


def figure2_component_breakdown(summary):
    """
    Figure 2: Stacked bar chart showing component breakdown
    """
    print("Generating Figure 2: Component breakdown...")

    datasets = []
    structural = []
    edge_type = []
    bridging = []

    for dataset, data in summary.items():
        datasets.append(dataset.upper())
        structural.append(data['structural_novelty']['mean'])
        edge_type.append(data['edge_novelty']['mean'])
        bridging.append(data['bridging_novelty']['mean'])

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(datasets))
    width = 0.6

    # Create grouped bars (not stacked, side-by-side components)
    component_width = width / 3

    bars1 = ax.bar(x - component_width, structural, component_width,
                    label='Structural', color=COMPONENT_COLORS['structural'],
                    alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, edge_type, component_width,
                    label='Edge-Type', color=COMPONENT_COLORS['edge_type'],
                    alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + component_width, bridging, component_width,
                    label='Bridging', color=COMPONENT_COLORS['bridging'],
                    alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Dataset', fontweight='bold', fontsize=16)
    ax.set_ylabel('Novelty Score', fontweight='bold', fontsize=16)
    ax.set_title('Novelty Component Breakdown by Dataset', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=14, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, framealpha=0.9, fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure2_component_breakdown.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure2_component_breakdown.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {FIGURES_DIR / 'figure2_component_breakdown.pdf'}")


def figure3_motif_size_sensitivity(sensitivity):
    """
    Figure 3: Line plot showing sensitivity to motif size k
    """
    print("Generating Figure 3: Motif size sensitivity...")

    if not any(k in sensitivity for k in ['k2', 'k3', 'k4', 'k5']):
        print("  ⚠ No sensitivity data found, skipping...")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    k_values = [2, 3, 4, 5]

    for dataset in ['qm9', 'arxiv', 'protein', 'reddit']:
        novelty_means = []
        novelty_stds = []

        for k in k_values:
            key = f'k{k}'
            if key in sensitivity and dataset in sensitivity[key]:
                data = sensitivity[key][dataset]
                novelty_means.append(data['mean_novelty'])
                novelty_stds.append(data['std_novelty'])
            else:
                novelty_means.append(np.nan)
                novelty_stds.append(0)

        # Only plot if we have data
        if not all(np.isnan(novelty_means)):
            ax.errorbar(k_values, novelty_means, yerr=novelty_stds,
                       marker='o', markersize=8, linewidth=2, capsize=5,
                       label=dataset.upper(), color=COLORS[dataset], alpha=0.8)

    ax.set_xlabel('Motif Size (k)', fontweight='bold', fontsize=16)
    ax.set_ylabel('Overall Novelty Score', fontweight='bold', fontsize=16)
    ax.set_title('Sensitivity to Motif Size Parameter', fontweight='bold', fontsize=18)
    ax.set_xticks(k_values)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, framealpha=0.9, fontsize=13)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure3_motif_sensitivity.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure3_motif_sensitivity.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {FIGURES_DIR / 'figure3_motif_sensitivity.pdf'}")


def figure4_weight_sensitivity(sensitivity):
    """
    Figure 4: Heatmap of weight sensitivity
    """
    print("Generating Figure 4: Weight sensitivity...")

    if 'weights' not in sensitivity:
        print("  ⚠ No weight sensitivity data found, skipping...")
        return

    # Extract weight configurations and scores
    weight_data = sensitivity['weights']

    # For each dataset, create a small sensitivity plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    datasets = ['qm9', 'arxiv', 'protein', 'reddit']

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]

        if dataset not in weight_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(dataset.upper())
            continue

        # Handle both 'configurations' and 'trials' formats
        configs = weight_data[dataset].get('configurations', weight_data[dataset].get('trials', []))

        if not configs:
            ax.text(0.5, 0.5, 'No configurations', ha='center', va='center')
            ax.set_title(dataset.upper())
            continue

        # Extract weights and scores (handle both formats)
        w_structural = [c['weights'][0] for c in configs]
        w_edge = [c['weights'][1] for c in configs]
        scores = [c.get('overall_novelty', c.get('mean_novelty', 0)) for c in configs]

        # Create scatter plot
        scatter = ax.scatter(w_structural, w_edge, c=scores, s=100,
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)

        ax.set_xlabel('Structural Weight', fontweight='bold')
        ax.set_ylabel('Edge-Type Weight', fontweight='bold')
        ax.set_title(f'{dataset.upper()}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Overall Novelty')

    plt.suptitle('Weight Configuration Sensitivity', fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure4_weight_sensitivity.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure4_weight_sensitivity.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {FIGURES_DIR / 'figure4_weight_sensitivity.pdf'}")


def figure5_baseline_comparison(summary, baselines):
    """
    Figure 5: Baseline comparison
    """
    print("Generating Figure 5: Baseline comparison...")

    if baselines is None:
        print("  ⚠ No baseline data found, skipping...")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get datasets
    datasets = list(summary.keys())
    methods = ['GCN', 'SET', 'MMD', 'GED', 'KERNEL', 'EMB']

    x = np.arange(len(datasets))
    width = 0.14

    # GCN scores
    gcn_scores = [summary[d]['overall_novelty']['mean'] for d in datasets]

    # Plot GCN
    ax.bar(x - 2.5*width, gcn_scores, width, label='GCN (Ours)',
           color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Plot baselines
    baseline_methods = ['SET', 'MMD', 'GED', 'KERNEL', 'EMB']
    baseline_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#95a5a6']

    for i, (method, color) in enumerate(zip(baseline_methods, baseline_colors)):
        scores = []
        for dataset in datasets:
            # Get method score from nested structure
            if dataset in baselines and method in baselines[dataset]:
                scores.append(baselines[dataset][method]['mean'])
            else:
                scores.append(0)

        ax.bar(x + (i - 1.5)*width, scores, width, label=method,
               color=color, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Dataset', fontweight='bold', fontsize=16)
    ax.set_ylabel('Novelty Score', fontweight='bold', fontsize=16)
    ax.set_title('Baseline Method Comparison', fontweight='bold', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=14, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, framealpha=0.9, fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure5_baseline_comparison.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure5_baseline_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {FIGURES_DIR / 'figure5_baseline_comparison.pdf'}")


def figure6_validation_correlations(validation):
    """
    Figure 6: Validation correlation results
    """
    print("Generating Figure 6: Validation correlations...")

    if validation is None:
        print("  ⚠ No validation data found, skipping...")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    tasks = ['molecular_synthesis', 'citation_impact', 'synthetic_downstream', 'expert_ranking']
    titles = ['Molecular Synthesis (SA-score)', 'Citation Impact',
              'Downstream Classification', 'Expert Ranking (Kendall τ)']

    for idx, (task, title) in enumerate(zip(tasks, titles)):
        ax = axes[idx]

        if task not in validation:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title, fontweight='bold')
            continue

        task_data = validation[task]

        if task == 'expert_ranking':
            # Bar plot of Kendall's tau by dataset
            datasets = list(task_data.keys())
            taus = [task_data[d]['mean_tau'] for d in datasets]
            stds = [task_data[d]['std_tau'] for d in datasets]

            x_pos = np.arange(len(datasets))
            bars = ax.bar(x_pos, taus, yerr=stds, capsize=5,
                         color=[COLORS.get(d, '#95a5a6') for d in datasets],
                         alpha=0.8, edgecolor='black', linewidth=1.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([d.upper() for d in datasets], fontsize=14)
            ax.set_ylabel("Kendall's τ", fontweight='bold', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_ylim(-0.5, 0.5)

        elif task == 'synthetic_downstream':
            # Line plot of accuracy vs novelty range
            datasets = list(task_data.keys())

            for dataset in datasets[:2]:  # Plot first 2 datasets to avoid clutter
                data = task_data[dataset]
                if 'range_results' in data:
                    ranges = [r['novelty_range'] for r in data['range_results']]
                    accs = [r['mean_accuracy'] for r in data['range_results']]
                    stds = [r['std_accuracy'] for r in data['range_results']]

                    x_labels = [f"{r[0]:.1f}-{r[1]:.1f}" for r in ranges]
                    x_pos = np.arange(len(ranges))

                    ax.errorbar(x_pos, accs, yerr=stds, marker='o',
                               label=dataset.upper(), capsize=5, linewidth=2,
                               color=COLORS.get(dataset, '#95a5a6'))

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, rotation=45)
            ax.set_xlabel('Novelty Range', fontweight='bold')
            ax.set_ylabel('Accuracy', fontweight='bold')
            ax.legend()
            ax.set_ylim(0.5, 1.0)

        else:
            # Scatter plot for correlation tasks
            ax.text(0.5, 0.5, f'{task} validation\n(scatter plot placeholder)',
                   ha='center', va='center')

        ax.set_title(title, fontweight='bold', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(alpha=0.3)

    plt.suptitle('Predictive Validation Results', fontweight='bold', fontsize=20, y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure6_validation.pdf', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure6_validation.png', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to {FIGURES_DIR / 'figure6_validation.pdf'}")


def main():
    print("="*60)
    print("Generating All Figures for GCN Paper")
    print("="*60)

    # Load all data
    print("\nLoading results...")
    summary = load_summary()
    baselines = load_baselines()
    sensitivity = load_sensitivity()
    validation = load_validation()

    print(f"  ✓ Loaded summary: {len(summary)} datasets")
    print(f"  ✓ Loaded baselines: {len(baselines) if baselines else 0} results")
    print(f"  ✓ Loaded sensitivity: {len(sensitivity)} experiments")
    print(f"  ✓ Loaded validation: {'Yes' if validation else 'No'}")

    # Generate all figures
    print("\n" + "="*60)
    figure1_overall_novelty(summary)
    figure2_component_breakdown(summary)
    figure3_motif_size_sensitivity(sensitivity)
    figure4_weight_sensitivity(sensitivity)
    figure5_baseline_comparison(summary, baselines)
    figure6_validation_correlations(validation)

    print("\n" + "="*60)
    print("✓ All figures generated successfully!")
    print(f"✓ Figures saved to: {FIGURES_DIR}")
    print("="*60)

    # List all generated figures
    print("\nGenerated files:")
    for fig_file in sorted(FIGURES_DIR.glob('figure*.pdf')):
        size_kb = fig_file.stat().st_size / 1024
        print(f"  - {fig_file.name} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()

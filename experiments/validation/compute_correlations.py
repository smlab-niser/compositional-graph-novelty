"""
Compute summary statistics across all validation experiments
"""

import argparse
import json
from pathlib import Path


def main(args):
    results_dir = Path(args.results_dir)

    print("Computing validation summary...")

    summary = {}

    # Load individual validation results
    validation_files = {
        'molecular_synthesis': 'molecular_synthesis.json',
        'citation_impact': 'citation_impact.json',
        'synthetic_downstream': 'synthetic_downstream.json',
        'expert_ranking': 'expert_ranking.json'
    }

    for name, filename in validation_files.items():
        filepath = results_dir / filename

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                summary[name] = data
                print(f"  ✓ Loaded {name}")
        else:
            print(f"  ⚠ Missing {name}")
            summary[name] = None

    # Compute overall validation score
    validation_scores = []

    # Molecular synthesis correlation
    if summary['molecular_synthesis']:
        rho = abs(summary['molecular_synthesis']['correlation']['rho'])
        validation_scores.append(rho)

    # Citation impact correlation
    if summary['citation_impact']:
        rho = abs(summary['citation_impact']['correlation']['rho'])
        validation_scores.append(rho)

    # Expert agreement
    if summary['expert_ranking']:
        for dataset_data in summary['expert_ranking'].values():
            if isinstance(dataset_data, dict) and 'mean_tau' in dataset_data:
                validation_scores.append(dataset_data['mean_tau'])

    if validation_scores:
        summary['overall_validation_score'] = float(sum(validation_scores) / len(validation_scores))
    else:
        summary['overall_validation_score'] = None

    # Save summary
    output_path = Path(args.output)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to {output_path}")

    if summary['overall_validation_score']:
        print(f"  Overall validation score: {summary['overall_validation_score']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    main(args)

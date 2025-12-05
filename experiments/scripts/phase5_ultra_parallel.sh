#!/bin/bash

# Phase 5: Predictive Validation (ULTRA parallelized - 96 cores)
# Each experiment uses multiprocessing internally

set -e

echo "=" * 80
echo " Phase 5: Predictive Validation (Ultra Parallel - 96 cores)"
echo "=" * 80

# Create output directory
mkdir -p results/validation logs/phase5

PYTHON="/home/smlab/miniconda3/envs/novelty-gnn/bin/python"
N_CORES=96

echo ""
echo "Configuration:"
echo "  Total CPUs: $N_CORES"
echo "  Strategy: Run validation tasks sequentially, each using all CPUs"
echo ""

# Molecular synthesis correlation
echo "=" * 80
echo "1. Molecular Synthesis Difficulty Correlation"
echo "=" * 80

if [ -f "results/validation/molecular_synthesis.json" ]; then
    echo "  ✓ Molecular synthesis already exists, skipping"
else
    echo "Computing correlation with SA-scores ($N_CORES cores)..."
    echo "Processing up to 50,000 molecules..."

    $PYTHON experiments/validation/molecular_synthesis_parallel.py \
        --dataset qm9 \
        --n_molecules 50000 \
        --n_cores $N_CORES \
        --output results/validation/molecular_synthesis.json \
        2>&1 | tee logs/phase5/molecular_synthesis.log

    echo "  ✓ Completed molecular synthesis correlation"
fi

# Citation impact
echo ""
echo "=" * 80
echo "2. Citation Impact Correlation"
echo "=" * 80

if [ -f "results/validation/citation_impact.json" ]; then
    echo "  ✓ Citation impact already exists, skipping"
else
    echo "Computing citation impact correlation..."

    $PYTHON experiments/validation/citation_impact.py \
        --dataset arxiv \
        --citation_window_years 2 \
        --control_variables author_h_index,venue,year \
        --output results/validation/citation_impact.json \
        2>&1 | tee logs/phase5/citation_impact.log

    echo "  ✓ Completed citation impact correlation"
fi

# Downstream performance
echo ""
echo "=" * 80
echo "3. Synthetic Graph Downstream Performance"
echo "=" * 80

if [ -f "results/validation/synthetic_downstream.json" ]; then
    echo "  ✓ Synthetic downstream already exists, skipping"
else
    echo "Computing downstream task performance..."

    $PYTHON experiments/validation/synthetic_downstream.py \
        --datasets reddit,protein \
        --task node_classification \
        --n_trials 10 \
        --novelty_ranges 0.0-0.3,0.3-0.5,0.5-0.7,0.7-1.0 \
        --output results/validation/synthetic_downstream.json \
        2>&1 | tee logs/phase5/synthetic_downstream.log

    echo "  ✓ Completed downstream performance"
fi

# Expert ranking
echo ""
echo "=" * 80
echo "4. Expert Ranking Agreement"
echo "=" * 80

if [ -f "results/validation/expert_ranking.json" ]; then
    echo "  ✓ Expert ranking already exists, skipping"
else
    echo "Collecting expert rankings..."

    $PYTHON experiments/validation/expert_ranking.py \
        --n_experts 5 \
        --n_pairs 30 \
        --datasets qm9,arxiv \
        --output results/validation/expert_ranking.json \
        2>&1 | tee logs/phase5/expert_ranking.log

    echo "  ✓ Completed expert ranking"
fi

# Compute summary
echo ""
echo "=" * 80
echo "Computing Validation Summary"
echo "=" * 80

echo "Aggregating correlation statistics..."
$PYTHON experiments/validation/compute_correlations.py \
    --results_dir results/validation/ \
    --output results/validation/summary.json \
    2>&1 | tee logs/phase5/summary.log

echo ""
echo "=" * 80
echo "Phase 5 Complete!"
echo "=" * 80
echo ""
echo "Generated files:"
ls -lh results/validation/*.json

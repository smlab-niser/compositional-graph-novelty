#!/bin/bash

# Phase 1: Corpus Preprocessing
# Extract motifs and compute corpus statistics for all datasets
# Run in parallel on 4 GPUs (one dataset per GPU)

set -e

echo "Starting Phase 1: Corpus Preprocessing"
echo "Processing 4 datasets in parallel on 4 GPUs..."

# Create output directory
mkdir -p data/processed

# Function to preprocess a dataset
preprocess_dataset() {
    local GPU=$1
    local DATASET=$2

    echo "GPU $GPU: Starting $DATASET preprocessing"

    CUDA_VISIBLE_DEVICES=$GPU /home/smlab/miniconda3/envs/novelty-gnn/bin/python experiments/preprocess_corpus.py \
        --dataset $DATASET \
        --k 3 \
        --output data/processed/${DATASET}_corpus.pkl

    echo "GPU $GPU: Completed $DATASET preprocessing"
}

# Launch parallel preprocessing (one dataset per GPU)
preprocess_dataset 0 qm9 &
PID1=$!

preprocess_dataset 1 arxiv &
PID2=$!

preprocess_dataset 2 reddit &
PID3=$!

preprocess_dataset 3 protein &
PID4=$!

# Wait for all to complete
echo "Waiting for all preprocessing to complete..."

wait $PID1
echo "✓ GPU 0 (qm9) complete"

wait $PID2
echo "✓ GPU 1 (arxiv) complete"

wait $PID3
echo "✓ GPU 2 (reddit) complete"

wait $PID4
echo "✓ GPU 3 (protein) complete"

echo "Phase 1 complete! All corpus statistics saved."
echo "Output files:"
ls -lh data/processed/*_corpus.pkl

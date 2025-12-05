# Compositional Novelty Metrics for Graph-Structured Data

[![Paper](https://img.shields.io/badge/Paper-blue)](https://github.com/smlab-niser/compositional-graph-novelty)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Compositional Novelty Metrics for Graph-Structured Data"** (ICPR 2026).

**Authors**: Rucha Bhalchandra Joshi¹, Subhankar Mishra²  
¹ The Cyprus Institute  
² NISER, Bhubaneswar

---

## Overview

A compositional framework for measuring novelty in graph-structured data through three interpretable components:

1. **Structural Novelty** - Rarity of local motif patterns
2. **Edge-Type Novelty** - Rarity of node-type relationships  
3. **Bridging Novelty** - Semantic distance bridged by connections

---

## Installation

```bash
git clone https://github.com/smlab-niser/compositional-graph-novelty.git
cd compositional-graph-novelty

# Create environment
conda create -n gcn-novelty python=3.11
conda activate gcn-novelty

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
from src.gcn.core import GraphCompositionalNovelty
from src.utils.data_loader import load_dataset

# Load dataset
corpus_graphs, eval_graphs = load_dataset('qm9', corpus_size=1000, eval_size=100)

# Initialize metric
gcn = GraphCompositionalNovelty(corpus_graphs, k=3)

# Compute novelty
novelty = gcn.compute_novelty(eval_graphs[0])
print(f"Overall: {novelty['overall_novelty']:.3f}")
print(f"Structural: {novelty['structural_novelty']:.3f}")
print(f"Edge-Type: {novelty['edge_novelty']:.3f}")
print(f"Bridging: {novelty['bridging_novelty']:.3f}")
```

**Example output**:
```
Overall: 0.631
Structural: 0.988
Edge-Type: 0.710
Bridging: 0.075
```

---

## Reproducing Paper Results

### Main Evaluation (Sections 5.1-5.2)
```bash
python experiments/main/evaluate_datasets.py \
    --datasets qm9 zinc arxiv cora citeseer reddit protein enzymes \
    --k 3 --corpus_size 1000 --eval_size 100
```

### Baseline Comparison (Section 5.3)
```bash
python experiments/baselines/compare_baselines.py --datasets qm9 arxiv reddit
```

### Synthesis Validation (Section 6.1)
```bash
python experiments/validation/synthesis_correlation.py --dataset qm9 --n_samples 1000
```

### Sensitivity Analysis (Sections 6.2-6.4)
```bash
# Motif size (k=2,3,4,5)
python experiments/sensitivity/motif_size.py --k 2 3 4 5 --datasets qm9 arxiv

# Component weights
python experiments/sensitivity/component_weights.py --n_configs 100 --datasets qm9

# Corpus size
python experiments/sensitivity/corpus_size.py --sizes 100 500 1000 5000 --dataset qm9
```

### Generation Comparison (Section 5.7)
```bash
bash experiments/generation/run_simple_experiments.sh
python experiments/generation/compare_generation_models.py
```

---

## Supported Datasets

| Dataset | Domain | Graphs | Avg Nodes | Avg Edges |
|---------|--------|--------|-----------|-----------|
| QM9 | Molecular | 134k | 18.0 | 18.7 |
| ZINC | Molecular | 250k | 23.2 | 24.9 |
| ArXiv | Citation | 170k | 1.0 | 4.5 |
| Reddit | Social | 233k | 1.0 | 492.9 |
| Protein | Biological | 1.1k | 39.1 | 72.8 |

*Datasets auto-download on first use*

---

## Repository Structure

```
compositional-graph-novelty/
├── src/
│   ├── gcn/              # Core implementation
│   └── utils/            # Data loading & utilities
├── experiments/          # All paper experiments
│   ├── main/            # Main evaluation
│   ├── baselines/       # Baseline comparison
│   ├── validation/      # Synthesis validation
│   ├── sensitivity/     # Sensitivity analysis
│   └── generation/      # Generation comparison
├── examples/            # Usage examples
└── results/             # Experimental results (JSON)
```

---

## Citation

```bibtex
@inproceedings{joshi2026compositional,
  title={Compositional Novelty Metrics for Graph-Structured Data},
  author={Joshi, Rucha Bhalchandra and Mishra, Subhankar},
  booktitle={},
  year={2026},
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Contact

- **Rucha Bhalchandra Joshi**: r.joshi@cyi.ac.cy
- **Subhankar Mishra**: smishra@niser.ac.in
- **Issues**: [GitHub Issues](https://github.com/smlab-niser/compositional-graph-novelty/issues)

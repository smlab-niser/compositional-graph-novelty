# Phase 4 Sensitivity Analysis Scripts

Quick reference for running motif size sensitivity experiments with different parallelization strategies.

## Available Scripts

### 1. Original Script (Limited Parallelization)
**File**: `phase4_sensitivity_96core.sh`
- Runs all experiments (k=2,3,4,5)
- 7 parallel jobs, ~16-28 cores used
- **Use when**: You want to run all sensitivity analyses together

```bash
./experiments/scripts/phase4_sensitivity_96core.sh
```

---

### 2. Optimized Full Script (Better Parallelization)
**File**: `phase4_sensitivity_96core_optimized.sh`
- Runs all experiments (k=2,3,4,5)
- 26 parallel jobs, ~26 cores used
- **Use when**: You want better CPU utilization for all experiments

```bash
./experiments/scripts/phase4_sensitivity_96core_optimized.sh
```

---

### 3. Multi-Seed Full Script (Maximum Parallelization)
**File**: `phase4_sensitivity_96core_multiseed.sh`
- Runs all experiments with 4 random seeds
- 104 parallel jobs, ~96 cores used (full utilization)
- Provides statistical robustness
- **Use when**: You want research-quality results with confidence intervals

```bash
./experiments/scripts/phase4_sensitivity_96core_multiseed.sh
```

---

### 4. Selective K-values (Single Run) ⭐
**File**: `phase4_sensitivity_selective.sh`
- Run **only specific k-values** (e.g., k=4,5)
- Single run per (k, dataset) combination
- **Use when**: You only need specific k-values, single run

```bash
# Run only k=4 and k=5
./experiments/scripts/phase4_sensitivity_selective.sh 4 5

# Run only k=3
./experiments/scripts/phase4_sensitivity_selective.sh 3

# Run all k-values (same as optimized version)
./experiments/scripts/phase4_sensitivity_selective.sh 2 3 4 5
```

**Jobs**: `n_k_values × 4 datasets`
- For k=4,5: **8 parallel jobs** (2 k-values × 4 datasets)
- For k=4 only: **4 parallel jobs** (1 k-value × 4 datasets)

---

### 5. Selective K-values (Multi-Seed) ⭐⭐
**File**: `phase4_sensitivity_selective_multiseed.sh`
- Run **only specific k-values** with 4 random seeds
- Statistical robustness for selected k-values
- **Use when**: You need specific k-values with confidence intervals

```bash
# Run k=4 and k=5 with 4 seeds each
./experiments/scripts/phase4_sensitivity_selective_multiseed.sh 4 5

# Run only k=3 with 4 seeds
./experiments/scripts/phase4_sensitivity_selective_multiseed.sh 3

# Run all k-values with 4 seeds each
./experiments/scripts/phase4_sensitivity_selective_multiseed.sh 2 3 4 5
```

**Jobs**: `n_k_values × 4 datasets × 4 seeds`
- For k=4,5: **32 parallel jobs** (2 k-values × 4 datasets × 4 seeds)
- For k=4 only: **16 parallel jobs** (1 k-value × 4 datasets × 4 seeds)

---

## Quick Examples

### Example 1: Run only k=4 and k=5 (single run)
```bash
cd /home/smlab/projects/graph-novelty
./experiments/scripts/phase4_sensitivity_selective.sh 4 5
```
- **Jobs**: 8 (2 k-values × 4 datasets)
- **Runtime**: ~5-10 minutes
- **Output**: `results/sensitivity/k4.json`, `results/sensitivity/k5.json`

### Example 2: Run only k=4 and k=5 (with 4 seeds)
```bash
cd /home/smlab/projects/graph-novelty
./experiments/scripts/phase4_sensitivity_selective_multiseed.sh 4 5
```
- **Jobs**: 32 (2 k-values × 4 datasets × 4 seeds)
- **Runtime**: ~5-10 minutes (same time due to parallelization!)
- **Output**:
  - Aggregated: `results/sensitivity/k4.json`, `results/sensitivity/k5.json`
  - Individual: `results/sensitivity/motif_size/k4_*_seed*.json`

### Example 3: Run only k=3 (single dataset for testing)
```bash
cd /home/smlab/projects/graph-novelty

# Edit the script to test with one dataset
./experiments/scripts/phase4_sensitivity_selective.sh 3
# (will run on all 4 datasets: qm9, arxiv, reddit, protein)
```

---

## Output Files

### Single Run Scripts
```
results/sensitivity/
├── k2.json          # Aggregated results for k=2 (all datasets)
├── k3.json          # Aggregated results for k=3
├── k4.json          # Aggregated results for k=4
├── k5.json          # Aggregated results for k=5
└── motif_size/
    ├── k2_qm9.json
    ├── k2_arxiv.json
    ├── k2_reddit.json
    ├── k2_protein.json
    ├── k3_qm9.json
    └── ...
```

### Multi-Seed Scripts
```
results/sensitivity/
├── k2.json          # Aggregated across seeds
├── k3.json
├── k4.json
├── k5.json
└── motif_size/
    ├── k2_qm9_seed0.json
    ├── k2_qm9_seed1.json
    ├── k2_qm9_seed2.json
    ├── k2_qm9_seed3.json
    ├── k2_arxiv_seed0.json
    └── ...
```

---

## Monitoring CPU Usage

While the script is running, monitor CPU usage in another terminal:

```bash
# Count active Python processes
watch -n 1 'ps aux | grep "python.*motif_size" | wc -l'

# Visual CPU monitoring
htop

# Detailed process view
ps aux | grep python
```

---

## Comparison Table

| Script | K-values | Seeds | Jobs | Cores Used | Runtime | Statistical |
|--------|----------|-------|------|------------|---------|-------------|
| `selective.sh 4 5` | 4, 5 | 1 | 8 | 8 | ~5-10 min | Single run |
| `selective_multiseed.sh 4 5` | 4, 5 | 4 | 32 | 32 | ~5-10 min | ✓ Robust |
| `selective.sh 3` | 3 | 1 | 4 | 4 | ~5 min | Single run |
| `selective_multiseed.sh 3` | 3 | 4 | 16 | 16 | ~5 min | ✓ Robust |
| `optimized.sh` | 2,3,4,5 | 1 | 16 | 16 | ~10-15 min | Single run |
| `multiseed.sh` | 2,3,4,5 | 4 | 64 | 64 | ~10-15 min | ✓ Robust |

---

## Recommendation for k=4,5 Only

**For single run**:
```bash
./experiments/scripts/phase4_sensitivity_selective.sh 4 5
```

**For research-quality results** (recommended):
```bash
./experiments/scripts/phase4_sensitivity_selective_multiseed.sh 4 5
```

This runs 32 parallel jobs (2 k-values × 4 datasets × 4 seeds) and provides:
- Mean novelty scores across 4 random seeds
- Standard deviation and confidence intervals
- Statistical robustness
- Same runtime as single-seed version (thanks to parallelization!)

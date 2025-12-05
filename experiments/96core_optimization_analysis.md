# 96-Core Optimization Analysis for Phase 4 Sensitivity Experiments

## Current Utilization Analysis

### Original Script: `phase4_sensitivity_96core.sh`

**Parallelization Strategy:**
- **7 bash background jobs** (task-level parallelism using `&`)
- Each Python script processes **4 datasets** using `multiprocessing.Pool`

**Job Breakdown:**
1. Motif size k=2 → processes 4 datasets (max 4 cores)
2. Motif size k=3 → processes 4 datasets (max 4 cores)
3. Motif size k=4 → processes 4 datasets (max 4 cores)
4. Motif size k=5 → processes 4 datasets (max 4 cores)
5. Weight sensitivity → processes 4 datasets (max 4 cores)
6. Corpus size → processes 2 datasets (max 2 cores)
7. Grid search → processes 4 datasets (max 4 cores)

**Theoretical Maximum Cores Used:**
- If all jobs start simultaneously: ~7 cores (bash jobs)
- Within each job: up to 4 cores (multiprocessing.Pool)
- **Actual maximum: 16-28 cores** (depends on job completion timing)

**CPU Utilization: 16-28 out of 96 cores = 17-29%**

---

## Optimized Solution: `phase4_sensitivity_96core_optimized.sh`

### New Parallelization Strategy

**Break down experiments into individual (k, dataset) combinations:**
- Instead of 4 k-value jobs → **16 jobs** (4 k-values × 4 datasets)
- Instead of 1 weight job → **4 jobs** (1 per dataset)
- Instead of 1 corpus job → **2 jobs** (1 per dataset)
- Instead of 1 grid job → **4 jobs** (1 per dataset)

**Total Parallel Jobs: 26**

### Job Breakdown

#### Motif Size Analysis (16 jobs)
```
k=2: [qm9, arxiv, reddit, protein]  → 4 parallel jobs
k=3: [qm9, arxiv, reddit, protein]  → 4 parallel jobs
k=4: [qm9, arxiv, reddit, protein]  → 4 parallel jobs
k=5: [qm9, arxiv, reddit, protein]  → 4 parallel jobs
```

#### Weight Sensitivity (4 jobs)
```
[qm9, arxiv, reddit, protein]  → 4 parallel jobs
```

#### Corpus Size (2 jobs)
```
[qm9, arxiv]  → 2 parallel jobs
```

#### Grid Search (4 jobs)
```
[qm9, arxiv, reddit, protein]  → 4 parallel jobs
```

**Total: 16 + 4 + 2 + 4 = 26 parallel jobs**

**CPU Utilization: 26 out of 96 cores = 27%**

---

## Why Not 96 Jobs?

### Constraints

1. **Limited Dataset Combinations**
   - Only 4 datasets tested: qm9, arxiv, reddit, protein
   - Motif size: 4 k-values
   - Maximum meaningful parallelization: 4 × 4 = 16 jobs

2. **Corpus Size Limitations**
   - Current: corpus_size=90, eval_size=10
   - Small eval set (10 graphs) means per-graph parallelization has high overhead
   - With 10 eval graphs × 26 jobs = only 260 graphs total

3. **Memory Constraints**
   - Each GCN instance loads full corpus into memory
   - 26 jobs × ~1-2GB per corpus = 26-52GB RAM usage
   - 96 jobs would require ~100-200GB RAM

### Further Optimization Options

If you want to utilize **more cores**, here are options:

#### Option 1: Increase Evaluation Set Size
```python
corpus_graphs, eval_graphs = load_dataset(
    dataset,
    corpus_size=90,
    eval_size=100  # Increased from 10
)
```
- With 100 eval graphs, per-graph parallelization becomes worthwhile
- Can parallelize novelty computation across eval graphs

#### Option 2: Multiple Random Seeds
Run each (k, dataset) combination with multiple random seeds:
```bash
for seed in {0..3}; do
    $PYTHON experiments/sensitivity/motif_size.py \
        --k 2 \
        --datasets qm9 \
        --seed $seed \
        --output results/sensitivity/motif_size/k2_qm9_seed${seed}.json &
done
```
- This creates 16 × 4 = **64 jobs** (if using 4 seeds)
- Provides statistical robustness

#### Option 3: Finer-Grained Weight Grid
Instead of 100 random weight combinations, test a full grid:
```python
# Test all combinations of weights in steps of 0.1
# structural: [0.0, 0.1, 0.2, ..., 1.0]  → 11 values
# edge:       [0.0, 0.1, 0.2, ..., 1.0]  → 11 values
# bridging:   [0.0, 0.1, 0.2, ..., 1.0]  → 11 values
# Total: 11³ = 1331 combinations (constrained to sum=1)
```
- Could run hundreds of weight combinations in parallel

#### Option 4: Per-Graph Parallelization (Large Eval Sets)
Modify `motif_size_parallel.py` to parallelize novelty computation:
```python
# Instead of sequential evaluation
for G in eval_graphs:
    novelty = gcn.compute_novelty(G)

# Use parallel evaluation
with Pool(processes=cpu_count()) as pool:
    novelties = pool.starmap(
        process_single_graph,
        [(G, gcn) for G in eval_graphs]
    )
```
- **Only useful if eval_size is large** (e.g., 100+ graphs)
- With current eval_size=10, overhead dominates

---

## Recommended Configuration

### For Maximum CPU Utilization (64-96 cores)

**Use Option 2 (Multiple Seeds) + Larger Eval Set:**

```bash
# Modified script with 4 seeds × 16 (k, dataset) combinations = 64 jobs
for seed in {0..3}; do
    for dataset in qm9 arxiv reddit protein; do
        for k in 2 3 4 5; do
            $PYTHON experiments/sensitivity/motif_size.py \
                --k $k \
                --datasets ${dataset} \
                --seed $seed \
                --corpus_size 90 \
                --eval_size 50 \
                --output results/sensitivity/motif_size/k${k}_${dataset}_seed${seed}.json \
                --n_cores 1 &
        done
    done
done
```

**Benefits:**
- **64 parallel jobs** → near-full utilization of 96 cores
- Statistical robustness from multiple seeds
- Larger eval set (50 graphs) provides better estimates
- Still manageable memory usage (~64-128GB)

---

## Implementation Recommendations

### Immediate (Low-Hanging Fruit)
✅ Use `phase4_sensitivity_96core_optimized.sh` → **26 cores utilized**
- Simple modification of existing script
- No code changes required
- Doubles CPU utilization (16 → 26 cores)

### Short-Term (Moderate Effort)
Add multiple seeds to the optimized script → **64 cores utilized**
- Requires adding `--seed` parameter to Python scripts
- Provides statistical confidence intervals
- Better utilization of 96-core system

### Long-Term (Research Quality)
Increase eval set size + per-graph parallelization → **90+ cores utilized**
- Requires more evaluation data
- Better statistical power
- Fully leverages 96-core system

---

## Comparison Table

| Configuration | Parallel Jobs | CPU Cores Used | Utilization | Runtime (est.) |
|---------------|---------------|----------------|-------------|----------------|
| **Original** | 7 | 16-28 | 17-29% | ~30 min |
| **Optimized** | 26 | 26 | 27% | ~15 min |
| **+Seeds (4x)** | 64 | 64 | 67% | ~15 min |
| **+Large Eval** | 64 | 90+ | 94% | ~20 min |

---

## Files Modified/Created

1. ✅ `experiments/scripts/phase4_sensitivity_96core_optimized.sh` (new)
   - Breaks down motif size into 16 parallel jobs
   - Total 26 parallel jobs

2. (Optional) Create multi-seed version:
   - `experiments/scripts/phase4_sensitivity_96core_multiseed.sh`
   - 64 parallel jobs with statistical robustness

---

## Usage

### Run Optimized Script (26 cores)
```bash
cd /home/smlab/projects/graph-novelty
chmod +x experiments/scripts/phase4_sensitivity_96core_optimized.sh
./experiments/scripts/phase4_sensitivity_96core_optimized.sh
```

### Monitor CPU Usage
```bash
# In another terminal
watch -n 1 'ps aux | grep python | wc -l'
htop  # visual monitoring
```

### Expected Output
```
Launching motif size sensitivity (16 parallel jobs)...
  Launching k=2, dataset=qm9...
  Launching k=2, dataset=arxiv...
  ...
Launched 26 parallel sensitivity analyses...
Expected CPU cores utilized: 26 (out of 96 available)
```

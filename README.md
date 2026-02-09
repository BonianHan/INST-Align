# INR-Align

Spatial transcriptomics slice alignment via implicit neural representations (INR).

INR-Align combines ICP rigid alignment with a Nerfies-style deformation network and a canonical expression field (ExprField) for joint nonrigid alignment and batch correction of spatial transcriptomics data.

---

## Table of Contents

- [Package Structure](#package-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Benchmark](#benchmark)
- [Method Overview](#method-overview)
- [Configuration Reference](#configuration-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Format](#data-format)
- [Custom Datasets](#custom-datasets)

---

## Package Structure

```
inr_align/
  __init__.py             # Public API exports
  config.py               # All hyperparameters (dataclasses + auto-generated CLI)
  model.py                # DeformationNet, ExprField, UnifiedCostMatcher, adaptive_icp
  loss.py                 # Matching loss, Jacobian regularization, canonical consistency
  train.py                # Training loop + inference
  metrics.py              # OT accuracy, NN accuracy, ratio
  utils.py                # Grid detection, coordinate normalization
  run.py                  # End-to-end pipeline (single dataset)
  benchmark.py            # Multi-method comparison (6 baselines)
run_test_acc.py           # Benchmark entry point with per-dataset overrides
run_benchmark.sh          # Shell script for full benchmark
requirements.txt          # Python dependencies
```

---

## Installation

### 1. Create conda environment

```bash
conda create -n inr_align python=3.9 -y
conda activate inr_align
```

### 2. Install core dependencies

```bash
pip install torch numpy scipy scikit-learn
pip install anndata scanpy pandas POT matplotlib seaborn numba
```

### 3. Install baseline methods (for benchmark comparison)

```bash
# PASTE
pip install paste-bio

# Spateo (from source)
pip install git+https://github.com/aristoteleo/spateo-release.git

# SPACEL
pip install SPACEL

# STalign
pip install git+https://github.com/JEFworks-Lab/STalign.git
```

### 4. Install INR-Align

```bash
git clone https://github.com/<your-username>/INR-Align.git
cd INR-Align
```

No `setup.py` needed — run scripts directly from the project root.

---

## Quick Start

### Run our method on a single dataset

```bash
python -m inr_align.run --dataset STARMap --use_expr_field
```

### Use as a library

```python
from inr_align import PipelineConfig, run

config = PipelineConfig(dataset="STARMap", data_dir="./Data")
config.use_expr_field = True
config.train.epochs = 200
config.train.lam_jacobian = 0.01

aligned_slices, metrics_df = run(config)
print(metrics_df)
```

---

## Benchmark

Compare 6 methods: **No-align**, **PASTE**, **SPACEL**, **STalign**, **Spateo** (rigid + nonrigid), **Ours** (rigid + spatial).

### Run full benchmark

```bash
bash run_benchmark.sh
```

This runs all methods on 4 datasets (STARMap, BaristaSeq, DLPFC x3, MERFISH_Brain x3) with per-dataset hyperparameter tuning.

### Run with options

```bash
# Our method only (skip baselines)
bash run_benchmark.sh --quick

# Specific dataset
python run_test_acc.py --datasets STARMap --use_expr_field

# Custom hyperparameters
python run_test_acc.py --datasets DLPFC --train_epochs 200 --train_lam_jacobian 0.1
```

### Output

- `benchmark_results.csv` — per-pair results (all methods, 3 metrics)
- `benchmark_summary.csv` — per-dataset and overall summaries
- `benchmark_results.png` — comparison box plot

---

## Method Overview

### Pipeline

```
1. Data Loading       →  AnnData h5ad files per slice
2. Preprocessing      →  HVG selection, joint PCA, coordinate normalization to [0,1]
3. ExprField Pretrain  →  Joint canonical expression field across all slices
4. Per-pair Alignment:
   a. ICP rigid init  →  Centroid alignment + ICP refinement
   b. INR deformation →  Nerfies PE + residual MLP, bidirectional matching
   c. Evaluation      →  OT Accuracy, NN Accuracy, Ratio
```

### DeformationNet

Residual MLP that predicts per-point spatial displacements: `x_out = x + delta(x)`.

- **Input**: 2D coordinates → Nerfies positional encoding (6 frequencies, windowed coarse-to-fine)
- **Architecture**: 6-layer MLP, 128 hidden units, ReLU
- **Output**: 2D displacement, initialized near zero (identity mapping)
- **Grid mode**: Auto-detected; supports snap-to-grid for Visium data

### ExprField (Canonical Expression Field)

Joint expression field with per-slice batch correction.

```
coords → PE → concat(batch_emb) → backbone(256h, 4L) → bottleneck(32d) → head → genes
```

- **Batch embeddings**: 16-dim per slice, initialized at zero
- **Canonical mode**: `batch_emb = 0` yields batch-corrected predictions
- **Bottleneck**: 32-dim latent space usable for downstream clustering
- **Joint pretraining**: 300 epochs on all slices, L2 regularization on batch embeddings

### Loss Functions

1. **Bidirectional matching loss**: Forward (source→target) + reverse (target→source) soft-assignment MSE
2. **Jacobian regularization**: SVD-based `Σ log²(σᵢ)`, penalizes non-isometric deformation
3. **Canonical consistency loss**: MSE between canonical predictions at deformed vs. matched coordinates (ramps from 0 during warmup)

### Matching Strategy

- **UnifiedCostMatcher**: Spatial distance + expression cosine similarity
- **Top-K**: 64 nearest spatial neighbors
- **Temperature**: Adaptive via EM, range [0.01, 0.5]
- **Softmax** normalization (default; Sinkhorn optional)

---

## Configuration Reference

### Best Configuration (per-dataset tuned)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Training epochs |
| `batch_size` | 2500 | Mini-batch size |
| `lr` | 1e-3 | Learning rate |
| `topk` | 64 | Spatial neighbors for matching |
| `weight_rev` | 1.0 | Reverse loss weight |
| `grad_clip` | 2.0 | Gradient clipping |
| `warmup_fraction` | 0.3 | Coarse-to-fine PE warmup |
| `icp.mode` | `icp_only` | No rotation search |
| `use_expr_field` | True | Enable canonical expression field |
| `expr_field.lam_canonical` | 0.005 | Canonical consistency weight |

### Per-Dataset Jacobian Overrides

| Dataset | `lam_jacobian` | `warmup_fraction` | Notes |
|---------|---------------|-------------------|-------|
| DLPFC | 0.1 | 0.3 | Grid data, needs strong smoothness |
| STARMap | 0.01 | 0.4 | Non-grid, moderate regularization |
| BaristaSeq | 0.01 | 0.4 | Non-grid |
| MERFISH | 0.001 | 0.4 | Sparse, flexible deformation |

### Full CLI Arguments

All parameters are auto-generated from config dataclasses:

```bash
python run_test_acc.py --help
```

Naming convention: `--{section}_{parameter}`, e.g. `--train_epochs 200`, `--matcher_tau_init 0.1`, `--model_hidden 128`.

---

## Evaluation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **OT Accuracy** | PASTE-style: `Σ(π * same_label)` where π is the EMD transport plan on aligned coordinates | [0, 1] | Higher |
| **NN Accuracy** | Bidirectional nearest-neighbor label accuracy (iSTBench-style) | [0, 1] | Higher |
| **Ratio** | Collapse measure: `abs(log₂(min(N1,N2) / n_unique_matches))`. Detects if deformation collapses many points to few | [0, ∞) | Lower |

---

## Data Format

### Expected directory structure

```
Data/
  STARMap/
    sample_data/
      slices1.h5ad
      slices2.h5ad
      slices3.h5ad
  DLPFC_sample1/
    original_data/          # Used by benchmark
      151507.h5ad
      151508.h5ad
      151509.h5ad
      151510.h5ad
    sample_data/            # Used by run.py
      slices1.h5ad
      ...
```

### AnnData requirements

Each `.h5ad` file needs:
- `adata.obsm["spatial"]` — (N, 2) spatial coordinates
- `adata.obs["original_domain"]` — cell type labels (for evaluation)
- `adata.X` — gene expression matrix (raw counts)

---

## Custom Datasets

### 1. Prepare h5ad files

Place `.h5ad` slices in `Data/{dataset_name}/sample_data/`.

### 2. Register slice ordering

Add to `inr_align/config.py`:

```python
SLICE_ORDER = {
    ...
    "MyDataset": ["slices1", "slices2", "slices3"],
}
```

### 3. Run

```python
from inr_align import PipelineConfig, run

config = PipelineConfig(dataset="MyDataset", data_dir="./Data")
config.use_expr_field = True
config.label_key = "cell_type"  # if different from "original_domain"
aligned, metrics = run(config)
```

---


## License

MIT License

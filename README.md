# INST-Align

**Implicit Neural Spatial Transcriptomics Alignment**

Spatial transcriptomics slice alignment via implicit neural representations with multi-task learning.

> **MICCAI 2026 Submission**

---

## Table of Contents

- [Overview](#overview)
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

## Overview

INST-Align aligns spatial transcriptomics slices by learning a continuous deformation field via an implicit neural representation (INR). The framework jointly optimizes:

1. **Spatial alignment** -- bidirectional soft matching with adaptive temperature
2. **Isometry regularization** -- Jacobian SVD + divergence anti-compression
3. **Embedding distillation** -- KL divergence from Splane pre-trained embeddings
4. **Gene reconstruction** -- GeneDecoder reconstructs expression from learned embeddings

---

## Package Structure

```
inr_align/
  __init__.py             # Public API exports
  config.py               # All hyperparameters (dataclasses + auto-generated CLI)
  model.py                # DeformationNet, GeneDecoder, ExprField, UnifiedCostMatcher, adaptive_icp
  loss.py                 # Matching loss, Jacobian regularization (SVD + divergence), KL loss
  train.py                # Training loop + inference
  metrics.py              # OT accuracy, NN accuracy, Ratio, CLC
  utils.py                # Grid detection, coordinate normalization, griddata resampling
  run.py                  # End-to-end pipeline (single dataset)
  benchmark.py            # Multi-method comparison (7 baselines)
run_test_acc.py           # Benchmark entry point with per-dataset overrides
benchmark_insta.py        # Quick INSTA-only benchmark with embedding evaluation
spacel_runner.py          # SPACEL subprocess runner
extract_splane_emb.py     # Extract Splane embeddings from SPACEL
requirements.txt          # Python dependencies
```

---

## Installation

### 1. Create conda environment

```bash
conda create -n insta python=3.9 -y
conda activate insta
```

### 2. Install PyTorch (CUDA 12.1 for A100)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install STalign (optional, for benchmark)

```bash
pip install git+https://github.com/JEFworks-Lab/STalign.git
```

### 5. Clone and run

```bash
git clone https://github.com/BonianHan/INST-Align.git
cd INST-Align
```

No `setup.py` needed -- run scripts directly from the project root.

---

## Quick Start

### Run our method on a single dataset

```bash
python -m inr_align.run --dataset STARMap
```

### Use as a library

```python
from inr_align import PipelineConfig, run

config = PipelineConfig(dataset="STARMap", data_dir="./Data")
config.train.epochs = 200
config.train.lam_jacobian = 0.01

aligned_slices, metrics_df = run(config)
print(metrics_df)
```

---

## Benchmark

Compare 8 methods: **No-align**, **PASTE**, **SPACEL**, **STalign**, **Spateo** (rigid + nonrigid), **INSTA** (rigid + nonrigid).

### Run full benchmark

```bash
# All methods on DLPFC
python run_test_acc.py --datasets DLPFC

# INSTA only (skip baselines)
python run_test_acc.py --datasets DLPFC_sample1 --no_paste --no_spateo --no_spacel --no_stalign

# All datasets
python run_test_acc.py --datasets DLPFC STARMap BaristaSeq

# Custom hyperparameters via CLI
python run_test_acc.py --datasets DLPFC --train_epochs 300 --train_lam_jacobian 0.1
```

### Quick INSTA benchmark with embedding evaluation

```bash
python benchmark_insta.py --sample_groups 0
```

### Output

- `benchmark_results.csv` -- per-pair results
- `benchmark_summary.csv` -- per-dataset and overall summaries
- `benchmark_results.png` -- comparison bar chart

---

## Method Overview

### Pipeline

```
1. Data Loading       ->  AnnData h5ad files per slice
2. Preprocessing      ->  HVG selection, joint PCA, coordinate normalization
3. Per-pair Alignment:
   a. ICP rigid init  ->  Centroid + ICP refinement
   b. INR deformation ->  Nerfies PE + residual MLP, multi-task losses
   c. Griddata post   ->  Convex hull resampling for grid data
   d. Evaluation      ->  OT Accuracy, NN Accuracy, Ratio, CLC
```

### DeformationNet

Residual MLP that predicts per-point spatial displacements and optional cell embeddings.

- **Input**: 2D coordinates -> Nerfies positional encoding (6 frequencies, coarse-to-fine windowed)
- **Architecture**: shared backbone (6-layer MLP, 128 hidden, ReLU) + coord_head (2D displacement) + optional emb_head (16-dim embedding)
- **Output**: `x_out = x + delta(x)`, plus optional embedding for downstream tasks
- **Grid mode**: Auto-detected; supports griddata post-processing for Visium data

### GeneDecoder

Reconstructs gene expression from learned cell embeddings + batch embedding.

```
cell_emb (16d) + batch_emb (16d) -> MLP(256h, 2L) -> n_genes
```

### Loss Functions

1. **Bidirectional matching**: Forward (source->target) + reverse (target->source) soft-assignment MSE
2. **Jacobian SVD regularization**: `mean(sum(log(sigma_i)^2))`, penalizes non-isometric deformation
3. **Divergence anti-compression**: `mean(ReLU(-div_delta)^2)` where `div_delta = tr(J_F) - D`, only penalizes compression
4. **Splane KL loss**: KL divergence from DeformationNet embedding head to pre-computed Splane embeddings
5. **Gene reconstruction**: MSE on nonzero + L1 + Dice, reconstructs HVG expression

### Matching Strategy

- **UnifiedCostMatcher**: Spatial distance + expression cosine similarity
- **Top-K**: 64 nearest spatial neighbors
- **Temperature**: Adaptive via EM, range [0.05, 1.0]
- **Softmax** normalization (Sinkhorn optional but not recommended)

---

## Configuration Reference

### Tuned Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 200 | Training epochs |
| `batch_size` | 2500 | Mini-batch size |
| `lr` | 1e-3 | Learning rate |
| `topk` | 64 | Spatial neighbors for matching |
| `weight_rev` | 1.0 | Reverse loss weight |
| `grad_clip` | 1.0 | Gradient clipping |
| `warmup_fraction` | 0.3 | Coarse-to-fine PE warmup |
| `lam_divergence` | 10.0 | Divergence anti-compression weight |
| `lam_kl` | 1.0 | Splane embedding KL weight |
| `lam_recon` | 0.5 | Gene reconstruction weight |
| `emb_dim` | 16 | Embedding head dimension |

### Per-Dataset Overrides

| Dataset | `lam_jacobian` | `warmup_fraction` | `lam_divergence` | Notes |
|---------|---------------|-------------------|-----------------|-------|
| DLPFC | 0.1 | 0.3 | 10.0 | Grid data, strong smoothness |
| STARMap | 0.01 | 0.4 | 0.0 | Non-grid, moderate |
| BaristaSeq | 0.01 | 0.4 | 0.0 | Non-grid |
| MERFISH | 0.001 | 0.4 | 0.0 | Sparse, flexible |

### Full CLI Arguments

All parameters are auto-generated from config dataclasses:

```bash
python run_test_acc.py --help
```

Naming convention: `--{section}_{parameter}`, e.g. `--train_epochs 200`, `--matcher_tau_init 0.1`.

---

## Evaluation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **OT Accuracy** | PASTE-style: transport plan weighted label match | [0, 1] | Higher |
| **NN Accuracy** | Bidirectional nearest-neighbor label accuracy | [0, 1] | Higher |
| **Ratio** | Collapse measure: `abs(log2(min(N1,N2) / n_unique_matches))` | [0, inf) | Lower |
| **CLC** | Contextual Label Consistency: spatial neighborhood label coherence | [0, 1] | Higher |

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
    original_data/
      151507.h5ad
      151508.h5ad
      151509.h5ad
      151510.h5ad
    splane_embeddings.npz   # Pre-computed Splane embeddings (optional)
```

### AnnData requirements

Each `.h5ad` file needs:
- `adata.obsm["spatial"]` -- (N, 2) spatial coordinates
- `adata.obs["original_domain"]` -- cell type / domain labels (for evaluation)
- `adata.X` -- gene expression matrix (raw counts preferred)

### Splane embeddings (optional)

Pre-computed Splane embeddings enable the KL + gene reconstruction losses:

```bash
# Extract Splane embeddings (requires SPACEL installed)
python extract_splane_emb.py
```

This saves `splane_embeddings.npz` with keys `emb_{sample_id}` per slice.

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
config.label_key = "cell_type"  # if different from "original_domain"
aligned, metrics = run(config)
```

---

## License

MIT License

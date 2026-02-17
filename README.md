# INST-Align

**Implicit Neural Spatial Transcriptomics Alignment**

Spatial transcriptomics slice alignment via implicit neural representations with multi-task learning.

> **MICCAI 2026 Submission**


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


### 1. Install package

```bash
pip install -r requirements.txt
```

### 2. Install STalign (optional, for benchmark)

```bash
pip install git+https://github.com/JEFworks-Lab/STalign.git
```

### 3. Clone and run

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


## License

MIT License

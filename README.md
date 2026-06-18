# INST-Align

**[Implicit Neural Spatial Transcriptomics Alignment](https://arxiv.org/pdf/2604.12084)**

Spatial transcriptomics slice alignment via implicit neural representations with multi-task learning.

> **MICCAI 2026 Accepted**


## Code Structure

```
insta/
  __init__.py
  config.py               # All hyperparameters (dataclasses + auto-generated CLI)
  model.py                # DeformationNet, ExprINR, ExprDecoder, UnifiedCostMatcher, adaptive_icp
  loss.py                 # Spatial-expression matching, Jacobian regularization, reconstruction loss
  trainer.py              # Training loop + inference
  metrics.py              # Paper metrics and metric helpers
  utils.py                # Coordinate, transport-plan, and preprocessing helpers
  pipeline.py             # Single-dataset INST-Align pipeline
  spatial_alignment.py    # Shared logic for the spatial alignment experiment
run_spatial_alignment.py  # Table 1: spatial alignment results
run_embedding.py          # Table 2: embedding evaluation
run_ablation_insta.py     # Table 3: INST-Align ablation study
data/README.md            # Data source and expected layout
requirements.txt          # Python dependencies
```

---

## Installation


### 1. Install package

```bash
pip install -r requirements.txt
```

### 2. Install STalign (optional baseline)

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

## Data

Download the prepared `Data.zip` archive from Zenodo:

```bash
wget -O Data.zip "https://zenodo.org/records/14906156/files/Data.zip?download=1"
md5sum Data.zip
unzip -q Data.zip
```

Expected checksum:

```text
ad37e0fac9691d23f9f91eaee28842ed  Data.zip
```

See `data/README.md` for dataset details and citation.

---

## Quick Start

### Run our method on a single dataset

```bash
python -m insta --dataset STARMap
```

### Use as a library

```python
from insta import PipelineConfig, run

config = PipelineConfig(dataset="STARMap", data_dir="./Data")
config.train.epochs = 200
config.train.lam_jacobian = 0.01

aligned_slices, metrics_df = run(config)
print(metrics_df)
```

---

## Reproducing Paper Experiments

### Spatial Alignment Results

```bash
# Run INST-Align and baselines
python run_spatial_alignment.py --datasets DLPFC

# Run only INST-Align
python run_spatial_alignment.py --methods insta --datasets DLPFC_sample1

# Run only baselines
python run_spatial_alignment.py --methods baseline --datasets DLPFC_sample1

# Custom hyperparameters via CLI
python run_spatial_alignment.py --datasets DLPFC --train_epochs 300
```

### Embedding Evaluation

```bash
python run_embedding.py --sample_groups 0 1 2 --run_embryo
```

### Ablation Study

```bash
python run_ablation_insta.py --dataset DLPFC_sample1
```

---


## License

MIT License

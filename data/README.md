# Data

Large datasets are not stored in this GitHub repository.

The experiment code expects a project-level `Data/` directory. We use the
prepared public data archive from Zenodo:

- Zenodo record: https://zenodo.org/records/14906156
- DOI: https://doi.org/10.5281/zenodo.14906156
- File: `Data.zip`
- MD5: `ad37e0fac9691d23f9f91eaee28842ed`
- License: Creative Commons Attribution 4.0 International

This archive is associated with:

Dong K, Gao Y, Zou Q, Cui Y, Han C, Lin S, Wang Z, Tang C, Cheng X, Meng F,
Chen X, Wang S, Jin X, Yang J, Zhang C, Chuai G, Yuan Z, Liu Q.
Benchmarking multi-slice integration and downstream applications in spatial
transcriptomics data analysis. Genome Biology. 2025;26:318.
doi: 10.1186/s13059-025-03796-z.

## Download

From the project root:

```bash
wget -O Data.zip "https://zenodo.org/records/14906156/files/Data.zip?download=1"
md5sum Data.zip
unzip -q Data.zip
```

The checksum should be:

```text
ad37e0fac9691d23f9f91eaee28842ed  Data.zip
```

## Paper Datasets

The paper experiments use the following datasets:

| Dataset name in code | Local data used by scripts | Notes |
| --- | --- | --- |
| `DLPFC` / `DLPFC_sample1` | `Data/DLPFC_sample1/original_data/151507.h5ad` to `151510.h5ad` | DLPFC sample group 1 |
| `DLPFC_sample2` | `Data/DLPFC_sample2/original_data/151669.h5ad` to `151672.h5ad` | DLPFC sample group 2 |
| `DLPFC_sample3` | `Data/DLPFC_sample3/original_data/151673.h5ad` to `151676.h5ad` | DLPFC sample group 3 |
| `STARMap` | `Data/STARMap/original_data/*.h5ad` | Converted to three `sample_data/slices*.h5ad` files |
| `MERFISH` | `Data/MERFISH/original_data/*.h5ad` | MERFISH hypothalamus dataset |
| `MERFISH_Brain_S2` | `Data/MERFISH_Brain_S2/sample_data/*.h5ad` | Prepared slice files only in current local project |
| `MERFISH_Brain_S7` | `Data/MERFISH_Brain_S7/sample_data/*.h5ad` | Prepared slice files only in current local project |
| `MERFISH_Brain_S11` | `Data/MERFISH_Brain_S11/sample_data/*.h5ad` | Prepared slice files only in current local project |
| `MouseEmbryo` | `Data/MouseEmbryo/sample_data/slices1.h5ad`, `slices2.h5ad` | Prepared slice files only in current local project |

## Expected Layout

After unzipping, the project root should contain:

```text
run_spatial_alignment.py
run_embedding.py
run_ablation_insta.py
Data/
  DLPFC_sample1/
    original_data/
    sample_data/
  DLPFC_sample2/
    original_data/
    sample_data/
  DLPFC_sample3/
    original_data/
    sample_data/
  STARMap/
    original_data/
    sample_data/
  MERFISH/
    original_data/
    sample_data/
  MERFISH_Brain_S2/
    sample_data/
  MERFISH_Brain_S7/
    sample_data/
  MERFISH_Brain_S11/
    sample_data/
  MouseEmbryo/
    sample_data/
```

## File Format

Input slices are `.h5ad` files with:

- `obsm["spatial"]`: spatial coordinates, shape `(n_obs, 2)`
- `obs["original_domain"]` for DLPFC, STARMap, MERFISH, and MERFISH Brain
- `obs["cellbin_SpatialDomain"]` for MouseEmbryo
- expression matrix `X`

## Local Processing Notes

From the local workspace:

- `download.ipynb` and `download copy.ipynb` unzip `Data.zip`.
- `Data/*/1_data_process.py` scripts read existing local `original_data/`
  files and write `sample_data/`.
- The local `Data.zip` checksum matches the Zenodo archive checksum above.

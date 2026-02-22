#!/usr/bin/env python
"""Run all alignment methods on all datasets and compare accuracy.

Command-line usage::

    python run_test_acc.py --datasets DLPFC STARMap BaristaSeq
    python run_test_acc.py --datasets DLPFC --use_expr_field
    python run_test_acc.py --no_paste --no_spateo

Library usage::

    from run_test_acc import main
    from inr_align.config import PipelineConfig

    config = PipelineConfig(data_dir="./Data")
    config.train.epochs = 300
    main(config, datasets=["STARMap", "DLPFC_sample1"])

This script iterates over all datasets in SLICE_ORDER (or a user-specified
subset), runs methods on each consecutive pair, and saves:

    1. benchmark_results.csv       -- per-pair accuracy for all datasets
    2. benchmark_summary.csv       -- per-dataset mean +/- std
    3. benchmark_results.png       -- box plot

Metrics:
    - Accuracy (OT)  -- PASTE-style transport-plan weighted label match
    - Accuracy_NN    -- Bidirectional nearest-neighbour label match (iSTBench)

Methods compared:
    1. No-align   -- raw coordinates, no alignment
    2. PASTE      -- Fused Gromov-Wasserstein (alpha=0.1)
    3. PASTE2     -- Partial pairwise alignment (KL divergence)
    4. GPSA       -- Gaussian Process Spatial Alignment
    5. STalign    -- LDDMM diffeomorphic registration
    6. Spateo     -- morpho_align rigid + nonrigid
    7. Ours       -- adaptive_icp + INR deformation (rigid / spatial)
"""

from __future__ import annotations

import argparse
import copy
import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from inr_align.benchmark import (
    _load_dlpfc_layer_groups,
    benchmark_all,
    benchmark_dataset,
    print_summary,
    DLPFC_LABEL_MAP,
)
from inr_align.config import SLICE_ORDER, PipelineConfig

warnings.filterwarnings("ignore")


# All datasets available for benchmarking
ALL_DATASETS = list(SLICE_ORDER.keys())

# ---------------------------------------------------------------------------
# Per-dataset hyperparameter overrides
#
# Grid-based data (DLPFC): stronger Jacobian regularization, default warmup
# Scattered point clouds (STARMap, BaristaSeq, MERFISH): weaker Jacobian,
# later warmup so coarse-to-fine PE has more time to ramp up
# ---------------------------------------------------------------------------
DATASET_OVERRIDES = {
    # DLPFC (grid): stronger Jacobian to prevent nonrigid collapse
    "DLPFC": {"lam_jacobian": 0.3, "warmup_fraction": 0.3},
    # Non-grid datasets: weaker Jacobian, delayed warmup
    "STARMap": {"lam_jacobian": 0.01, "warmup_fraction": 0.4},
    "BaristaSeq": {"lam_jacobian": 0.01, "warmup_fraction": 0.4},
    "MERFISH": {"lam_jacobian": 0.001, "warmup_fraction": 0.4},
    # MERFISH_Brain S2/S3/S8/S12 (non-grid, ~5k-10k cells):
    # Key findings: icp_only safer than PCA (avoids wrong 180° rotations),
    # lr=1e-3 better than 3e-4 for these smaller datasets,
    # same nonrigid recipe as MouseEmbryo (no jac, no warmup, frozen INR)
    "MERFISH_Brain": {
        "epochs": 300, "batch_size": 4000, "lr": 1e-3, "grad_clip": 2.0,
        "lam_jacobian": 0.0, "lam_deform_mag": 0.0, "lam_uniqueness": 0.0,
        "tau_min": 0.005,
        # icp_only mode (default) — PCA mode chose wrong 180° on S2
        "inr_pretrain_epochs": 150,
        "freeze_inr_phase2": True, "lam_recon_phase2": 0.0,
        "warmup_fraction": 0.0,
        "scheduler_patience": 9999,
    },
    # Mouse Embryo E9.5 (non-grid, large cells ~17k-20k): notebook-tuned config
    "Mouse": {
        "epochs": 150, "batch_size": 2000, "grad_clip": 2.0,
        "lam_jacobian": 0.005, "tau_min": 0.005,
        "mode": "pca",  # ICP: rotation search for Mouse
    },
    # MouseEmbryo (non-grid, 17k-20k cells, ~180° rotation needed):
    # Key findings from tuning rounds 1-8:
    # - warmup_fraction=0 critical (full PE from start enables fast deformation)
    # - No Jacobian reg (allows flexible deformation matching Spateo quality)
    # - lr=3e-4 (prevents deformation overshoot vs default 1e-3)
    # - Freeze INR in Phase 2 (prevents embedding drift)
    # - Scheduler effectively disabled (match loss fluctuates during deformation)
    "MouseEmbryo": {
        "epochs": 500, "batch_size": 4000, "lr": 3e-4, "grad_clip": 2.0,
        "lam_jacobian": 0.0, "lam_deform_mag": 0.0, "lam_uniqueness": 0.0,
        "tau_min": 0.005, "mode": "pca",
        "inr_pretrain_epochs": 150,
        "freeze_inr_phase2": True, "lam_recon_phase2": 0.0,
        "warmup_fraction": 0.0,
        "scheduler_patience": 9999,
    },
}


def _config_for_dataset(config: PipelineConfig, dataset: str) -> PipelineConfig:
    """Return a (possibly modified) copy of *config* with dataset-specific overrides."""
    # Match by longest prefix: MERFISH_Brain_S2 -> "MERFISH_Brain" (not "MERFISH")
    key = None
    for prefix in DATASET_OVERRIDES:
        if dataset.startswith(prefix):
            if key is None or len(prefix) > len(key):
                key = prefix
    if key is None:
        return config  # no override needed

    overrides = DATASET_OVERRIDES[key]
    cfg = copy.deepcopy(config)
    for field_name, value in overrides.items():
        if hasattr(cfg.train, field_name):
            setattr(cfg.train, field_name, value)
        elif hasattr(cfg.joint, field_name):
            setattr(cfg.joint, field_name, value)
        elif hasattr(cfg.matcher, field_name):
            setattr(cfg.matcher, field_name, value)
        elif hasattr(cfg.icp, field_name):
            setattr(cfg.icp, field_name, value)
    parts = [f"{k}={v}" for k, v in overrides.items()]
    print(f"  [dataset override] {dataset}: {', '.join(parts)}")
    return cfg


def save_results(df_all: pd.DataFrame, output_dir: str = ".", output_name: str = "benchmark_results") -> None:
    """Save per-pair CSV, summary table CSV, and print report."""

    # ---- 1. Save raw per-pair results ----
    csv_path = os.path.join(output_dir, f"{output_name}.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n✅ Per-pair results saved to {csv_path}")

    # ---- 2. Build the summary table ----
    # Method renaming & filtering
    METHOD_RENAME = {
        "No-align": "Raw",
        "PASTE": "PASTE",
        "PASTE2": "PASTE2",
        "GPSA": "GPSA",
        "STalign": "STalign",
        "Spateo_Nonrigid": "Spateo",
        "INSTA-Rigid": "Ours-Rigid",
        "INSTA-Nonrigid": "Ours-Nonrigid",
    }
    METHOD_ORDER = ["Raw", "PASTE", "PASTE2", "GPSA", "STalign", "Spateo",
                    "Ours-Rigid", "Ours-Nonrigid"]
    METRIC_COLS = ["Time", "Accuracy", "Accuracy_NN", "Ratio", "CLC",
                   "LISI", "Silhouette", "Chamfer"]
    METRIC_HEADERS = ["Time(s)", "OT_Acc", "NN_Acc", "Ratio", "CLC",
                      "LISI", "Silhouette", "Chamfer"]

    # Filter & rename
    df = df_all[df_all["Method"].isin(METHOD_RENAME)].copy()
    df["Method"] = df["Method"].map(METHOD_RENAME)

    # Compute spot counts per dataset (from first pair's first method)
    spot_info = {}
    for dataset in df["Dataset"].unique():
        df_ds = df[df["Dataset"] == dataset]
        if "N1" in df_ds.columns and "N2" in df_ds.columns:
            # Sum unique spots across all pairs
            total_spots = set()
            for _, row in df_ds[df_ds["Method"] == "Raw"].iterrows():
                total_spots.add(int(row.get("N1", 0)))
                total_spots.add(int(row.get("N2", 0)))
            # Just show range: min~max per slice
            n1_vals = df_ds[df_ds["Method"] == "Raw"]["N1"].unique()
            n2_vals = df_ds[df_ds["Method"] == "Raw"]["N2"].unique()
            all_n = sorted(set(n1_vals) | set(n2_vals))
            if len(all_n) == 1:
                spot_info[dataset] = f"{int(all_n[0])} spots/slice"
            else:
                spot_info[dataset] = f"{int(min(all_n))}~{int(max(all_n))} spots/slice"
        else:
            spot_info[dataset] = ""

    # Number of pairs per dataset
    n_pairs_info = {}
    for dataset in df["Dataset"].unique():
        df_ds = df[(df["Dataset"] == dataset) & (df["Method"] == "Raw")]
        n_pairs_info[dataset] = len(df_ds)

    # Build summary rows: per dataset, per method → mean across pairs
    summary_rows = []
    datasets_ordered = list(df["Dataset"].unique())

    for dataset in datasets_ordered:
        df_ds = df[df["Dataset"] == dataset]
        n_pairs = n_pairs_info[dataset]
        for method in METHOD_ORDER:
            df_m = df_ds[df_ds["Method"] == method]
            if len(df_m) == 0:
                continue
            row = {"Dataset": dataset, "Method": method}
            for col in METRIC_COLS:
                if col in df_m.columns:
                    vals = df_m[col].dropna()
                    if len(vals) > 0:
                        row[col] = vals.mean()
                    else:
                        row[col] = np.nan
                else:
                    row[col] = np.nan
            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # ---- 3. Save summary CSV (flat, easy to import into Excel) ----
    summary_path = os.path.join(output_dir, f"{output_name}_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to {summary_path}")

    # ---- 4. Print formatted table ----
    # Column widths
    W_DS = 18
    W_INFO = 22
    W_METHOD = 14
    W_NUM = 10

    metric_cols_present = [c for c in METRIC_COLS if c in df_summary.columns]
    metric_headers = [METRIC_HEADERS[METRIC_COLS.index(c)] for c in metric_cols_present]

    # Header
    header = (f"{'Dataset':<{W_DS}} {'Info':<{W_INFO}} {'Method':<{W_METHOD}} "
              + " ".join(f"{h:>{W_NUM}}" for h in metric_headers))
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for dataset in datasets_ordered:
        df_ds = df_summary[df_summary["Dataset"] == dataset]
        n_pairs = n_pairs_info[dataset]
        info_str = f"{spot_info[dataset]}, {n_pairs}p"

        for idx, (_, row) in enumerate(df_ds.iterrows()):
            ds_col = dataset if idx == 0 else ""
            info_col = info_str if idx == 0 else ""
            method = row["Method"]
            vals = []
            for col in metric_cols_present:
                v = row.get(col, np.nan)
                if pd.isna(v):
                    vals.append(f"{'—':>{W_NUM}}")
                elif col == "Time":
                    vals.append(f"{v:>{W_NUM}.1f}")
                else:
                    vals.append(f"{v:>{W_NUM}.4f}")
            line = (f"{ds_col:<{W_DS}} {info_col:<{W_INFO}} {method:<{W_METHOD}} "
                    + " ".join(vals))
            print(line)
        print(sep)

    # ---- 5. Group summaries (DLPFC_ALL etc.) ----
    group_defs = [
        ("DLPFC_", "DLPFC_ALL"),
        ("MERFISH_Brain_", "MERFISH_Brain_ALL"),
    ]
    for prefix, group_label in group_defs:
        group_ds = [d for d in datasets_ordered if d.startswith(prefix)]
        if len(group_ds) <= 1:
            continue
        df_grp = df[df["Dataset"].isin(group_ds)]
        print(f"\n{'=' * len(header)}")
        print(f"{group_label} (mean across {len(group_ds)} samples)")
        print(f"{'=' * len(header)}")
        for method in METHOD_ORDER:
            df_m = df_grp[df_grp["Method"] == method]
            if len(df_m) == 0:
                continue
            vals = []
            for col in metric_cols_present:
                v = df_m[col].mean() if col in df_m.columns else np.nan
                if pd.isna(v):
                    vals.append(f"{'—':>{W_NUM}}")
                elif col == "Time":
                    vals.append(f"{v:>{W_NUM}.1f}")
                else:
                    vals.append(f"{v:>{W_NUM}.4f}")
            line = (f"{'':<{W_DS}} {'':<{W_INFO}} {method:<{W_METHOD}} "
                    + " ".join(vals))
            print(line)



def run_dlpfc_benchmark(
    config: PipelineConfig,
    device: str,
    run_paste: bool,
    run_spateo: bool,
    run_stalign: bool = True,
) -> pd.DataFrame:
    """Run DLPFC benchmark using original_data format (3 sample groups)."""
    print("\n" + "#" * 60)
    print("# DLPFC (3 sample groups x 3 pairs = 9 pairs)")
    print("#" * 60)

    layer_groups = _load_dlpfc_layer_groups(config.data_dir)
    df, *_ = benchmark_all(
        layer_groups, config, device=device,
        label_key="original_domain", label_map=DLPFC_LABEL_MAP,
        run_paste=run_paste, run_spateo=run_spateo,
        run_stalign=run_stalign,
    )

    # Map Sample index → DLPFC real sample IDs
    df["Dataset"] = df["Sample"].map({0: "DLPFC_151507", 1: "DLPFC_151669", 2: "DLPFC_151673"})
    df = df.drop(columns=["Sample"])
    return df


def main(
    config: Optional[PipelineConfig] = None,
    datasets: Optional[List[str]] = None,
    run_paste: bool = True,
    run_paste2: bool = True,
    run_gpsa: bool = True,
    run_spateo: bool = True,
    run_stalign: bool = True,
    output_name: str = "benchmark_results",
) -> None:
    """Run benchmark on all (or specified) datasets.

    Args:
        config: Pipeline config. Defaults to PipelineConfig().
        datasets: List of dataset names, or None for all.
            Use "DLPFC" shorthand to include all 3 DLPFC samples.
        run_paste: Include PASTE baseline.
        run_paste2: Include PASTE2 baseline.
        run_gpsa: Include GPSA baseline.
        run_spateo: Include Spateo baseline.
        run_stalign: Include STalign baseline.
    """
    if config is None:
        config = PipelineConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve datasets
    if datasets is None:
        datasets = ALL_DATASETS
    else:
        expanded = []
        for d in datasets:
            if d == "DLPFC":
                expanded.extend(["DLPFC_151507", "DLPFC_151669", "DLPFC_151673"])
            else:
                expanded.append(d)
        datasets = expanded

    # Separate DLPFC (uses original_data) from others (uses sample_data)
    dlpfc_datasets = [d for d in datasets if d.startswith("DLPFC_")]
    other_datasets = [d for d in datasets if not d.startswith("DLPFC_")]

    all_dfs = []

    # ---- DLPFC: use original_data format with DLPFC_LABEL_MAP ----
    if dlpfc_datasets:
        from inr_align.config import DLPFC_SAMPLE_GROUPS
        # Map DLPFC_151507 -> index 0, DLPFC_151669 -> index 1, DLPFC_151673 -> index 2
        _DLPFC_ID_TO_IDX = {"151507": 0, "151669": 1, "151673": 2}
        sample_indices = []
        for d in dlpfc_datasets:
            sid = d.replace("DLPFC_", "")
            if sid in _DLPFC_ID_TO_IDX:
                sample_indices.append(_DLPFC_ID_TO_IDX[sid])

        if sample_indices:
            print("\n" + "#" * 60)
            print(f"# DLPFC ({len(sample_indices)} sample groups)")
            print("#" * 60)

            from inr_align.config import DLPFC_SAMPLE_GROUPS as ALL_GROUPS
            layer_groups = []
            sample_id_groups = []
            dataset_folders = []
            for idx in sample_indices:
                group = ALL_GROUPS[idx]
                slices = []
                _IDX_TO_FOLDER = {0: "DLPFC_sample1", 1: "DLPFC_sample2", 2: "DLPFC_sample3"}
                folder = _IDX_TO_FOLDER[idx]
                for sample_id in group:
                    import scanpy as sc
                    path = f"{config.data_dir}/{folder}/original_data/{sample_id}.h5ad"
                    adata = sc.read_h5ad(path)
                    slices.append(adata)
                    print(f"  Loaded {sample_id}: {adata.shape}")
                layer_groups.append(slices)
                sample_id_groups.append(list(group))
                dataset_folders.append(folder)

            dlpfc_config = _config_for_dataset(config, "DLPFC")
            df_dlpfc, *_ = benchmark_all(
                layer_groups, dlpfc_config, device=device,
                label_key="original_domain", label_map=DLPFC_LABEL_MAP,
                run_paste=run_paste, run_paste2=run_paste2,
                run_gpsa=run_gpsa, run_spateo=run_spateo,
                run_stalign=run_stalign,
                sample_id_groups=sample_id_groups,
                dataset_folders=dataset_folders,
            )
            # Map Sample index → dataset name
            _IDX_TO_ID = {0: "151507", 1: "151669", 2: "151673"}
            idx_to_name = {i: f"DLPFC_{_IDX_TO_ID[sample_indices[i]]}" for i in range(len(sample_indices))}
            df_dlpfc["Dataset"] = df_dlpfc["Sample"].map(idx_to_name)
            df_dlpfc = df_dlpfc.drop(columns=["Sample"])
            all_dfs.append(df_dlpfc)

    # ---- Other datasets: use sample_data format ----
    for dataset in other_datasets:
        print(f"\n{'#' * 60}")
        print(f"# {dataset}")
        print(f"{'#' * 60}")

        ds_config = _config_for_dataset(config, dataset)
        df = benchmark_dataset(
            dataset, ds_config, device=device,
            label_key="original_domain",
            run_paste=run_paste, run_paste2=run_paste2,
            run_gpsa=run_gpsa, run_spateo=run_spateo,
            run_stalign=run_stalign,
        )
        if len(df) > 0:
            all_dfs.append(df)

    # ---- Combine and save ----
    if not all_dfs:
        print("\nNo results to save.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    save_results(df_all, output_name=output_name)


if __name__ == "__main__":
    from inr_align.config import add_pipeline_args, config_from_args, print_config

    parser = argparse.ArgumentParser(
        description="Benchmark alignment methods on spatial transcriptomics datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Benchmark-specific args
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to benchmark (default: all). Use 'DLPFC' for all 3 samples.",
    )
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output file base name (default: benchmark_results)")
    parser.add_argument("--no_paste", action="store_true", help="Skip PASTE baseline")
    parser.add_argument("--no_paste2", action="store_true", help="Skip PASTE2 baseline")
    parser.add_argument("--no_gpsa", action="store_true", help="Skip GPSA baseline")
    parser.add_argument("--no_spateo", action="store_true", help="Skip Spateo baseline")
    parser.add_argument("--no_stalign", action="store_true", help="Skip STalign baseline")
    # All pipeline config args
    add_pipeline_args(parser)

    args = parser.parse_args()
    config = config_from_args(args)
    print_config(config)

    main(
        config=config,
        datasets=args.datasets,
        run_paste=not args.no_paste,
        run_paste2=not args.no_paste2,
        run_gpsa=not args.no_gpsa,
        run_spateo=not args.no_spateo,
        run_stalign=not args.no_stalign,
        output_name=args.output,
    )

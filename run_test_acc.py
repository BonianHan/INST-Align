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
    3. SPACEL     -- Scube.align (graph-based)
    4. STalign    -- LDDMM diffeomorphic registration
    5. Spateo     -- morpho_align rigid + nonrigid
    6. Ours       -- adaptive_icp + INR deformation (rigid / spatial)
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
    plot_comparison,
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
    # DLPFC (grid): strong Jacobian, CPD outlier matching
    "DLPFC": {"lam_jacobian": 0.1, "warmup_fraction": 0.3, "outlier_weight": 0.3},
    # Non-grid datasets: weaker Jacobian, delayed warmup
    "STARMap": {"lam_jacobian": 0.01, "warmup_fraction": 0.4},
    "BaristaSeq": {"lam_jacobian": 0.01, "warmup_fraction": 0.4},
    "MERFISH": {"lam_jacobian": 0.001, "warmup_fraction": 0.4},
}


def _config_for_dataset(config: PipelineConfig, dataset: str) -> PipelineConfig:
    """Return a (possibly modified) copy of *config* with dataset-specific overrides."""
    # Match by prefix: DLPFC_sample* -> "DLPFC", MERFISH_Brain_* -> "MERFISH"
    key = None
    for prefix in DATASET_OVERRIDES:
        if dataset.startswith(prefix):
            key = prefix
            break
    if key is None:
        return config  # no override needed

    overrides = DATASET_OVERRIDES[key]
    cfg = copy.deepcopy(config)
    for field_name, value in overrides.items():
        if hasattr(cfg.train, field_name):
            setattr(cfg.train, field_name, value)
        elif hasattr(cfg.matcher, field_name):
            setattr(cfg.matcher, field_name, value)
    parts = [f"{k}={v}" for k, v in overrides.items()]
    print(f"  [dataset override] {dataset}: {', '.join(parts)}")
    return cfg


def save_results(df_all: pd.DataFrame, output_dir: str = ".") -> None:
    """Save per-pair CSV, summary CSV, and plot."""

    has_nn = "Accuracy_NN" in df_all.columns
    has_ratio = "Ratio" in df_all.columns
    has_clc = "CLC" in df_all.columns

    # ---- Per-pair results ----
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n✅ Per-pair results saved to {csv_path}")

    # ---- Per-dataset summary (mean ± std) ----
    summary_rows = []
    methods = df_all["Method"].unique()

    # Metrics to summarize
    metric_cols = [("Accuracy", "OT")]
    if has_nn:
        metric_cols.append(("Accuracy_NN", "NN"))
    if has_ratio:
        metric_cols.append(("Ratio", "Ratio"))
    if has_clc:
        metric_cols.append(("CLC", "CLC"))

    def _add_method_stats(row, df_sub, methods, metric_col, suffix):
        """Add mean/std for each method to row dict."""
        for m in methods:
            vals = df_sub[df_sub["Method"] == m][metric_col]
            if len(vals) > 0:
                row[f"{m}_{suffix}_mean"] = vals.mean()
                row[f"{m}_{suffix}_std"] = vals.std()
            else:
                row[f"{m}_{suffix}_mean"] = np.nan
                row[f"{m}_{suffix}_std"] = np.nan

    for dataset in df_all["Dataset"].unique():
        row = {"Dataset": dataset}
        df_ds = df_all[df_all["Dataset"] == dataset]
        n_pairs = len(df_ds[df_ds["Method"] == df_ds["Method"].iloc[0]])
        row["N_pairs"] = n_pairs
        for metric_col, suffix in metric_cols:
            _add_method_stats(row, df_ds, methods, metric_col, suffix)
        summary_rows.append(row)

    # ---- Group-level summaries (DLPFC_ALL, MERFISH_Brain_ALL, etc.) ----
    all_datasets = df_all["Dataset"].unique()

    group_defs = [
        ("DLPFC_sample", "DLPFC_ALL"),
        ("MERFISH_Brain_", "MERFISH_Brain_ALL"),
    ]
    for prefix, group_label in group_defs:
        group_ds = [d for d in all_datasets if d.startswith(prefix)]
        if len(group_ds) > 1:
            df_grp = df_all[df_all["Dataset"].isin(group_ds)]
            n_pairs = len(df_grp[df_grp["Method"] == df_grp["Method"].iloc[0]])
            row = {"Dataset": group_label, "N_pairs": n_pairs}
            for metric_col, suffix in metric_cols:
                _add_method_stats(row, df_grp, methods, metric_col, suffix)
            summary_rows.append(row)

    # Add overall row
    overall = {"Dataset": "OVERALL", "N_pairs": len(df_all[df_all["Method"] == methods[0]])}
    for metric_col, suffix in metric_cols:
        _add_method_stats(overall, df_all, methods, metric_col, suffix)
    summary_rows.append(overall)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to {summary_path}")

    # ---- Print summary ----
    for metric_col, suffix in metric_cols:
        print("\n" + "=" * 80)
        print(f"=== Per-Dataset Summary ({suffix} Accuracy) ===")
        print("=" * 80)
        for _, row in df_summary.iterrows():
            print(f"\n  {row['Dataset']} ({int(row['N_pairs'])} pairs):")
            for m in methods:
                mean_col = f"{m}_{suffix}_mean"
                std_col = f"{m}_{suffix}_std"
                if mean_col in row and not np.isnan(row[mean_col]):
                    print(f"    {m:20s}: {row[mean_col]:.4f} ± {row[std_col]:.4f}")

    # ---- Plot ----
    png_path = os.path.join(output_dir, "benchmark_results.png")
    try:
        plot_comparison(df_all, png_path)
    except Exception as e:
        print(f"  Plot failed: {e}")


def run_dlpfc_benchmark(
    config: PipelineConfig,
    device: str,
    run_paste: bool,
    run_spateo: bool,
    run_spacel: bool = True,
    run_stalign: bool = True,
) -> pd.DataFrame:
    """Run DLPFC benchmark using original_data format (3 sample groups)."""
    print("\n" + "#" * 60)
    print("# DLPFC (3 sample groups x 3 pairs = 9 pairs)")
    print("#" * 60)

    layer_groups = _load_dlpfc_layer_groups(config.data_dir)
    df = benchmark_all(
        layer_groups, config, device=device,
        label_key="original_domain", label_map=DLPFC_LABEL_MAP,
        run_paste=run_paste, run_spateo=run_spateo,
        run_spacel=run_spacel, run_stalign=run_stalign,
    )

    # Map Sample index → DLPFC_sampleN, keep Pair
    df["Dataset"] = df["Sample"].map({0: "DLPFC_sample1", 1: "DLPFC_sample2", 2: "DLPFC_sample3"})
    df = df.drop(columns=["Sample"])
    return df


def main(
    config: Optional[PipelineConfig] = None,
    datasets: Optional[List[str]] = None,
    run_paste: bool = True,
    run_spateo: bool = True,
    run_spacel: bool = True,
    run_stalign: bool = True,
) -> None:
    """Run benchmark on all (or specified) datasets.

    Args:
        config: Pipeline config. Defaults to PipelineConfig().
        datasets: List of dataset names, or None for all.
            Use "DLPFC" shorthand to include all 3 DLPFC samples.
        run_paste: Include PASTE baseline.
        run_spateo: Include Spateo baseline.
        run_spacel: Include SPACEL baseline.
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
                expanded.extend(["DLPFC_sample1", "DLPFC_sample2", "DLPFC_sample3"])
            else:
                expanded.append(d)
        datasets = expanded

    # Separate DLPFC (uses original_data) from others (uses sample_data)
    dlpfc_datasets = [d for d in datasets if d.startswith("DLPFC_sample")]
    other_datasets = [d for d in datasets if not d.startswith("DLPFC_sample")]

    all_dfs = []

    # ---- DLPFC: use original_data format with DLPFC_LABEL_MAP ----
    if dlpfc_datasets:
        from inr_align.config import DLPFC_SAMPLE_GROUPS
        sample_indices = []
        for d in dlpfc_datasets:
            idx = int(d.replace("DLPFC_sample", "")) - 1
            if 0 <= idx < len(DLPFC_SAMPLE_GROUPS):
                sample_indices.append(idx)

        if sample_indices:
            print("\n" + "#" * 60)
            print(f"# DLPFC ({len(sample_indices)} sample groups)")
            print("#" * 60)

            from inr_align.config import DLPFC_SAMPLE_GROUPS as ALL_GROUPS
            layer_groups = []
            for idx in sample_indices:
                group = ALL_GROUPS[idx]
                slices = []
                folder = f"DLPFC_sample{idx + 1}"
                for sample_id in group:
                    import scanpy as sc
                    path = f"{config.data_dir}/{folder}/original_data/{sample_id}.h5ad"
                    adata = sc.read_h5ad(path)
                    slices.append(adata)
                    print(f"  Loaded {sample_id}: {adata.shape}")
                layer_groups.append(slices)

            dlpfc_config = _config_for_dataset(config, "DLPFC")
            df_dlpfc = benchmark_all(
                layer_groups, dlpfc_config, device=device,
                label_key="original_domain", label_map=DLPFC_LABEL_MAP,
                run_paste=run_paste, run_spateo=run_spateo,
                run_spacel=run_spacel, run_stalign=run_stalign,
            )
            # Map Sample index → dataset name
            idx_to_name = {i: f"DLPFC_sample{sample_indices[i] + 1}" for i in range(len(sample_indices))}
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
            run_paste=run_paste, run_spateo=run_spateo,
            run_spacel=run_spacel, run_stalign=run_stalign,
        )
        if len(df) > 0:
            all_dfs.append(df)

    # ---- Combine and save ----
    if not all_dfs:
        print("\nNo results to save.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    save_results(df_all)


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
    parser.add_argument("--no_paste", action="store_true", help="Skip PASTE baseline")
    parser.add_argument("--no_spateo", action="store_true", help="Skip Spateo baseline")
    parser.add_argument("--no_spacel", action="store_true", help="Skip SPACEL baseline")
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
        run_spateo=not args.no_spateo,
        run_spacel=not args.no_spacel,
        run_stalign=not args.no_stalign,
    )

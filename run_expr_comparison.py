#!/usr/bin/env python
"""Compare INR alignment accuracy: with vs without Expression INR.

Runs the full benchmark pipeline twice:
  1. Without Expression INR (baseline)
  2. With Expression INR (--use_expr_inr)

Outputs:
  - expr_comparison_results.csv   (per-pair, all methods both runs)
  - expr_comparison_summary.csv   (per-dataset mean + overall mean)
"""

from __future__ import annotations

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore")

from inr_align.config import SLICE_ORDER, DLPFC_SAMPLE_GROUPS, PipelineConfig
from inr_align.benchmark import (
    benchmark_all,
    benchmark_dataset,
    _load_dlpfc_layer_groups,
    DLPFC_LABEL_MAP,
)

# ---- All datasets ----
ALL_DATASETS = list(SLICE_ORDER.keys())

# Separate DLPFC vs sample_data datasets
DLPFC_DATASETS = [d for d in ALL_DATASETS if d.startswith("DLPFC_sample")]
OTHER_DATASETS = [d for d in ALL_DATASETS if not d.startswith("DLPFC_sample")]


def make_config(use_expr_inr: bool = False) -> PipelineConfig:
    """Create default config, optionally with Expression INR."""
    config = PipelineConfig(data_dir="./Data")
    config.icp.mode = "icp_only"
    config.train.epochs = 150
    config.train.lr = 1e-3
    config.joint.lam_jacobian = 0.005
    config.train.batch_size = 2000
    config.train.topk = 64
    config.train.warmup_fraction = 0.3
    config.train.weight_rev = 1.0
    config.train.grad_clip = 2.0
    config.train.scheduler_patience = 30
    config.train.scheduler_factor = 0.5
    config.matcher.lambda_feat = 1.0
    config.matcher.tau_init = 0.1
    config.matcher.ema_decay = 0.9
    config.model.hidden = 128
    config.model.layers = 6
    config.model.n_freqs = 6

    config.use_expr_field = use_expr_inr
    if use_expr_inr:
        config.expr_field.hidden = 256
        config.expr_field.encoder_layers = 4
        config.expr_field.lam_expr = 0.1
        config.expr_field.n_hvg = 200

    return config


def run_all_benchmarks(config: PipelineConfig, device: str, tag: str) -> pd.DataFrame:
    """Run benchmark on all datasets. Returns df with 'Tag' column."""
    all_dfs = []

    # ---- DLPFC ----
    if DLPFC_DATASETS:
        sample_indices = []
        for d in DLPFC_DATASETS:
            idx = int(d.replace("DLPFC_sample", "")) - 1
            if 0 <= idx < len(DLPFC_SAMPLE_GROUPS):
                sample_indices.append(idx)

        if sample_indices:
            print(f"\n{'#' * 60}")
            print(f"# [{tag}] DLPFC ({len(sample_indices)} sample groups)")
            print(f"{'#' * 60}")

            layer_groups = []
            for idx in sample_indices:
                group = DLPFC_SAMPLE_GROUPS[idx]
                folder = f"DLPFC_sample{idx + 1}"
                slices = []
                for sample_id in group:
                    path = f"{config.data_dir}/{folder}/original_data/{sample_id}.h5ad"
                    adata = sc.read_h5ad(path)
                    slices.append(adata)
                    print(f"  Loaded {sample_id}: {adata.shape}")
                layer_groups.append(slices)

            df_dlpfc, *_ = benchmark_all(
                layer_groups, config, device=device,
                label_key="original_domain", label_map=DLPFC_LABEL_MAP,
                run_paste=False, run_spateo=False,  # Only run Ours to save time
            )
            idx_to_name = {i: f"DLPFC_sample{sample_indices[i] + 1}" for i in range(len(sample_indices))}
            df_dlpfc["Dataset"] = df_dlpfc["Sample"].map(idx_to_name)
            df_dlpfc = df_dlpfc.drop(columns=["Sample"])
            all_dfs.append(df_dlpfc)

    # ---- Other datasets ----
    for dataset in OTHER_DATASETS:
        print(f"\n{'#' * 60}")
        print(f"# [{tag}] {dataset}")
        print(f"{'#' * 60}")

        df = benchmark_dataset(
            dataset, config, device=device,
            label_key="original_domain",
            run_paste=False, run_spateo=False,  # Only run Ours
        )
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["Tag"] = tag
    return df_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Datasets: {len(ALL_DATASETS)} total")

    t0 = time.time()

    # ---- Run 1: Without Expression INR ----
    print("\n" + "=" * 70)
    print("  RUN 1: WITHOUT Expression INR")
    print("=" * 70)
    config_base = make_config(use_expr_inr=False)
    df_base = run_all_benchmarks(config_base, device, tag="without_ExprINR")

    # ---- Run 2: With Expression INR ----
    print("\n" + "=" * 70)
    print("  RUN 2: WITH Expression INR")
    print("=" * 70)
    config_expr = make_config(use_expr_inr=True)
    df_expr = run_all_benchmarks(config_expr, device, tag="with_ExprINR")

    # ---- Combine ----
    df_all = pd.concat([df_base, df_expr], ignore_index=True)

    # ---- Per-pair CSV ----
    csv_path = "expr_comparison_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n✅ Per-pair results saved to {csv_path}")

    # ---- Summary: per-dataset mean ----
    # For each (Dataset, Tag, Method), compute mean accuracy
    summary_rows = []
    for dataset in df_all["Dataset"].unique():
        for tag in ["without_ExprINR", "with_ExprINR"]:
            df_sub = df_all[(df_all["Dataset"] == dataset) & (df_all["Tag"] == tag)]
            if len(df_sub) == 0:
                continue
            for method in df_sub["Method"].unique():
                vals = df_sub[df_sub["Method"] == method]["Accuracy"]
                summary_rows.append({
                    "Dataset": dataset,
                    "Tag": tag,
                    "Method": method,
                    "Mean_Acc": vals.mean(),
                    "Std_Acc": vals.std(),
                    "N_pairs": len(vals),
                })

    # Overall mean across all datasets
    for tag in ["without_ExprINR", "with_ExprINR"]:
        df_tag = df_all[df_all["Tag"] == tag]
        if len(df_tag) == 0:
            continue
        for method in df_tag["Method"].unique():
            vals = df_tag[df_tag["Method"] == method]["Accuracy"]
            summary_rows.append({
                "Dataset": "OVERALL_MEAN",
                "Tag": tag,
                "Method": method,
                "Mean_Acc": vals.mean(),
                "Std_Acc": vals.std(),
                "N_pairs": len(vals),
            })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = "expr_comparison_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"✅ Summary saved to {summary_path}")

    # ---- Pretty print ----
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY: Without vs With ExprINR")
    print("=" * 90)

    # Pivot: rows = (Dataset, Method), cols = Tag
    if len(df_summary) > 0:
        pivot = df_summary.pivot_table(
            index=["Dataset", "Method"],
            columns="Tag",
            values="Mean_Acc",
        )
        if "without_ExprINR" in pivot.columns and "with_ExprINR" in pivot.columns:
            pivot["Delta"] = pivot["with_ExprINR"] - pivot["without_ExprINR"]
        print(pivot.to_string(float_format="{:.4f}".format))

    elapsed = time.time() - t0
    print(f"\n⏱  Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()

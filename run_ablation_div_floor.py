#!/usr/bin/env python
"""Ablation: divergence loss + sigma floor loss.

Runs 3 configs on DLPFC (9 pairs) + STARMap (2 pairs) + BaristaSeq (2 pairs):
  A) baseline (current defaults: lam_divergence=10, lam_sigma_floor=1)
  B) no divergence (lam_divergence=0, lam_sigma_floor=1)
  C) no div + no floor (lam_divergence=0, lam_sigma_floor=0)

Only runs our method (no baselines) for speed.
"""

from __future__ import annotations

import copy
import warnings

import pandas as pd
import torch

from inr_align.benchmark import benchmark_all, benchmark_dataset, _load_dlpfc_layer_groups, DLPFC_LABEL_MAP
from inr_align.config import PipelineConfig

warnings.filterwarnings("ignore")

DATASET_OVERRIDES = {
    "DLPFC": {"lam_jacobian": 0.1, "warmup_fraction": 0.3, "lam_uniqueness": 0.0},
    "STARMap": {"lam_jacobian": 0.01, "warmup_fraction": 0.4, "lam_uniqueness": 0.1},
    "BaristaSeq": {"lam_jacobian": 0.01, "warmup_fraction": 0.4, "lam_uniqueness": 0.1},
}

ABLATION_CONFIGS = {
    "baseline":       {"lam_divergence": 10.0, "lam_sigma_floor": 1.0, "sigma_min": 0.8},
    "no_div":         {"lam_divergence": 0.0,  "lam_sigma_floor": 1.0, "sigma_min": 0.8},
    "no_div_nofloor": {"lam_divergence": 0.0,  "lam_sigma_floor": 0.0, "sigma_min": 0.8},
}


def make_config(base: PipelineConfig, dataset: str, ablation: dict) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    # Dataset overrides
    for prefix, overrides in DATASET_OVERRIDES.items():
        if dataset.startswith(prefix):
            for k, v in overrides.items():
                setattr(cfg.train, k, v)
            break
    # Ablation overrides
    for k, v in ablation.items():
        setattr(cfg.train, k, v)
    return cfg


def run_dlpfc(base_config, device, ablation_name, ablation_params):
    layer_groups = _load_dlpfc_layer_groups(base_config.data_dir)
    cfg = make_config(base_config, "DLPFC", ablation_params)
    df = benchmark_all(
        layer_groups, cfg, device=device,
        label_key="original_domain", label_map=DLPFC_LABEL_MAP,
        run_paste=False, run_spateo=False, run_spacel=False, run_stalign=False,
    )
    df["Dataset"] = df["Sample"].map({0: "DLPFC_sample1", 1: "DLPFC_sample2", 2: "DLPFC_sample3"})
    df = df.drop(columns=["Sample"])
    df["Ablation"] = ablation_name
    return df


def run_dataset(base_config, device, dataset, ablation_name, ablation_params):
    cfg = make_config(base_config, dataset, ablation_params)
    df = benchmark_dataset(
        dataset, cfg, device=device, label_key="original_domain",
        run_paste=False, run_spateo=False, run_spacel=False, run_stalign=False,
    )
    df["Ablation"] = ablation_name
    return df


def main():
    base = PipelineConfig(data_dir="./Data")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    all_dfs = []
    for abl_name, abl_params in ABLATION_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"  ABLATION: {abl_name}  {abl_params}")
        print(f"{'=' * 60}")

        # DLPFC
        all_dfs.append(run_dlpfc(base, device, abl_name, abl_params))

        # STARMap + BaristaSeq
        for ds in ["STARMap", "BaristaSeq"]:
            df = run_dataset(base, device, ds, abl_name, abl_params)
            if len(df) > 0:
                all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv("ablation_div_floor.csv", index=False)

    # Summary
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (Nonrigid only)")
    print("=" * 80)

    df_nr = df_all[df_all["Method"] == "INSTA-Nonrigid"]
    metrics = ["Accuracy", "Accuracy_NN", "Ratio", "CLC", "LISI", "Silhouette", "Chamfer"]
    summary = df_nr.groupby("Ablation")[metrics].agg(["mean", "std"])

    for abl in ABLATION_CONFIGS:
        if abl not in summary.index:
            continue
        row = summary.loc[abl]
        print(f"\n  {abl}:")
        for m in metrics:
            print(f"    {m:15s}: {row[(m, 'mean')]:.4f} ± {row[(m, 'std')]:.4f}")

    print(f"\n✅ Full results: ablation_div_floor.csv")


if __name__ == "__main__":
    main()

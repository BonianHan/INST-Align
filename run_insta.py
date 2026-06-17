#!/usr/bin/env python
"""Run INST-Align only for the spatial alignment experiment.

This script saves rows for ``INSTA-Rigid`` and ``INSTA-Nonrigid``.
"""

from __future__ import annotations

import argparse
import copy
import os
import warnings
from typing import List, Optional

import pandas as pd
import scanpy as sc
import torch

from insta.spatial_alignment import (
    DLPFC_LABEL_MAP,
    run_spatial_alignment_all,
    run_spatial_alignment_dataset,
)
from insta.config import (
    DLPFC_SAMPLE_GROUPS,
    PipelineConfig,
    add_pipeline_args,
    config_from_args,
    print_config,
)

warnings.filterwarnings("ignore")


PAPER_DATASETS = [
    "DLPFC",
    "STARMap",
    "MERFISH",
    "MERFISH_Brain_S2",
    "MERFISH_Brain_S7",
    "MERFISH_Brain_S11",
    "MouseEmbryo",
]

DATASET_OVERRIDES = {
    "MouseEmbryo": {
        "epochs": 500, "batch_size": 4000, "lr": 3e-4, "grad_clip": 2.0,
        "lam_jacobian": 0.01, "lam_deform_mag": 0.0,
        "tau_min": 0.005, "mode": "pca",
        "inr_pretrain_epochs": 150,
        "freeze_inr_phase2": True, "lam_recon_phase2": 0.0,
        "warmup_fraction": 0.0, "scheduler_patience": 9999,
    },
}

_DLPFC_ID_TO_IDX = {"151507": 0, "151669": 1, "151673": 2}
_DLPFC_IDX_TO_ID = {0: "151507", 1: "151669", 2: "151673"}


def config_for_dataset(config: PipelineConfig, dataset: str) -> PipelineConfig:
    key = None
    for prefix in DATASET_OVERRIDES:
        if dataset.startswith(prefix) and (key is None or len(prefix) > len(key)):
            key = prefix
    if key is None:
        return config

    cfg = copy.deepcopy(config)
    for field_name, value in DATASET_OVERRIDES[key].items():
        if hasattr(cfg.train, field_name):
            setattr(cfg.train, field_name, value)
        elif hasattr(cfg.joint, field_name):
            setattr(cfg.joint, field_name, value)
        elif hasattr(cfg.matcher, field_name):
            setattr(cfg.matcher, field_name, value)
        elif hasattr(cfg.icp, field_name):
            setattr(cfg.icp, field_name, value)
    print(f"  [dataset override] {dataset}: {DATASET_OVERRIDES[key]}")
    return cfg


def resolve_datasets(datasets: Optional[List[str]]) -> List[str]:
    if datasets is None:
        datasets = PAPER_DATASETS

    expanded = []
    sample_name_to_id = {
        "DLPFC_sample1": "DLPFC_151507",
        "DLPFC_sample2": "DLPFC_151669",
        "DLPFC_sample3": "DLPFC_151673",
    }
    for dataset in datasets:
        if dataset == "DLPFC":
            expanded.extend(["DLPFC_151507", "DLPFC_151669", "DLPFC_151673"])
        elif dataset in sample_name_to_id:
            expanded.append(sample_name_to_id[dataset])
        else:
            expanded.append(dataset)
    return expanded


def load_dlpfc_groups(data_dir: str, sample_indices: List[int]):
    layer_groups = []
    sample_id_groups = []
    dataset_folders = []
    for idx in sample_indices:
        group = DLPFC_SAMPLE_GROUPS[idx]
        folder = f"DLPFC_sample{idx + 1}"
        slices = []
        for sample_id in group:
            path = os.path.join(data_dir, folder, "original_data", f"{sample_id}.h5ad")
            adata = sc.read_h5ad(path)
            slices.append(adata)
            print(f"  Loaded {sample_id}: {adata.shape}")
        layer_groups.append(slices)
        sample_id_groups.append(list(group))
        dataset_folders.append(folder)
    return layer_groups, sample_id_groups, dataset_folders


def save_results(df: pd.DataFrame, output_name: str) -> None:
    csv_path = f"{output_name}.csv"
    summary_path = f"{output_name}_summary.csv"
    df.to_csv(csv_path, index=False)

    metric_cols = [
        c for c in ["Time", "Accuracy", "Accuracy_NN", "Chamfer"]
        if c in df.columns
    ]
    summary = df.groupby(["Dataset", "Method"], as_index=False)[metric_cols].mean()
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {csv_path}")
    print(f"Saved {summary_path}")


def main(
    config: Optional[PipelineConfig] = None,
    datasets: Optional[List[str]] = None,
    output_name: str = "insta_results",
) -> None:
    if config is None:
        config = PipelineConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = resolve_datasets(datasets)

    dlpfc_datasets = [d for d in datasets if d.startswith("DLPFC_")]
    other_datasets = [d for d in datasets if not d.startswith("DLPFC_")]
    all_dfs = []

    if dlpfc_datasets:
        sample_indices = []
        for dataset in dlpfc_datasets:
            sid = dataset.replace("DLPFC_", "")
            if sid in _DLPFC_ID_TO_IDX:
                sample_indices.append(_DLPFC_ID_TO_IDX[sid])
        if sample_indices:
            print("\n" + "#" * 60)
            print(f"# DLPFC INST-Align ({len(sample_indices)} sample groups)")
            print("#" * 60)
            layer_groups, sample_id_groups, dataset_folders = load_dlpfc_groups(
                config.data_dir, sample_indices,
            )
            df, *_ = run_spatial_alignment_all(
                layer_groups, config_for_dataset(config, "DLPFC"), device=device,
                label_key="original_domain", label_map=DLPFC_LABEL_MAP,
                run_paste=False, run_paste2=False, run_gpsa=False,
                run_spateo=False, run_stalign=False, run_insta=True,
                sample_id_groups=sample_id_groups,
                dataset_folders=dataset_folders,
            )
            idx_to_name = {i: f"DLPFC_{_DLPFC_IDX_TO_ID[sample_indices[i]]}" for i in range(len(sample_indices))}
            df["Dataset"] = df["Sample"].map(idx_to_name)
            df = df.drop(columns=["Sample"])
            all_dfs.append(df)

    for dataset in other_datasets:
        print(f"\n{'#' * 60}")
        print(f"# {dataset} INST-Align")
        print(f"{'#' * 60}")
        ds_config = config_for_dataset(config, dataset)
        df = run_spatial_alignment_dataset(
            dataset, ds_config, device=device,
            label_key="original_domain",
            run_paste=False, run_paste2=False, run_gpsa=False,
            run_spateo=False, run_stalign=False, run_insta=True,
        )
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        print("\nNo INST-Align results to save.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all[df_all["Method"].isin(["INSTA-Rigid", "INSTA-Nonrigid"])].copy()
    save_results(df_all, output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run INST-Align only for Experiment 1")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run. Default: paper Experiment 1 datasets.")
    parser.add_argument("--output", default="insta_results",
                        help="Output file prefix.")
    add_pipeline_args(parser)

    args = parser.parse_args()
    cfg = config_from_args(args)
    print_config(cfg)
    main(config=cfg, datasets=args.datasets, output_name=args.output)

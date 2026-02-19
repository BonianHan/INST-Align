#!/usr/bin/env python3
"""Ablation: run INSTA on DLPFC sample1 with different configs.

Configs:
  A) Baseline        — no gene decoder
  B) + gene decoder  — lam_kl + lam_recon (requires Splane embeddings)

Usage:
    conda activate spateo
    python run_ablation_dlpfc.py
"""
import copy
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import spateo as st

from inr_align.benchmark import (
    DLPFC_LABEL_MAP,
    run_ours,
)
from inr_align.config import DLPFC_SAMPLE_GROUPS, PipelineConfig
from inr_align.metrics import (
    calculate_clc,
    chamfer_distance,
    compute_lisi,
    compute_silhouette,
    coords_to_pi,
    mapping_accuracy_nn_bidi,
    mapping_accuracy_paste,
)
from inr_align.utils import normalize_coordinates

# ============================================================================
# Setup
# ============================================================================

SAMPLE_IDX = 0  # DLPFC_sample1
GROUP = DLPFC_SAMPLE_GROUPS[SAMPLE_IDX]
FOLDER = f"DLPFC_sample{SAMPLE_IDX + 1}"
DATA_DIR = "./Data"
LABEL_KEY = "original_domain"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


def load_and_preprocess():
    """Load DLPFC sample1, preprocess, return (slices, sample_ids)."""
    slices = []
    for sid in GROUP:
        path = os.path.join(DATA_DIR, FOLDER, "original_data", f"{sid}.h5ad")
        adata = sc.read_h5ad(path)
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if "highly_variable" not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        slices.append(adata)
        print(f"  Loaded {sid}: {adata.shape}")
    st.align.group_pca(slices, pca_key="X_pca")
    return slices


def make_base_config():
    """Base config with DLPFC-tuned hyperparameters."""
    cfg = PipelineConfig(data_dir=DATA_DIR)
    cfg.joint.lam_jacobian = 0.1
    cfg.train.warmup_fraction = 0.3
    return cfg


# ============================================================================
# Run ablation
# ============================================================================

def run_config(slices, config, config_name, dataset_folder):
    """Run our method on all consecutive pairs, return results DataFrame."""
    print(f"\n{'=' * 60}")
    print(f"  Config: {config_name}")
    print(f"{'=' * 60}")

    rows = []
    for i in range(len(slices) - 1):
        s1, s2 = slices[i], slices[i + 1]
        sid1, sid2 = GROUP[i], GROUP[i + 1]
        print(f"\n  Pair {i}: {sid1} → {sid2}")

        c2_rigid, c2_final, elapsed = run_ours(
            s1, s2, config, device=device,
            label_key=LABEL_KEY,
            sample_id2=sid2, dataset_folder=FOLDER,
        )

        # Compute metrics on both rigid and nonrigid results
        c1 = s1.obsm["spatial"]
        l1 = np.asarray(s1.obs[LABEL_KEY])
        l2 = np.asarray(s2.obs[LABEL_KEY])

        def _metrics(c2_aligned):
            pi = coords_to_pi(c1, c2_aligned)
            acc_ot = mapping_accuracy_paste(s1.obs[LABEL_KEY], s2.obs[LABEL_KEY], pi, DLPFC_LABEL_MAP)
            acc_nn, ratio = mapping_accuracy_nn_bidi(l1, l2, c1, c2_aligned)
            clc = calculate_clc(l1, l2, c1, c2_aligned)
            lisi = compute_lisi(c1, c2_aligned, l1, l2)
            sil = compute_silhouette(c1, c2_aligned, l1, l2)
            cham = chamfer_distance(c1, c2_aligned)
            return acc_ot, acc_nn, ratio, clc, lisi, sil, cham

        r_ot, r_nn, r_rat, r_clc, r_lisi, r_sil, r_cham = _metrics(c2_rigid)
        s_ot, s_nn, s_rat, s_clc, s_lisi, s_sil, s_cham = _metrics(c2_final)

        rows.append({
            "Config": config_name,
            "Pair": f"{sid1}→{sid2}",
            "Rigid_OT": r_ot, "Rigid_NN": r_nn,
            "Rigid_Ratio": r_rat, "Rigid_CLC": r_clc,
            "Spatial_OT": s_ot, "Spatial_NN": s_nn,
            "Spatial_Ratio": s_rat, "Spatial_CLC": s_clc,
            "Spatial_LISI": s_lisi, "Spatial_Silhouette": s_sil, "Spatial_Chamfer": s_cham,
            "Time": elapsed,
        })
        print(f"    Rigid:   OT={r_ot:.4f}  NN={r_nn:.4f}  Ratio={r_rat:.4f}  CLC={r_clc:.4f}")
        print(f"    Spatial: OT={s_ot:.4f}  NN={s_nn:.4f}  Ratio={s_rat:.4f}  CLC={s_clc:.4f}  LISI={s_lisi:.3f}  Sil={s_sil:.4f}  Cham={s_cham:.4f}")
        print(f"    Time: {elapsed:.1f}s")

    return pd.DataFrame(rows)


def main():
    slices = load_and_preprocess()

    all_dfs = []

    # ----- A) Baseline: no gene decoder -----
    cfg_a = make_base_config()
    df_a = run_config(slices, cfg_a, "A_baseline", FOLDER)
    all_dfs.append(df_a)

    # ----- B) + higher recon weight -----
    cfg_b = make_base_config()
    cfg_b.joint.lam_recon_phase2 = 0.5
    df_b = run_config(slices, cfg_b, "B_recon_0.5", FOLDER)
    all_dfs.append(df_b)

    # ----- Combine and print -----
    df_all = pd.concat(all_dfs, ignore_index=True)

    print("\n" + "=" * 80)
    print("ABLATION RESULTS — DLPFC sample1 (Spatial metrics)")
    print("=" * 80)

    # Summary per config
    for config_name in df_all["Config"].unique():
        df_c = df_all[df_all["Config"] == config_name]
        print(f"\n  {config_name} ({len(df_c)} pairs):")
        for metric in ["Spatial_OT", "Spatial_NN", "Spatial_Ratio", "Spatial_CLC",
                       "Spatial_LISI", "Spatial_Silhouette", "Spatial_Chamfer"]:
            vals = df_c[metric]
            print(f"    {metric:16s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Save
    out_path = "ablation_dlpfc.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark INSTA vs baselines on DLPFC sample1.

Metrics:
  Alignment: OT, NN, Ratio, CLC
  Embedding: ARI, NMI (on Splane embeddings, clustering against ground-truth domains)

Methods compared:
  No-align, PASTE, SPACEL, STalign, Spateo (Rigid/Nonrigid), INSTA (Rigid/Nonrigid)

Usage:
    conda activate spateo
    python benchmark_insta.py [--no_paste] [--no_stalign] [--no_spacel] [--no_spateo]
"""
import argparse
import copy
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from inr_align.benchmark import (
    DLPFC_LABEL_MAP,
    _load_dlpfc_layer_groups,
    benchmark_all,
    print_summary,
    plot_comparison,
)
from inr_align.config import DLPFC_SAMPLE_GROUPS, PipelineConfig
from inr_align.model1 import JointConfig


# ============================================================================
# ARI / NMI helpers
# ============================================================================


def compute_ari_nmi(embeddings, labels, n_clusters=7):
    """Cluster embeddings and compute ARI + NMI against ground truth."""
    valid = np.array([str(l).strip() not in ("", "nan") for l in labels])
    emb = embeddings[valid]
    lab = labels[valid]
    unique_labels = np.unique(lab)
    if len(unique_labels) < 2:
        return 0.0, 0.0
    n_c = min(n_clusters, len(unique_labels))
    km = KMeans(n_clusters=n_c, n_init=10, random_state=42)
    pred = km.fit_predict(emb)
    ari = adjusted_rand_score(lab, pred)
    nmi = normalized_mutual_info_score(lab, pred)
    return ari, nmi


def evaluate_embeddings_dlpfc(slices, sample_ids, pca_key="X_pca", label_key="original_domain"):
    """Evaluate PCA and Splane embeddings on each slice via ARI+NMI.

    Returns DataFrame with columns: Slice, Embedding, ARI, NMI
    """
    rows = []

    # Number of ground-truth domains
    all_labels = set()
    for s in slices:
        all_labels.update(np.asarray(s.obs[label_key]))
    n_clusters = len([l for l in all_labels if str(l).strip() not in ("", "nan")])

    # --- PCA embeddings ---
    for i, s in enumerate(slices):
        pca_emb = s.obsm[pca_key].copy()
        labels = np.asarray(s.obs[label_key])
        ari, nmi = compute_ari_nmi(pca_emb, labels, n_clusters)
        rows.append({"Slice": sample_ids[i], "Embedding": "PCA", "ARI": ari, "NMI": nmi})

    # --- Splane embeddings (if available) ---
    splane_path = f"./Data/DLPFC_sample1/splane_embeddings.npz"
    if os.path.exists(splane_path):
        splane_data = np.load(splane_path)
        for i, sid in enumerate(sample_ids):
            emb_key = f"emb_{sid}"
            if emb_key in splane_data:
                splane_emb = splane_data[emb_key]
                labels = np.asarray(slices[i].obs[label_key])
                ari, nmi = compute_ari_nmi(splane_emb, labels, n_clusters)
                rows.append({"Slice": sid, "Embedding": "Splane", "ARI": ari, "NMI": nmi})
    else:
        print(f"  [WARN] Splane embeddings not found at {splane_path}")
        print(f"         Run: conda activate spacel && python extract_splane_emb.py")

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark INSTA on DLPFC sample1")
    parser.add_argument("--no_paste", action="store_true")
    parser.add_argument("--no_spateo", action="store_true")
    parser.add_argument("--no_spacel", action="store_true")
    parser.add_argument("--no_stalign", action="store_true")
    parser.add_argument("--sample_groups", type=int, nargs="+", default=[0],
                        help="Which DLPFC sample groups to run (0-indexed). Default: [0] = sample1 only")
    parser.add_argument("--joint", action="store_true",
                        help="Also run INSTA-Joint (Encoder+Decoder+Discriminator architecture)")
    parser.add_argument("--joint_lam_adv", type=float, default=0.1,
                        help="Joint: adversarial loss weight")
    parser.add_argument("--joint_lam_recon", type=float, default=1.0,
                        help="Joint: reconstruction loss weight")
    parser.add_argument("--joint_emb_dim", type=int, default=64,
                        help="Joint: encoder embedding dimension")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = PipelineConfig(data_dir="./Data")
    # DLPFC hyperparameter overrides
    config.train.lam_jacobian = 0.1
    config.train.warmup_fraction = 0.3
    config.icp.mode = "icp_only"

    # Joint config
    jcfg = None
    if args.joint:
        jcfg = JointConfig(
            emb_dim=args.joint_emb_dim,
            lam_adv=args.joint_lam_adv,
            lam_recon=args.joint_lam_recon,
        )

    # ================================================================
    # Part 1: Alignment Benchmark (4 metrics: OT, NN, Ratio, CLC)
    # ================================================================
    print("\n" + "#" * 70)
    print("# Part 1: Alignment Benchmark")
    print("#" * 70)

    # Load requested sample groups
    sample_indices = args.sample_groups
    layer_groups = []
    sample_id_groups = []
    dataset_folders = []
    for idx in sample_indices:
        group = DLPFC_SAMPLE_GROUPS[idx]
        slices = []
        folder = f"DLPFC_sample{idx + 1}"
        for sample_id in group:
            path = f"./Data/{folder}/original_data/{sample_id}.h5ad"
            adata = sc.read_h5ad(path)
            slices.append(adata)
            print(f"  Loaded {sample_id}: {adata.shape}")
        layer_groups.append(slices)
        sample_id_groups.append(group)
        dataset_folders.append(folder)

    df_align = benchmark_all(
        layer_groups, config, device=device,
        label_key="original_domain", label_map=DLPFC_LABEL_MAP,
        run_paste=not args.no_paste,
        run_spateo=not args.no_spateo,
        run_spacel=not args.no_spacel,
        run_stalign=not args.no_stalign,
        sample_id_groups=sample_id_groups,
        dataset_folders=dataset_folders,
        run_joint=args.joint,
        joint_config=jcfg,
    )

    # Add Dataset column
    idx_to_name = {i: f"DLPFC_sample{sample_indices[i] + 1}" for i in range(len(sample_indices))}
    df_align["Dataset"] = df_align["Sample"].map(idx_to_name)

    print("\n" + "=" * 70)
    print("Alignment Results")
    print("=" * 70)
    print_summary(df_align)

    # Save alignment results
    df_align.to_csv("benchmark_alignment.csv", index=False)
    print(f"\n✅ Alignment results saved to benchmark_alignment.csv")

    try:
        plot_comparison(df_align, "benchmark_alignment.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # ================================================================
    # Part 2: Embedding Evaluation (ARI, NMI)
    # ================================================================
    print("\n" + "#" * 70)
    print("# Part 2: Embedding Evaluation (ARI, NMI)")
    print("#" * 70)

    import spateo as st

    all_emb_dfs = []
    for idx in sample_indices:
        group = DLPFC_SAMPLE_GROUPS[idx]
        folder = f"DLPFC_sample{idx + 1}"
        print(f"\n  --- DLPFC sample{idx + 1} ({group}) ---")

        # Load and preprocess
        slices = []
        for sample_id in group:
            path = f"./Data/{folder}/original_data/{sample_id}.h5ad"
            adata = sc.read_h5ad(path)
            if "counts" not in adata.layers:
                adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            if "highly_variable" not in adata.var.columns:
                sc.pp.highly_variable_genes(adata, n_top_genes=2000)
            slices.append(adata)
        st.align.group_pca(slices, pca_key="X_pca")

        df_emb = evaluate_embeddings_dlpfc(slices, group)
        df_emb["Sample"] = f"DLPFC_sample{idx + 1}"
        all_emb_dfs.append(df_emb)

    if all_emb_dfs:
        df_emb_all = pd.concat(all_emb_dfs, ignore_index=True)

        print("\n" + "=" * 70)
        print("Embedding Evaluation Results (ARI / NMI)")
        print("=" * 70)

        # Print per-slice
        for sample in df_emb_all["Sample"].unique():
            print(f"\n  {sample}:")
            df_s = df_emb_all[df_emb_all["Sample"] == sample]
            for emb_type in df_s["Embedding"].unique():
                df_e = df_s[df_s["Embedding"] == emb_type]
                print(f"    {emb_type:10s} | "
                      f"ARI: {df_e['ARI'].mean():.4f} ± {df_e['ARI'].std():.4f} | "
                      f"NMI: {df_e['NMI'].mean():.4f} ± {df_e['NMI'].std():.4f}")

        # Print per-slice detail
        print("\n  Per-slice detail:")
        print(df_emb_all.to_string(index=False, float_format="{:.4f}".format))

        df_emb_all.to_csv("benchmark_embedding.csv", index=False)
        print(f"\n✅ Embedding results saved to benchmark_embedding.csv")


if __name__ == "__main__":
    main()

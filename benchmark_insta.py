#!/usr/bin/env python3
"""Benchmark INSTA on DLPFC (all sample groups).

Metrics:
  Alignment: OT, NN, Ratio, CLC, LISI, Silhouette, Chamfer
  Embedding: ARI, NMI (on PCA / INR / Splane embeddings)

Usage:
    conda activate spateo
    python benchmark_insta.py [--sample_groups 0 1 2]
"""
import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # Save plots only, no display

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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


def evaluate_embeddings_dlpfc(slices, sample_ids, inr_models_for_group=None,
                              norm_params_for_group=None, pca_key="X_pca",
                              label_key="original_domain", device="cpu"):
    """Evaluate PCA, INR, and Splane embeddings on each slice via ARI+NMI.

    Args:
        inr_models_for_group: list of trained ExprINR models (one per pair,
            i.e. len = len(slices)-1). Each INR1 was trained on pair (i, i+1)
            as the target INR. We evaluate slice i with pair i's INR1.
        norm_params_for_group: list of (mean, std) tuples, one per pair.
            Each INR was trained using pair-specific normalization.
    """
    rows = []

    all_labels = set()
    for s in slices:
        all_labels.update(np.asarray(s.obs[label_key]))
    n_clusters = len([l for l in all_labels if str(l).strip() not in ("", "nan")])

    # PCA embeddings
    for i, s in enumerate(slices):
        pca_emb = s.obsm[pca_key].copy()
        labels = np.asarray(s.obs[label_key])
        ari, nmi = compute_ari_nmi(pca_emb, labels, n_clusters)
        rows.append({"Slice": sample_ids[i], "Embedding": "PCA", "ARI": ari, "NMI": nmi})

    # INR embeddings (from trained ExprINR)
    # Pair i's INR1 is trained on slice i (target) using pair i's normalization.
    # Only evaluate slices 0..n_pairs-1 (each has a matching INR1).
    if inr_models_for_group and norm_params_for_group:
        for i in range(len(inr_models_for_group)):
            s = slices[i]
            inr_model = inr_models_for_group[i]
            mean, std = norm_params_for_group[i]
            if inr_model is not None:
                # Normalize coords with the SAME params used during training
                raw_coords = s.obsm["spatial"]
                coords_norm = (raw_coords - mean) / std
                inr_model.eval()
                with torch.no_grad():
                    coords_t = torch.tensor(coords_norm.astype(np.float32), device=device)
                    alpha = torch.tensor(float(inr_model.n_freqs), device=device)
                    inr_emb = inr_model(coords_t, alpha).cpu().numpy()
                labels = np.asarray(s.obs[label_key])
                ari, nmi = compute_ari_nmi(inr_emb, labels, n_clusters)
                rows.append({"Slice": sample_ids[i], "Embedding": "INR", "ARI": ari, "NMI": nmi})

    # Splane embeddings (if available)
    splane_path = "./Data/DLPFC_sample1/splane_embeddings.npz"
    if os.path.exists(splane_path):
        splane_data = np.load(splane_path)
        for i, sid in enumerate(sample_ids):
            emb_key = f"emb_{sid}"
            if emb_key in splane_data:
                splane_emb = splane_data[emb_key]
                labels = np.asarray(slices[i].obs[label_key])
                ari, nmi = compute_ari_nmi(splane_emb, labels, n_clusters)
                rows.append({"Slice": sid, "Embedding": "Splane", "ARI": ari, "NMI": nmi})

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark INSTA on DLPFC")
    parser.add_argument("--no_paste", action="store_true", default=True)
    parser.add_argument("--no_spateo", action="store_true", default=False)
    parser.add_argument("--no_stalign", action="store_true", default=True)
    parser.add_argument("--sample_groups", type=int, nargs="+", default=[0, 1, 2],
                        help="Which DLPFC sample groups to run (0-indexed). Default: all 3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = PipelineConfig(data_dir="./Data")
    config.icp.mode = "icp_only"
    # DLPFC-specific: lam_jacobian=0.1 (default 1.0 is too strong for grid data)
    config.joint.lam_jacobian = 0.1

    # ================================================================
    # Part 1: Alignment Benchmark
    # ================================================================
    print("\n" + "#" * 70)
    print("# Part 1: Alignment Benchmark")
    print("#" * 70)

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

    df_align, inr_models, norm_params = benchmark_all(
        layer_groups, config, device=device,
        label_key="original_domain", label_map=DLPFC_LABEL_MAP,
        run_paste=not args.no_paste,
        run_spateo=not args.no_spateo,
        run_stalign=not args.no_stalign,
        sample_id_groups=sample_id_groups,
        dataset_folders=dataset_folders,
    )

    idx_to_name = {i: f"DLPFC_sample{sample_indices[i] + 1}" for i in range(len(sample_indices))}
    df_align["Dataset"] = df_align["Sample"].map(idx_to_name)

    print("\n" + "=" * 70)
    print("Alignment Results")
    print("=" * 70)
    print_summary(df_align)

    df_align.to_csv("benchmark_alignment.csv", index=False)
    print(f"\nAlignment results saved to benchmark_alignment.csv")

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
    for gi, idx in enumerate(sample_indices):
        group = DLPFC_SAMPLE_GROUPS[idx]
        folder = f"DLPFC_sample{idx + 1}"
        print(f"\n  --- DLPFC sample{idx + 1} ({group}) ---")

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

        # Collect trained INR models and their normalization params
        inr_list = []
        norm_list = []
        for pi in range(len(group) - 1):
            inr_m = inr_models.get((gi, pi), None)
            inr_list.append(inr_m)
            norm_list.append(norm_params.get((gi, pi), None))

        df_emb = evaluate_embeddings_dlpfc(slices, group,
                                           inr_models_for_group=inr_list,
                                           norm_params_for_group=norm_list,
                                           device=device)
        df_emb["Sample"] = f"DLPFC_sample{idx + 1}"
        all_emb_dfs.append(df_emb)

    if all_emb_dfs:
        df_emb_all = pd.concat(all_emb_dfs, ignore_index=True)

        print("\n" + "=" * 70)
        print("Embedding Evaluation Results (ARI / NMI)")
        print("=" * 70)

        for sample in df_emb_all["Sample"].unique():
            print(f"\n  {sample}:")
            df_s = df_emb_all[df_emb_all["Sample"] == sample]
            for emb_type in df_s["Embedding"].unique():
                df_e = df_s[df_s["Embedding"] == emb_type]
                print(f"    {emb_type:10s} | "
                      f"ARI: {df_e['ARI'].mean():.4f} +/- {df_e['ARI'].std():.4f} | "
                      f"NMI: {df_e['NMI'].mean():.4f} +/- {df_e['NMI'].std():.4f}")

        print("\n  Per-slice detail:")
        print(df_emb_all.to_string(index=False, float_format="{:.4f}".format))

        df_emb_all.to_csv("benchmark_embedding.csv", index=False)
        print(f"\nEmbedding results saved to benchmark_embedding.csv")


if __name__ == "__main__":
    main()

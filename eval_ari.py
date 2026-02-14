"""Evaluate ExprField embedding quality via ARI (Adjusted Rand Index).

Three evaluations:
1. Pretrain-only: ExprField pretrained jointly on all slices, evaluate on each slice
2. Frozen: ExprField frozen during deformation, evaluate on source slice after alignment
3. Unfrozen: ExprField fine-tuned during deformation, evaluate on source slice after alignment

Usage:
    python eval_ari.py [--data_dir ./Data] [--device cuda]
"""

import argparse
import sys
import warnings

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Local imports
sys.path.insert(0, ".")
import spateo as st

from inr_align.benchmark import DLPFC_LABEL_MAP, DLPFC_SAMPLE_GROUPS
from inr_align.config import ExprFieldConfig, PipelineConfig
from inr_align.model import ExprField, pretrain_expr_field
from inr_align.run import pretrain_expr_field_pipeline
from inr_align.utils import normalize_coordinates
from inr_align.model import normalize_expression


def compute_ari(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = 7) -> float:
    """Cluster embeddings with KMeans and compute ARI against ground truth."""
    # Remove NaN labels
    valid = ~(labels == "nan") & ~(labels == "")
    if hasattr(labels[0], "strip"):
        valid = np.array([l.strip() != "" and l.strip() != "nan" for l in labels])
    emb = embeddings[valid]
    lab = labels[valid]
    if len(np.unique(lab)) < 2:
        return 0.0
    n_clusters = min(n_clusters, len(np.unique(lab)))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred = km.fit_predict(emb)
    return adjusted_rand_score(lab, pred)


def compute_ari_pca(pca_emb: np.ndarray, labels: np.ndarray, n_clusters: int = 7) -> float:
    """Compute ARI using PCA embeddings as baseline."""
    return compute_ari(pca_emb, labels, n_clusters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./Data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    config = PipelineConfig()
    config.use_expr_field = True
    ef_config = config.expr_field
    label_key = "original_domain"

    print("=" * 70)
    print("ExprField Embedding ARI Evaluation")
    print("=" * 70)

    for j, group in enumerate(DLPFC_SAMPLE_GROUPS):
        print(f"\n{'=' * 60}")
        print(f"DLPFC Sample Group {j} ({group})")
        print(f"{'=' * 60}")

        # Load slices
        slices = []
        for sample_id in group:
            folder = f"DLPFC_sample{j + 1}"
            path = f"{args.data_dir}/{folder}/original_data/{sample_id}.h5ad"
            adata = sc.read_h5ad(path)
            slices.append(adata)
            print(f"  Loaded {sample_id}: {adata.shape}")

        # Preprocess
        for ad_ in slices:
            if "counts" not in ad_.layers:
                ad_.layers["counts"] = ad_.X.copy()
            sc.pp.normalize_total(ad_)
            sc.pp.log1p(ad_)
            if "highly_variable" not in ad_.var.columns:
                sc.pp.highly_variable_genes(ad_, n_top_genes=config.n_top_genes)
        st.align.group_pca(slices, pca_key=config.pca_key)

        # Normalize coordinates
        all_coords = [s.obsm[config.spatial_key] for s in slices]
        coords_norm_list, mean_, std_ = normalize_coordinates(all_coords)

        # Get ground truth labels
        all_labels = [np.asarray(s.obs[label_key]) for s in slices]

        # Number of unique labels (clusters)
        all_unique = set()
        for lab in all_labels:
            all_unique.update(lab)
        n_clusters = len([l for l in all_unique if l != "nan" and l.strip() != ""])
        print(f"  {n_clusters} domains: {sorted(all_unique)}")

        # ================================================================
        # 1. PCA baseline ARI
        # ================================================================
        print(f"\n  --- PCA Embedding ARI ---")
        for i, s in enumerate(slices):
            pca_emb = s.obsm[config.pca_key]
            ari = compute_ari(pca_emb, all_labels[i], n_clusters)
            print(f"    Slice {i} ({group[i]}): ARI = {ari:.4f}")

        # ================================================================
        # 2. ExprField Pretrain ARI (all slices)
        # ================================================================
        print(f"\n  --- ExprField Pretrain ARI ---")
        pretrained_ef = pretrain_expr_field_pipeline(
            [s.copy() for s in slices], coords_norm_list, config, device,
        )
        if pretrained_ef is None:
            print("    ExprField disabled, skipping")
            continue

        expr_field = pretrained_ef.model
        expr_field.eval()

        for i in range(len(slices)):
            coords_t = torch.tensor(coords_norm_list[i].astype(np.float32), device=device)
            with torch.no_grad():
                emb = expr_field.get_embedding(coords_t).cpu().numpy()
            ari = compute_ari(emb, all_labels[i], n_clusters)
            print(f"    Slice {i} ({group[i]}): ARI = {ari:.4f}  (emb dim={emb.shape[1]})")

        # ================================================================
        # 3. ExprField Canonical (batch_emb=0) vs Slice-specific
        # ================================================================
        print(f"\n  --- ExprField Slice-specific (with batch_emb) ARI ---")
        for i in range(len(slices)):
            coords_t = torch.tensor(coords_norm_list[i].astype(np.float32), device=device)
            slice_ids = torch.full((coords_t.shape[0],), i, dtype=torch.long, device=device)
            with torch.no_grad():
                # Slice-specific: use actual batch embedding
                pe = expr_field.encoder(coords_t, None)
                b = expr_field.batch_emb(slice_ids)
                h = expr_field.backbone(torch.cat([pe, b], dim=-1))
                emb_specific = expr_field.bottleneck(h).cpu().numpy()
            ari = compute_ari(emb_specific, all_labels[i], n_clusters)
            print(f"    Slice {i} ({group[i]}): ARI = {ari:.4f}")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()

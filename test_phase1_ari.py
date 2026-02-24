#!/usr/bin/env python3
"""Test Phase 1 INR embedding quality (ARI/NMI) — dual INR, shared decoder.

Trains two independent ExprINRs (one per slice) with a shared decoder via
reconstruction loss, then evaluates the learned embeddings via clustering.

The shared decoder forces both INRs' embedding spaces to align — the same
decoder must reconstruct HVG expression from either INR's output.

Usage:
    python test_phase1_ari.py
    python test_phase1_ari.py --sample_groups 0 --dlpfc_pairs "0,2"
    python test_phase1_ari.py --emb_dim 128 --pretrain_epochs 2000
"""
import argparse
import gc
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA as skPCA
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, os.path.dirname(__file__))

from inr_align.config import DLPFC_SAMPLE_GROUPS, PipelineConfig, JointConfig
from inr_align.model import ExprINR, ExprDecoder, build_joint_models
from inr_align.loss import recon_loss_from_emb
from inr_align.utils import normalize_coordinates

import spateo as st


# ============================================================================
# Clustering (same as benchmark_insta.py)
# ============================================================================

def _mclust_R(emb, n_clusters):
    """Run R mclust via rpy2, fallback to sklearn GMM."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        ro.r.library("mclust")
        r_emb = numpy2ri.py2rpy(emb)
        ro.r.assign("data", r_emb)
        ro.r(f'result <- Mclust(data, G={n_clusters}, modelNames="EEE")')
        pred = np.array(ro.r('result$classification')).astype(int)
        numpy2ri.deactivate()
        return pred
    except Exception:
        gm = GaussianMixture(
            n_components=n_clusters, covariance_type="tied",
            reg_covar=1e-4, init_params="kmeans", random_state=42,
        )
        return gm.fit_predict(emb) + 1


def compute_ari_nmi(embeddings, labels, n_clusters=7, method="mclust",
                    spatial_coords=None, refine_radius=None):
    """Cluster embeddings and compute ARI + NMI."""
    import anndata as ad
    import scipy.sparse as sp

    valid = np.array([str(l).strip() not in ("", "nan") for l in labels])
    emb = embeddings[valid]
    lab = labels[valid]
    unique_labels = np.unique(lab)
    if len(unique_labels) < 2:
        return 0.0, 0.0
    n_c = min(n_clusters, len(unique_labels))

    if method == "mclust":
        n_pca = min(20, emb.shape[1], emb.shape[0] - 1)
        if emb.shape[1] > n_pca:
            emb_for_clust = skPCA(n_components=n_pca, random_state=42).fit_transform(emb)
        else:
            emb_for_clust = emb
        pred = _mclust_R(emb_for_clust, n_c)
    elif method == "kmeans":
        pred = KMeans(n_clusters=n_c, random_state=42, n_init=10).fit_predict(emb)
    else:
        raise ValueError(f"Unknown method: {method}")

    ari = adjusted_rand_score(lab, pred)
    nmi = normalized_mutual_info_score(lab, pred, average_method="arithmetic")
    return ari, nmi


# ============================================================================
# Phase 1 training — dual INR, shared decoder
# ============================================================================

def train_phase1_dual(
    coords_norm_s1: np.ndarray,
    coords_norm_s2: np.ndarray,
    hvg_s1: torch.Tensor,
    hvg_s2: torch.Tensor,
    jcfg: JointConfig,
    device: str,
    print_every: int = 100,
):
    """Train two ExprINRs + shared Decoder on both slices (Phase 1).

    Each INR learns its own slice's spatial->embedding mapping.
    The shared decoder forces the two embedding spaces to be compatible.

    Returns:
        (expr_inr_s1, expr_inr_s2, decoder) — trained models.
    """
    models = build_joint_models(jcfg, device=device)
    expr_inr_s1 = models["expr_inr_s1"]
    expr_inr_s2 = models["expr_inr_s2"]
    decoder = models["decoder"]

    N1, N2 = coords_norm_s1.shape[0], coords_norm_s2.shape[0]
    x1 = torch.tensor(coords_norm_s1.astype(np.float32), device=device)
    x2 = torch.tensor(coords_norm_s2.astype(np.float32), device=device)
    batch_size = min(4096, max(N1, N2))
    n_freqs = expr_inr_s1.n_freqs
    epochs = jcfg.inr_pretrain_epochs

    # Single optimizer for both INRs + shared decoder
    optimizer = torch.optim.Adam(
        list(expr_inr_s1.parameters()) +
        list(expr_inr_s2.parameters()) +
        list(decoder.parameters()),
        lr=jcfg.inr_pretrain_lr,
    )

    best_recon = float("inf")
    best_states = {}
    patience_counter = 0
    patience_limit = 300
    warmup = max(epochs // 3, 1)

    print(f"\n  Phase 1 Pretrain (dual INR + shared decoder):")
    print(f"    epochs={epochs}, emb_dim={jcfg.emb_dim}, "
          f"hidden={jcfg.inr_hidden}, layers={jcfg.inr_layers}, "
          f"decoder_layers={jcfg.decoder_layers}, batch={batch_size}")
    print(f"    s1: {N1} spots, s2: {N2} spots")

    for ep in range(epochs):
        alpha = n_freqs * (ep / warmup) if ep < warmup else float(n_freqs)
        alpha_t = torch.tensor(alpha, device=device)

        # Sample batches from both slices
        idx1 = torch.randint(0, N1, (batch_size,), device=device)
        idx2 = torch.randint(0, N2, (batch_size,), device=device)

        optimizer.zero_grad(set_to_none=True)

        # Forward: each INR encodes its own slice, shared decoder reconstructs
        emb1 = expr_inr_s1(x1[idx1], alpha_t)
        emb2 = expr_inr_s2(x2[idx2], alpha_t)

        L_recon1 = recon_loss_from_emb(emb1, decoder, hvg_s1[idx1])
        L_recon2 = recon_loss_from_emb(emb2, decoder, hvg_s2[idx2])

        loss = (L_recon1 + L_recon2) / 2.0
        loss.backward()

        torch.nn.utils.clip_grad_norm_(expr_inr_s1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(expr_inr_s2.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()

        recon_val = loss.item()
        r1_val = L_recon1.item()
        r2_val = L_recon2.item()

        if ep >= warmup:
            if recon_val < best_recon:
                best_recon = recon_val
                best_states = {
                    "expr_inr_s1": {k: v.cpu().clone() for k, v in expr_inr_s1.state_dict().items()},
                    "expr_inr_s2": {k: v.cpu().clone() for k, v in expr_inr_s2.state_dict().items()},
                    "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1

        if ep % print_every == 0 or ep == epochs - 1:
            marker = " *" if recon_val <= best_recon else ""
            print(f"    ep={ep:04d} | recon={recon_val:.5f} (s1={r1_val:.5f} s2={r2_val:.5f}){marker}")

        if ep >= warmup and patience_counter >= patience_limit:
            print(f"    Early stopping at ep={ep} (no improvement for {patience_limit} epochs)")
            break

    if best_states:
        expr_inr_s1.load_state_dict({k: v.to(device) for k, v in best_states["expr_inr_s1"].items()})
        expr_inr_s2.load_state_dict({k: v.to(device) for k, v in best_states["expr_inr_s2"].items()})
        decoder.load_state_dict({k: v.to(device) for k, v in best_states["decoder"].items()})
        print(f"  Phase 1 done. Best recon: {best_recon:.5f} (restored)")
    else:
        print(f"  Phase 1 done. Final recon: {recon_val:.5f}")

    return expr_inr_s1, expr_inr_s2, decoder


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Phase 1 dual-INR embedding ARI")
    parser.add_argument("--sample_groups", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dlpfc_pairs", nargs="+", default=None,
                        help="e.g. '0,2' '0,1' '0,2'")
    parser.add_argument("--pretrain_epochs", type=int, default=None,
                        help="Override inr_pretrain_epochs (default: use config = 2000)")
    parser.add_argument("--emb_dim", type=int, default=None,
                        help="Override emb_dim (default: use config = 64)")
    parser.add_argument("--clustering", nargs="+", default=["mclust"],
                        choices=["mclust", "kmeans"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = PipelineConfig(data_dir="./Data")
    jcfg = config.joint

    if args.pretrain_epochs is not None:
        jcfg.inr_pretrain_epochs = args.pretrain_epochs
    if args.emb_dim is not None:
        jcfg.emb_dim = args.emb_dim

    print(f"\nConfig: emb_dim={jcfg.emb_dim}, "
          f"hidden={jcfg.inr_hidden}, layers={jcfg.inr_layers}, "
          f"epochs={jcfg.inr_pretrain_epochs}")

    # Parse dlpfc_pairs
    dlpfc_pairs = None
    if args.dlpfc_pairs:
        dlpfc_pairs = [[int(x) for x in p.split(",")] for p in args.dlpfc_pairs]

    all_rows = []

    for gi, idx in enumerate(args.sample_groups):
        group_full = DLPFC_SAMPLE_GROUPS[idx]
        folder = f"DLPFC_sample{idx + 1}"
        ds_name = f"DLPFC_sample{idx + 1}"

        if dlpfc_pairs and gi < len(dlpfc_pairs):
            sel = dlpfc_pairs[gi]
            group = [group_full[i] for i in sel]
        else:
            group = list(group_full)

        print(f"\n{'='*70}")
        print(f"  {ds_name}: {group}")
        print(f"{'='*70}")

        # Load and preprocess slices
        slices_raw = []
        for sample_id in group:
            path = f"./Data/{folder}/original_data/{sample_id}.h5ad"
            adata = sc.read_h5ad(path)
            slices_raw.append(adata)
            print(f"  Loaded {sample_id}: {adata.shape}")

        slices = []
        for adata in slices_raw:
            a = adata.copy()
            if "counts" not in a.layers:
                a.layers["counts"] = a.X.copy()
            sc.pp.normalize_total(a)
            sc.pp.log1p(a)
            if "highly_variable" not in a.var.columns:
                sc.pp.highly_variable_genes(a, n_top_genes=2000)
            slices.append(a)

        # Group PCA (for PCA baseline)
        st.align.group_pca(slices, pca_key="X_pca")

        n_clusters = 7  # DLPFC has 7 layers

        # ---- PCA baseline ----
        print("\n  --- PCA baseline ---")
        for clust_method in args.clustering:
            for i, s in enumerate(slices):
                pca_emb = s.obsm["X_pca"].copy()
                labels = np.asarray(s.obs["original_domain"])
                ari, nmi = compute_ari_nmi(pca_emb, labels, n_clusters, method=clust_method)
                print(f"    {group[i]} PCA [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
                all_rows.append({
                    "Dataset": ds_name, "Slice": group[i],
                    "Method": "PCA", "Clustering": clust_method,
                    "ARI": ari, "NMI": nmi,
                })

        # ---- Prepare HVG data for both slices ----
        import scipy.sparse as sp

        # Normalize coordinates using all slices in the pair
        all_coords = [s.obsm["spatial"] for s in slices]
        coords_norm_list, mean, std = normalize_coordinates(all_coords)

        # HVG expression for both slices (use intersection of HVG)
        hvg_mask_s1 = slices[0].var["highly_variable"].values
        hvg_mask_s2 = slices[1].var["highly_variable"].values
        # Use s1's HVG mask for consistency (both slices have same var after preprocessing)
        hvg_mask = hvg_mask_s1

        hvg_tensors = []
        for i, s in enumerate(slices):
            X_hvg = s.X[:, hvg_mask]
            if sp.issparse(X_hvg):
                X_hvg = X_hvg.toarray()
            hvg_tensors.append(torch.tensor(X_hvg.astype(np.float32), device=device))

        # Set n_output to actual HVG count
        jcfg.n_output = hvg_tensors[0].shape[1]

        print(f"\n  --- Training dual INR (Phase 1) on {group[0]} + {group[1]} ---")
        print(f"      n_hvg={jcfg.n_output}, s1={hvg_tensors[0].shape[0]} spots, "
              f"s2={hvg_tensors[1].shape[0]} spots")

        expr_inr_s1, expr_inr_s2, decoder = train_phase1_dual(
            coords_norm_list[0], coords_norm_list[1],
            hvg_tensors[0], hvg_tensors[1],
            jcfg, device,
        )

        # ---- Evaluate INR embedding on both slices ----
        print(f"\n  --- Evaluating INR embedding ---")
        inr_models = [expr_inr_s1, expr_inr_s2]
        for m in inr_models:
            m.eval()
        decoder.eval()

        for i, s in enumerate(slices):
            coords_i = coords_norm_list[i]
            inr_i = inr_models[i]
            with torch.no_grad():
                coords_t = torch.tensor(coords_i.astype(np.float32), device=device)
                alpha = torch.tensor(float(inr_i.n_freqs), device=device)
                inr_emb = inr_i(coords_t, alpha).cpu().numpy()

            labels = np.asarray(s.obs["original_domain"])
            for clust_method in args.clustering:
                ari, nmi = compute_ari_nmi(inr_emb, labels, n_clusters, method=clust_method)
                print(f"    {group[i]} INR_s{i+1} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
                all_rows.append({
                    "Dataset": ds_name, "Slice": group[i],
                    "Method": f"INR_s{i+1}_phase1", "Clustering": clust_method,
                    "ARI": ari, "NMI": nmi,
                })

        # ---- Also evaluate decoder reconstruction as embedding ----
        print(f"\n  --- Evaluating Decoder recon as embedding ---")
        for i, s in enumerate(slices):
            coords_i = coords_norm_list[i]
            inr_i = inr_models[i]
            with torch.no_grad():
                coords_t = torch.tensor(coords_i.astype(np.float32), device=device)
                alpha = torch.tensor(float(inr_i.n_freqs), device=device)
                inr_emb = inr_i(coords_t, alpha)
                recon_expr = decoder(inr_emb).cpu().numpy()

            labels = np.asarray(s.obs["original_domain"])
            for clust_method in args.clustering:
                ari, nmi = compute_ari_nmi(recon_expr, labels, n_clusters, method=clust_method)
                print(f"    {group[i]} Recon_s{i+1} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
                all_rows.append({
                    "Dataset": ds_name, "Slice": group[i],
                    "Method": f"INR_recon_s{i+1}", "Clustering": clust_method,
                    "ARI": ari, "NMI": nmi,
                })

        # ---- Cross-slice: evaluate s2 coords with s1's INR (test alignment) ----
        print(f"\n  --- Cross-slice: s1 INR on s2 coords (embedding alignment test) ---")
        for i, s in enumerate(slices):
            coords_i = coords_norm_list[i]
            # Always use s1's INR for cross-slice test
            with torch.no_grad():
                coords_t = torch.tensor(coords_i.astype(np.float32), device=device)
                alpha = torch.tensor(float(expr_inr_s1.n_freqs), device=device)
                cross_emb = expr_inr_s1(coords_t, alpha).cpu().numpy()

            labels = np.asarray(s.obs["original_domain"])
            for clust_method in args.clustering:
                ari, nmi = compute_ari_nmi(cross_emb, labels, n_clusters, method=clust_method)
                tag = "same" if i == 0 else "cross"
                print(f"    {group[i]} INR_s1({tag}) [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
                all_rows.append({
                    "Dataset": ds_name, "Slice": group[i],
                    "Method": f"INR_s1_{tag}", "Clustering": clust_method,
                    "ARI": ari, "NMI": nmi,
                })

        # Cleanup
        del expr_inr_s1, expr_inr_s2, decoder
        for t in hvg_tensors:
            del t
        torch.cuda.empty_cache()
        gc.collect()

    # ---- Print summary table ----
    df = pd.DataFrame(all_rows)
    print("\n" + "=" * 80)
    print("Summary: Phase 1 Dual-INR Embedding Quality")
    print("=" * 80)

    for ds in df["Dataset"].unique():
        sub = df[df["Dataset"] == ds]
        print(f"\n  {ds}:")
        for clust in df["Clustering"].unique():
            csub = sub[sub["Clustering"] == clust]
            print(f"    [{clust}]")
            for method in csub["Method"].unique():
                msub = csub[csub["Method"] == method]
                for _, row in msub.iterrows():
                    print(f"      {row['Slice']:>8s}  {method:<20s}  ARI={row['ARI']:.4f}  NMI={row['NMI']:.4f}")
            # Print mean per method
            print(f"    {'--- Mean ---':>8s}")
            for method in csub["Method"].unique():
                msub = csub[csub["Method"] == method]
                print(f"      {'mean':>8s}  {method:<20s}  ARI={msub['ARI'].mean():.4f}  NMI={msub['NMI'].mean():.4f}")

    # Save
    df.to_csv("phase1_ari_test.csv", index=False)
    print(f"\nResults saved to phase1_ari_test.csv")


if __name__ == "__main__":
    main()

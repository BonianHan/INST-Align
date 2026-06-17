#!/usr/bin/env python3
"""Run the paper embedding evaluation (ARI/NMI).

Metrics:
  Embedding: ARI, NMI (PCA / Seurat / STAligner / GraphST / SPIRAL / INR)

Usage:
    conda activate spateo
    python run_embedding.py [--sample_groups 0 1 2]
"""
import argparse
import gc
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
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA as skPCA


def _free_gpu():
    """Release GPU memory and trigger garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from insta.spatial_alignment import (
    DLPFC_LABEL_MAP,
    _load_dlpfc_layer_groups,
    print_summary,
    run_ours,
    run_spatial_alignment_all,
)
from insta.config import DLPFC_SAMPLE_GROUPS, PipelineConfig


# ============================================================================
# ARI / NMI helpers
# ============================================================================


_MCLUST_LOADED = False


def _mclust_R(embedding, n_clusters, model_names="EEE", random_seed=666):
    """Run R mclust on embedding. Returns integer cluster labels (0-indexed).

    Falls back to sklearn GaussianMixture if R/rpy2/mclust not available.
    """
    global _MCLUST_LOADED
    try:
        import rpy2.robjects as robjects
        if not _MCLUST_LOADED:
            robjects.r.library("mclust")
            _MCLUST_LOADED = True

        # Convert numpy array to R matrix (column-major / Fortran order)
        nr, nc = embedding.shape
        r_vec = robjects.FloatVector(embedding.flatten(order='F'))
        r_mat = robjects.r['matrix'](r_vec, nrow=nr, ncol=nc)
        robjects.globalenv['emb.mat'] = r_mat
        robjects.globalenv['nc.val'] = robjects.IntVector([n_clusters])
        robjects.globalenv['mn.val'] = robjects.StrVector([model_names])
        robjects.r('set.seed(%d)' % random_seed)
        res = robjects.r('Mclust(emb.mat, G=nc.val, modelNames=mn.val)')
        if res == robjects.NULL:
            raise RuntimeError("mclust returned NULL (model did not converge)")
        labels = np.array(res.rx2('classification')).astype(int) - 1
        return labels
    except Exception as e:
        print(f"    [mclust fallback to sklearn GMM: {e}]")
        # Fallback: sklearn GaussianMixture (tied covariance ≈ EEE)
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type="tied",
            n_init=10, random_state=random_seed, max_iter=300,
        )
        return gmm.fit_predict(embedding)


def _spatial_refine(pred, spatial_coords, n_clusters, radius):
    """Spatial refinement: majority voting among neighbors within radius."""
    from sklearn.neighbors import BallTree
    tree = BallTree(spatial_coords)
    new_pred = pred.copy()
    for i in range(len(spatial_coords)):
        neighbors = tree.query_radius(spatial_coords[i:i + 1], r=radius)[0]
        if len(neighbors) > 1:
            nl = pred[neighbors]
            counts = np.bincount(nl, minlength=n_clusters)
            new_pred[i] = np.argmax(counts)
    return new_pred


def _leiden_target_k(adata, target_k, lo=0.05, hi=3.0, max_iter=30):
    """Binary search for Leiden resolution that produces target_k clusters."""
    best_res = 1.0
    best_diff = 999
    best_labels = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        sc.tl.leiden(adata, resolution=mid, key_added="_leiden_tmp")
        k = adata.obs["_leiden_tmp"].nunique()
        if abs(k - target_k) < best_diff:
            best_diff = abs(k - target_k)
            best_res = mid
            best_labels = adata.obs["_leiden_tmp"].values.copy()
        if k == target_k:
            return best_labels, mid
        elif k > target_k:
            hi = mid
        else:
            lo = mid
    return best_labels, best_res


def compute_ari_nmi(embeddings, labels, n_clusters=7, method="mclust",
                    spatial_coords=None, refine_radius=None):
    """Cluster embeddings and compute ARI + NMI against ground truth.

    Args:
        embeddings: np.ndarray (n_cells, n_features).
        labels: array-like ground truth labels.
        n_clusters: number of clusters.
        method: "mclust" (R mclust EEE, or sklearn GMM fallback),
                "kmeans" (legacy),
                "leiden" (scanpy Leiden, expression neighbors only),
                "leiden_spatial" (Leiden, fusing expression + spatial graph).
        spatial_coords: optional (n_cells, 2) for spatial refinement / fusion.
        refine_radius: if set, apply spatial refinement (majority voting among
                       neighbors within this radius). NOTE: radius must match
                       the coordinate scale of spatial_coords.

    Returns:
        (ari, nmi) tuple.
    """
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
        # PCA to 20 dims (standard protocol: STAligner/GraphST/BenchmarkST)
        n_pca = min(20, emb.shape[1], emb.shape[0] - 1)
        if emb.shape[1] > n_pca:
            emb_for_clust = skPCA(n_components=n_pca, random_state=42).fit_transform(emb)
        else:
            emb_for_clust = emb
        pred = _mclust_R(emb_for_clust, n_c)
    elif method in ("leiden", "leiden_spatial"):
        # Build expression neighbor graph
        a_tmp = ad.AnnData(np.zeros((emb.shape[0], 1)))
        a_tmp.obsm["X_emb"] = emb.copy()
        sc.pp.neighbors(a_tmp, use_rep="X_emb", n_neighbors=15)

        if method == "leiden_spatial" and spatial_coords is not None:
            # Build spatial neighbor graph and fuse with expression graph
            coords_v = spatial_coords[valid]
            from sklearn.neighbors import NearestNeighbors as _NN
            nn_sp = _NN(n_neighbors=7).fit(coords_v)
            dist_s, idx_s = nn_sp.kneighbors(coords_v)
            n = len(coords_v)
            rows_s, cols_s, vals_s = [], [], []
            for i in range(n):
                for j in range(1, idx_s.shape[1]):
                    rows_s.append(i)
                    cols_s.append(idx_s[i, j])
                    vals_s.append(1.0 / (dist_s[i, j] + 1e-6))
            adj_spatial = sp.csr_matrix((vals_s, (rows_s, cols_s)), shape=(n, n))
            adj_spatial = adj_spatial + adj_spatial.T

            # Fuse: expression + spatial (weight=1.0)
            adj_expr = a_tmp.obsp["connectivities"]
            a_tmp.obsp["connectivities"] = adj_expr + adj_spatial
            a_tmp.obsp["distances"] = a_tmp.obsp["connectivities"].copy()

        pred, _ = _leiden_target_k(a_tmp, n_c)
        pred = np.asarray(pred)
    else:
        km = KMeans(n_clusters=n_c, n_init=10, random_state=42)
        pred = km.fit_predict(emb)

    # Optional spatial refinement (for mclust/kmeans)
    if refine_radius is not None and spatial_coords is not None and method not in ("leiden_spatial",):
        coords_valid = spatial_coords[valid]
        pred = _spatial_refine(pred, coords_valid, n_c, refine_radius)

    ari = adjusted_rand_score(lab, pred)
    nmi = normalized_mutual_info_score(lab, pred)
    return ari, nmi


# ============================================================================
# Integration method runners (STAligner / GraphST / SPIRAL)
# ============================================================================


def run_staligner_embedding(slices_raw, label_key="original_domain",
                            rad_cutoff=150, knn_neigh=50, n_hvg=5000,
                            device="cuda"):
    """Run STAligner integration and return per-cell embeddings.

    Args:
        slices_raw: list of AnnData (raw counts, NOT preprocessed).

    Returns:
        list of np.ndarray embeddings, one per slice (shape n_cells x 30).
    """
    import STAligner
    import anndata as ad
    import scipy.sparse as sp
    import scipy.linalg

    Batch_list = []
    adj_list = []

    for i, adata in enumerate(slices_raw):
        a = adata.copy()
        a.var_names_make_unique()
        a.obs_names = [f"s{i}_{x}" for x in a.obs_names]

        # Ensure sparse X (STAligner requires it)
        if not sp.issparse(a.X):
            a.X = sp.csr_matrix(a.X)

        # Build spatial neighbor graph
        STAligner.Cal_Spatial_Net(a, rad_cutoff=rad_cutoff)

        # HVG + normalize
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_hvg)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a = a[:, a.var["highly_variable"]]

        if not sp.issparse(a.X):
            a.X = sp.csr_matrix(a.X)

        adj_list.append(a.uns["adj"])
        Batch_list.append(a)

    # Concatenate
    section_ids = [f"slice_{i}" for i in range(len(slices_raw))]
    adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype("category")
    adata_concat.obs_names_make_unique()

    if not sp.issparse(adata_concat.X):
        adata_concat.X = sp.csr_matrix(adata_concat.X)

    # Block-diagonal adjacency
    adj_concat = np.asarray(adj_list[0].todense())
    for bid in range(1, len(section_ids)):
        adj_concat = scipy.linalg.block_diag(
            adj_concat, np.asarray(adj_list[bid].todense())
        )
    adata_concat.uns["edgeList"] = np.nonzero(adj_concat)

    # Train
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    adata_concat = STAligner.train_STAligner(
        adata_concat, verbose=True, knn_neigh=knn_neigh, device=dev,
    )

    # Split embeddings back per slice
    emb_all = adata_concat.obsm["STAligner"]
    slice_sizes = [len(b) for b in Batch_list]
    embeddings = []
    offset = 0
    for sz in slice_sizes:
        embeddings.append(emb_all[offset: offset + sz])
        offset += sz

    return embeddings


def run_stagate_embedding(slices_raw, label_key="original_domain",
                          rad_cutoff=150, n_hvg=5000, n_epochs=500,
                          device="cuda"):
    """Run STAGATE independently on each slice and return per-cell embeddings.

    Unlike STAligner, STAGATE does NOT do cross-slice alignment — it only
    learns a spatial-aware autoencoder embedding per slice.

    Returns:
        list of np.ndarray embeddings, one per slice (shape n_cells x 30).
    """
    import STAligner
    import scipy.sparse as sp

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    embeddings = []

    for i, adata in enumerate(slices_raw):
        a = adata.copy()
        a.var_names_make_unique()

        if not sp.issparse(a.X):
            a.X = sp.csr_matrix(a.X)

        # Build spatial graph
        STAligner.Cal_Spatial_Net(a, rad_cutoff=rad_cutoff)

        # HVG + normalize
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_hvg)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a = a[:, a.var["highly_variable"]]

        if not sp.issparse(a.X):
            a.X = sp.csr_matrix(a.X)

        # Train STAGATE (single slice, no cross-slice alignment)
        print(f"      STAGATE: training slice {i} ({a.n_obs} cells)...")
        a = STAligner.train_STAGATE(a, hidden_dims=[512, 30],
                                     n_epochs=n_epochs, device=dev)
        embeddings.append(a.obsm["STAligner"].copy())
        _free_gpu()

    return embeddings


def run_graphst_embedding(slices_raw, label_key="original_domain",
                          n_hvg=3000, epochs=600, device="cuda"):
    """Run GraphST integration and return per-cell embeddings.

    Args:
        slices_raw: list of AnnData (raw counts, NOT preprocessed).

    Returns:
        list of np.ndarray embeddings, one per slice (shape n_cells x 64).
    """
    from GraphST.GraphST import GraphST as GraphSTModel
    import anndata as ad
    import scipy.sparse as sp

    slice_sizes = []
    adata_list = []
    for i, adata in enumerate(slices_raw):
        a = adata.copy()
        a.var_names_make_unique()
        a.obs["batch"] = str(i)
        # Offset spatial coords so slices don't overlap in KNN graph
        coords = a.obsm["spatial"].copy().astype(np.float64)
        coords[:, 1] += i * 10000.0  # large Y offset per slice
        a.obsm["spatial"] = coords
        adata_list.append(a)
        slice_sizes.append(a.n_obs)

    adata_concat = ad.concat(adata_list, merge="same")
    adata_concat.obs_names_make_unique()

    if not sp.issparse(adata_concat.X):
        adata_concat.X = sp.csr_matrix(adata_concat.X)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = GraphSTModel(adata_concat, device=dev, epochs=epochs,
                         dim_input=n_hvg, dim_output=64)
    adata_concat = model.train()

    # GraphST paper uses obsm['emb'] (reconstruction, dim=n_hvg) for clustering.
    # The clustering pipeline: emb → PCA(20) → mclust → spatial refinement.
    # We return the reconstruction embedding; PCA(20) is applied in compute_ari_nmi.
    emb_all = adata_concat.obsm["emb"]
    embeddings = []
    offset = 0
    for sz in slice_sizes:
        embeddings.append(emb_all[offset: offset + sz])
        offset += sz

    return embeddings


def _run_spiral_pair(pair_slices, label_key, n_hvg, knn, epochs, zdim,
                     znoise_dim, device, max_seconds=300):
    """Run SPIRAL on a pair of slices (2 slices only)."""
    import tempfile
    import argparse
    import shutil
    import time as _time
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors
    from spiral.main import SPIRAL_integration
    from spiral.layers import MeanAggregator
    from spiral.utils import layer_map

    tmpdir = tempfile.mkdtemp()

    # Preprocess
    processed = []
    for adata in pair_slices:
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_hvg)
        processed.append(a)

    # Union of HVGs
    hvg_union = set()
    for a in processed:
        hvg_union |= set(a.var_names[a.var["highly_variable"]])
    hvg_list = sorted(hvg_union)
    N = len(hvg_list)

    feat_files, edge_files, meta_files = [], [], []
    slice_sizes = []

    for i, (a, a_raw) in enumerate(zip(processed, pair_slices)):
        name = f"slice{i}"
        cells = [f"{name}_{j}" for j in range(a.n_obs)]
        slice_sizes.append(a.n_obs)

        Xfull = a.X.toarray() if sp.issparse(a.X) else np.array(a.X)
        mat = pd.DataFrame(0.0, index=cells, columns=hvg_list)
        common = [g for g in hvg_list if g in a.var_names]
        if common:
            col_idx = [list(a.var_names).index(g) for g in common]
            mat.loc[:, common] = Xfull[:, col_idx]

        feat_path = os.path.join(tmpdir, f"{name}_feat.csv")
        mat.to_csv(feat_path)
        feat_files.append(feat_path)

        lab_vals = np.asarray(a.obs[label_key]) if label_key in a.obs.columns else ["unknown"] * a.n_obs
        meta = pd.DataFrame({"celltype": lab_vals, "batch": name}, index=cells)
        meta_path = os.path.join(tmpdir, f"{name}_meta.csv")
        meta.to_csv(meta_path)
        meta_files.append(meta_path)

        coords = a.obsm["spatial"]
        nbrs = NearestNeighbors(n_neighbors=min(knn + 1, len(coords))).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        edges = []
        for ci in range(len(coords)):
            for j in indices[ci, 1:]:
                edges.append([cells[ci], cells[j]])
        edge_path = os.path.join(tmpdir, f"{name}_edges.txt")
        np.savetxt(edge_path, edges, fmt="%s")
        edge_files.append(edge_path)

    # Configure SPIRAL (always binary for pairs)
    M = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--AEdims", type=list, default=[N, [512], zdim])
    parser.add_argument("--AEdimsR", type=list, default=[zdim, [512], N])
    parser.add_argument("--GSdims", type=list, default=[512, zdim])
    parser.add_argument("--zdim", type=int, default=zdim)
    parser.add_argument("--znoise_dim", type=int, default=znoise_dim)
    parser.add_argument("--CLdims", type=list, default=[znoise_dim, [], M])
    parser.add_argument("--DIdims", type=list, default=[zdim - znoise_dim, [32, 16], M])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--agg_class", default=MeanAggregator)
    parser.add_argument("--num_samples", default=knn)
    parser.add_argument("--N_WALKS", type=int, default=knn)
    parser.add_argument("--WALK_LEN", type=int, default=1)
    parser.add_argument("--N_WALK_LEN", type=int, default=knn)
    parser.add_argument("--NUM_NEG", type=int, default=knn)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--alpha1", type=float, default=float(N))
    parser.add_argument("--alpha2", type=float, default=1.0)
    parser.add_argument("--alpha3", type=float, default=1.0)
    parser.add_argument("--alpha4", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--Q", type=float, default=10.0)
    params, _ = parser.parse_known_args([])

    # Train with time-based early stopping
    SPII = SPIRAL_integration(params, feat_files, edge_files, meta_files)
    from torch.utils.data import DataLoader as DL
    import torch.nn as _nn
    SPII.data_loader = DL(
        dataset=SPII.dataset, batch_size=SPII.BS,
        shuffle=True, num_workers=0, drop_last=True,
    )
    # Inline training loop (mirrors SPIRAL_integration.train) with timeout
    _t0 = _time.time()
    SPII.model.train()
    completed_epochs = 0
    for _ep in range(SPII.epochs):
        total_loss = 0.0; AE_loss = 0.0; GS_loss = 0.0; CLAS_loss = 0; DISC_loss = 0
        IDX = []
        for (batch_idx, target_idx) in enumerate(SPII.data_loader):
            if len(np.unique(np.array(IDX))) == SPII.feat.shape[0]:
                break
            target_idx = target_idx[0]
            all_idx = np.asarray(list(SPII.unsupervised_loss.extend_nodes(target_idx.tolist())))
            IDX = IDX + all_idx.tolist()
            all_layer, all_mapping = layer_map(all_idx.tolist(), SPII.adj, len(SPII.params.GSdims))
            all_rows = SPII.adj.tolil().rows[all_layer[0]]
            all_feature = SPII.feat1[all_layer[0], :]
            all_embed, ae_out, clas_out, disc_out = SPII.model(
                all_feature, all_layer, all_mapping, all_rows,
                SPII.params.lamda, SPII.de_act, SPII.cl_act,
            )
            [ae_embed, gs_embed, embed] = all_embed
            [x_bar, x] = ae_out
            gs_loss = SPII.unsupervised_loss.get_loss_xent(embed, all_idx)
            ae_loss = _nn.BCELoss()(x_bar, x)
            if SPII.sample_num == 2:
                true_batch = SPII.Y1[all_layer[-1]]
                clas_loss = _nn.BCELoss()(clas_out, true_batch.reshape(-1, 1))
                disc_loss = _nn.BCELoss()(disc_out, true_batch.reshape(-1, 1))
            else:
                true_batch = SPII.Y1[all_layer[-1]].long()
                clas_loss = _nn.CrossEntropyLoss()(clas_out, true_batch)
                disc_loss = _nn.CrossEntropyLoss()(disc_out, true_batch)
            loss = (ae_loss * SPII.params.alpha1 + gs_loss * SPII.params.alpha2
                    + clas_loss * SPII.params.alpha3 + disc_loss * SPII.params.alpha4)
            SPII.optim.zero_grad()
            loss.backward()
            SPII.optim.step()
            total_loss += loss.item()
            AE_loss += ae_loss.item()
            GS_loss += gs_loss.item()
            CLAS_loss += clas_loss.item()
            DISC_loss += disc_loss.item()
        aa = batch_idx + 1
        completed_epochs += 1
        elapsed = _time.time() - _t0
        print(f'      epoch {_ep+1}/{epochs} ({elapsed:.0f}s) '
              f'loss={total_loss/aa:.4f}')
        if elapsed >= max_seconds:
            print(f'      SPIRAL early stop: {elapsed:.0f}s >= {max_seconds}s '
                  f'(completed {completed_epochs}/{epochs} epochs)')
            break

    # Extract embeddings
    SPII.model.eval()
    all_idx = np.arange(SPII.feat.shape[0])
    all_layer, all_mapping = layer_map(
        all_idx.tolist(), SPII.adj, len(params.GSdims)
    )
    all_rows = SPII.adj.tolil().rows[all_layer[0]]
    all_feature = torch.Tensor(
        SPII.feat.iloc[all_layer[0], :].values
    ).float().to(device)

    with torch.no_grad():
        all_embed, _, _, _ = SPII.model(
            all_feature, all_layer, all_mapping, all_rows,
            params.lamda, SPII.de_act, SPII.cl_act,
        )
    embed = all_embed[2].cpu().numpy()
    bio_embed = embed[:, znoise_dim:]  # biological dims only

    # Split per slice
    pair_embeddings = []
    offset = 0
    for sz in slice_sizes:
        pair_embeddings.append(bio_embed[offset: offset + sz])
        offset += sz

    shutil.rmtree(tmpdir, ignore_errors=True)
    _free_gpu()
    return pair_embeddings


def run_seurat_embedding(slices_raw, label_key="original_domain",
                          n_hvg=3000, n_pcs=30):
    """Run Seurat v3 CCA integration via rpy2 and return per-cell embeddings.

    Uses FindIntegrationAnchors + IntegrateData (CCA-based), then PCA on the
    integrated assay.  Requires R with Seurat installed and rpy2 in Python.

    Args:
        slices_raw: list of AnnData (raw counts, NOT preprocessed).
        n_hvg: number of highly variable genes per slice.
        n_pcs: number of PCs / CCA dims.

    Returns:
        list of np.ndarray embeddings, one per slice (shape n_cells x n_pcs).
    """
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import localconverter
    import scipy.sparse as sp

    conv = ro.default_converter + numpy2ri.converter

    n_slices = len(slices_raw)
    slice_sizes = [adata.n_obs for adata in slices_raw]

    # ---- Pre-filter to union of HVGs in Python to save memory ----
    hvg_sets = []
    for adata in slices_raw:
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_hvg)
        hvg_sets.append(set(a.var_names[a.var["highly_variable"]]))
    hvg_union = sorted(hvg_sets[0].union(*hvg_sets[1:]))
    print(f"      Seurat: pre-filtered to {len(hvg_union)} union HVGs")

    with localconverter(conv):
        ro.r('suppressPackageStartupMessages(library(Seurat))')
        ro.r('options(future.globals.maxSize = 4 * 1024^3)')  # 4 GB

        # Convert each AnnData (HVG-subset, raw counts) to Seurat object in R
        for i, adata in enumerate(slices_raw):
            a_sub = adata[:, hvg_union].copy()
            X = a_sub.X.toarray() if sp.issparse(a_sub.X) else np.array(a_sub.X)
            genes = list(a_sub.var_names)
            cells = [f"s{i}_{j}" for j in range(a_sub.n_obs)]

            # Pass count matrix to R (genes x cells — Seurat convention)
            ro.globalenv[f'mat_{i}'] = ro.r['matrix'](
                ro.FloatVector(X.T.flatten(order='F')),
                nrow=len(genes), ncol=len(cells)
            )
            ro.globalenv[f'genes_{i}'] = ro.StrVector(genes)
            ro.globalenv[f'cells_{i}'] = ro.StrVector(cells)

            ro.r(f'''
            rownames(mat_{i}) <- genes_{i}
            colnames(mat_{i}) <- cells_{i}
            obj_{i} <- CreateSeuratObject(counts = mat_{i}, project = "slice{i}")
            obj_{i} <- NormalizeData(obj_{i}, verbose = FALSE)
            obj_{i} <- FindVariableFeatures(obj_{i}, nfeatures = {n_hvg},
                                            verbose = FALSE)
            rm(mat_{i}); gc()
            ''')
            print(f"      Seurat: created object for slice {i} "
                  f"({a_sub.n_obs} cells, {len(genes)} genes)")
            del X, a_sub

        # Build object list and integrate
        obj_list_str = ", ".join([f"obj_{i}" for i in range(n_slices)])
        print(f"      Seurat: finding integration anchors (dims=1:{n_pcs}) ...")
        ro.r(f'''
        obj.list <- list({obj_list_str})
        anchors <- FindIntegrationAnchors(object.list = obj.list,
                                           dims = 1:{n_pcs}, verbose = FALSE)
        integrated <- IntegrateData(anchorset = anchors,
                                     dims = 1:{n_pcs}, verbose = FALSE)
        DefaultAssay(integrated) <- "integrated"
        integrated <- ScaleData(integrated, verbose = FALSE)
        integrated <- RunPCA(integrated, npcs = {n_pcs}, verbose = FALSE)
        pca_emb <- Embeddings(integrated, "pca")
        cell_order <- colnames(integrated)
        ''')
        print("      Seurat: integration done.")

        pca_emb = np.array(ro.r('pca_emb'))
        cell_order = list(ro.r('cell_order'))

    # Map cell names back to (slice_idx, local_idx)
    cell_to_pos = {name: ci for ci, name in enumerate(cell_order)}

    embeddings = []
    for i in range(n_slices):
        sz = slice_sizes[i]
        emb = np.zeros((sz, n_pcs), dtype=np.float32)
        for j in range(sz):
            emb[j] = pca_emb[cell_to_pos[f"s{i}_{j}"]]
        embeddings.append(emb)

    # Clean up R objects
    for i in range(n_slices):
        ro.r(f'rm(genes_{i}, cells_{i}, obj_{i})')
    ro.r('rm(obj.list, anchors, integrated, pca_emb, cell_order); gc()')

    return embeddings


def run_spiral_embedding(slices_raw, label_key="original_domain",
                         n_hvg=3000, knn=6, epochs=10, zdim=32,
                         znoise_dim=4, device="cuda", max_seconds=300):
    """Run SPIRAL integration and return per-cell embeddings.

    For >2 slices, runs SPIRAL on non-overlapping pairs to keep
    runtime feasible (SPIRAL scales poorly with cell count).
    max_seconds: time budget (default 300s = 5min). Training stops
    early if exceeded, but still returns embeddings from current model.

    Args:
        slices_raw: list of AnnData (raw counts, NOT preprocessed).

    Returns:
        list of np.ndarray embeddings, one per slice (shape n_cells x zdim-znoise_dim).
    """
    import time as _time
    n = len(slices_raw)
    _t0 = _time.time()
    if n <= 2:
        # Run directly on 2 slices
        return _run_spiral_pair(
            slices_raw, label_key, n_hvg, knn, epochs, zdim, znoise_dim, device,
            max_seconds=max_seconds,
        )

    # For >2 slices, split into non-overlapping pairs: (0,1), (2,3), ...
    # If odd number, last group has 3 slices but we only pair the last 2.
    embeddings = [None] * n
    for start in range(0, n, 2):
        end = min(start + 2, n)
        if end - start < 2:
            # Odd slice out — pair with the previous slice
            start = start - 1
            end = start + 2
        # Budget remaining time for this pair
        remaining = max(30, max_seconds - (_time.time() - _t0))
        pair = slices_raw[start:end]
        print(f"      SPIRAL: training pair ({start}, {end - 1}), budget {remaining:.0f}s...")
        pair_embs = _run_spiral_pair(
            pair, label_key, n_hvg, knn, epochs, zdim, znoise_dim, device,
            max_seconds=remaining,
        )
        for j, emb in enumerate(pair_embs):
            idx = start + j
            if embeddings[idx] is None:
                embeddings[idx] = emb

    return embeddings


# ============================================================================
# Embedding evaluation
# ============================================================================


def evaluate_embeddings_dlpfc(slices, slices_raw, sample_ids,
                              inr_models_for_group=None,
                              norm_params_for_group=None,
                              deform_models_for_group=None,
                              rigid_coords_for_group=None,
                              pca_key="X_pca",
                              label_key="original_domain", device="cpu",
                              run_stagate=True, run_staligner=True,
                              run_graphst=True,
                              run_spiral=True, run_seurat=True,
                              staligner_rad_cutoff=150,
                              clustering_methods=("mclust", "leiden_spatial")):
    """Evaluate PCA / INR / STAligner / GraphST / SPIRAL embeddings via ARI+NMI.

    Args:
        slices: list of preprocessed AnnData (normalized, with PCA).
        slices_raw: list of raw AnnData (original counts, for integration methods).
        inr_models_for_group: list of trained ExprINR models (one per pair).
        norm_params_for_group: list of (mean, std) tuples, one per pair.
        deform_models_for_group: list of trained DeformationNet models (one per pair).
        rigid_coords_for_group: list of coords2_rigid in normalized space (one per pair).
        clustering_methods: tuple of clustering methods to evaluate.
    """
    rows = []

    all_labels = set()
    for s in slices:
        all_labels.update(np.asarray(s.obs[label_key]))
    n_clusters = len([l for l in all_labels if str(l).strip() not in ("", "nan")])

    # PCA embeddings
    print("    Running PCA embedding eval...")
    for clust_method in clustering_methods:
        for i, s in enumerate(slices):
            pca_emb = s.obsm[pca_key].copy()
            labels = np.asarray(s.obs[label_key])
            coords = slices_raw[i].obsm["spatial"]
            ari, nmi = compute_ari_nmi(pca_emb, labels, n_clusters,
                                       method=clust_method,
                                       spatial_coords=coords)
            suffix = f"_{clust_method}" if clust_method != "mclust" else ""
            rows.append({"Slice": sample_ids[i], "Embedding": f"PCA{suffix}",
                         "ARI": ari, "NMI": nmi})
            print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")

    # INR embeddings (from trained ExprINR)
    # For pair i: INR was trained on slice i (reference).
    #   - Slice i (ref): normalize coords → INR → embedding
    #   - Slice i+1 (source): normalize coords → rigid → DeformNet → INR → embedding
    if inr_models_for_group and norm_params_for_group:
        print("    Running INR embedding eval...")
        inr_evaluated = set()  # track which slices already evaluated
        for clust_method in clustering_methods:
            for i in range(len(inr_models_for_group)):
                inr_model = inr_models_for_group[i]
                np_entry = norm_params_for_group[i]
                if inr_model is None or np_entry is None:
                    continue
                mean, std = np_entry
                inr_model.eval()

                # --- Slice i (reference): direct coords → INR ---
                s_ref = slices[i]
                raw_coords_ref = s_ref.obsm["spatial"]
                coords_norm_ref = (raw_coords_ref - mean) / std
                with torch.no_grad():
                    coords_t = torch.tensor(coords_norm_ref.astype(np.float32), device=device)
                    alpha = torch.tensor(float(inr_model.n_freqs), device=device)
                    inr_emb_ref = inr_model(coords_t, alpha).cpu().numpy()
                labels_ref = np.asarray(s_ref.obs[label_key])
                coords_ref = s_ref.obsm["spatial"]
                refine_r = None
                if clust_method == "mclust":
                    from sklearn.neighbors import NearestNeighbors
                    _nn = NearestNeighbors(n_neighbors=2).fit(coords_ref)
                    _d, _ = _nn.kneighbors(coords_ref)
                    refine_r = float(np.median(_d[:, 1])) * 3.0
                ari, nmi = compute_ari_nmi(inr_emb_ref, labels_ref, n_clusters,
                                           method=clust_method,
                                           spatial_coords=coords_ref,
                                           refine_radius=refine_r)
                suffix = f"_{clust_method}" if clust_method != "mclust" else ""
                if (i, clust_method) not in inr_evaluated:
                    rows.append({"Slice": sample_ids[i], "Embedding": f"INR{suffix}",
                                 "ARI": ari, "NMI": nmi})
                    print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
                    inr_evaluated.add((i, clust_method))

                # --- Slice i+1 (source): rigid coords → DeformNet → INR ---
                src_idx = i + 1
                if src_idx < len(slices):
                    deform_model = (deform_models_for_group[i]
                                    if deform_models_for_group else None)
                    c2_rigid = (rigid_coords_for_group[i]
                                if rigid_coords_for_group else None)
                    if deform_model is not None and c2_rigid is not None:
                        deform_model.eval()
                        s_src = slices[src_idx]
                        with torch.no_grad():
                            x2_rigid_t = torch.tensor(c2_rigid.astype(np.float32), device=device)
                            alpha_def = torch.tensor(float(deform_model.n_freqs), device=device)
                            x2_def = deform_model(x2_rigid_t, alpha_def)
                            alpha_inr = torch.tensor(float(inr_model.n_freqs), device=device)
                            inr_emb_src = inr_model(x2_def, alpha_inr).cpu().numpy()
                        labels_src = np.asarray(s_src.obs[label_key])
                        coords_src = s_src.obsm["spatial"]
                        refine_r = None
                        if clust_method == "mclust":
                            from sklearn.neighbors import NearestNeighbors
                            _nn = NearestNeighbors(n_neighbors=2).fit(coords_src)
                            _d, _ = _nn.kneighbors(coords_src)
                            refine_r = float(np.median(_d[:, 1])) * 3.0
                        ari, nmi = compute_ari_nmi(inr_emb_src, labels_src, n_clusters,
                                                   method=clust_method,
                                                   spatial_coords=coords_src,
                                                   refine_radius=refine_r)
                        if (src_idx, clust_method) not in inr_evaluated:
                            rows.append({"Slice": sample_ids[src_idx], "Embedding": f"INR{suffix}",
                                         "ARI": ari, "NMI": nmi})
                            print(f"      Slice {sample_ids[src_idx]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
                            inr_evaluated.add((src_idx, clust_method))

    # ---- STAGATE embeddings (per-slice, no cross-slice alignment) ----
    if run_stagate:
        print("    Running STAGATE...")
        try:
            sg_embs = run_stagate_embedding(
                [s.copy() for s in slices_raw], label_key=label_key,
                rad_cutoff=staligner_rad_cutoff, device=device,
            )
            for clust_method in clustering_methods:
                for i, emb in enumerate(sg_embs):
                    labels = np.asarray(slices[i].obs[label_key])
                    coords = slices_raw[i].obsm["spatial"]
                    refine_r = None
                    if clust_method == "mclust":
                        from sklearn.neighbors import NearestNeighbors
                        _nn = NearestNeighbors(n_neighbors=2).fit(coords)
                        _d, _ = _nn.kneighbors(coords)
                        refine_r = float(np.median(_d[:, 1])) * 3.0
                    ari, nmi = compute_ari_nmi(emb, labels, n_clusters,
                                               method=clust_method,
                                               spatial_coords=coords,
                                               refine_radius=refine_r)
                    suffix = f"_{clust_method}" if clust_method != "mclust" else ""
                    rows.append({"Slice": sample_ids[i], "Embedding": f"STAGATE{suffix}",
                                 "ARI": ari, "NMI": nmi})
                    print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
        except Exception as e:
            print(f"    STAGATE failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            _free_gpu()

    # ---- STAligner embeddings ----
    if run_staligner:
        print("    Running STAligner...")
        try:
            st_embs = run_staligner_embedding(
                [s.copy() for s in slices_raw], label_key=label_key,
                rad_cutoff=staligner_rad_cutoff, device=device,
            )
            for clust_method in clustering_methods:
                for i, emb in enumerate(st_embs):
                    labels = np.asarray(slices[i].obs[label_key])
                    coords = slices_raw[i].obsm["spatial"]
                    ari, nmi = compute_ari_nmi(emb, labels, n_clusters,
                                               method=clust_method,
                                               spatial_coords=coords)
                    suffix = f"_{clust_method}" if clust_method != "mclust" else ""
                    rows.append({"Slice": sample_ids[i], "Embedding": f"STAligner{suffix}",
                                 "ARI": ari, "NMI": nmi})
                    print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
        except Exception as e:
            print(f"    STAligner failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            _free_gpu()

    # ---- GraphST embeddings ----
    if run_graphst:
        print("    Running GraphST...")
        try:
            gs_embs = run_graphst_embedding(
                [s.copy() for s in slices_raw], label_key=label_key, device=device,
            )
            for clust_method in clustering_methods:
                for i, emb in enumerate(gs_embs):
                    labels = np.asarray(slices[i].obs[label_key])
                    coords = slices_raw[i].obsm["spatial"]
                    # For mclust: use spatial refinement (GraphST paper protocol)
                    refine_r = None
                    if clust_method == "mclust":
                        from sklearn.neighbors import NearestNeighbors
                        _nn = NearestNeighbors(n_neighbors=2).fit(coords)
                        _d, _ = _nn.kneighbors(coords)
                        refine_r = float(np.median(_d[:, 1])) * 3.0
                    ari, nmi = compute_ari_nmi(emb, labels, n_clusters,
                                               method=clust_method,
                                               spatial_coords=coords,
                                               refine_radius=refine_r)
                    suffix = f"_{clust_method}" if clust_method != "mclust" else ""
                    rows.append({"Slice": sample_ids[i], "Embedding": f"GraphST{suffix}",
                                 "ARI": ari, "NMI": nmi})
                    print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
        except Exception as e:
            print(f"    GraphST failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            _free_gpu()

    # ---- SPIRAL embeddings ----
    if run_spiral:
        print("    Running SPIRAL (max 300s, early stop)...")
        try:
            sp_embs = run_spiral_embedding(
                [s.copy() for s in slices_raw], label_key=label_key, device=device,
                max_seconds=300,
            )
            for clust_method in clustering_methods:
                for i, emb in enumerate(sp_embs):
                    labels = np.asarray(slices[i].obs[label_key])
                    coords = slices_raw[i].obsm["spatial"]
                    ari, nmi = compute_ari_nmi(emb, labels, n_clusters,
                                               method=clust_method,
                                               spatial_coords=coords)
                    suffix = f"_{clust_method}" if clust_method != "mclust" else ""
                    rows.append({"Slice": sample_ids[i], "Embedding": f"SPIRAL{suffix}",
                                 "ARI": ari, "NMI": nmi})
                    print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
        except Exception as e:
            print(f"    SPIRAL failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            _free_gpu()

    # ---- Seurat embeddings ----
    if run_seurat:
        print("    Running Seurat (CCA integration) ...")
        try:
            sr_embs = run_seurat_embedding(
                [s.copy() for s in slices_raw], label_key=label_key,
            )
            for clust_method in clustering_methods:
                for i, emb in enumerate(sr_embs):
                    labels = np.asarray(slices[i].obs[label_key])
                    coords = slices_raw[i].obsm["spatial"]
                    ari, nmi = compute_ari_nmi(emb, labels, n_clusters,
                                               method=clust_method,
                                               spatial_coords=coords)
                    suffix = f"_{clust_method}" if clust_method != "mclust" else ""
                    rows.append({"Slice": sample_ids[i], "Embedding": f"Seurat{suffix}",
                                 "ARI": ari, "NMI": nmi})
                    print(f"      Slice {sample_ids[i]} [{clust_method}]: ARI={ari:.4f}  NMI={nmi:.4f}")
        except Exception as e:
            print(f"    Seurat failed: {e}")
            import traceback; traceback.print_exc()

    return pd.DataFrame(rows)


# ============================================================================
# Formatted output (matching resultalign.xlsx style)
# ============================================================================


def save_embedding_results(df_emb_all, spot_counts, slice_ids,
                           output_path="embedding_results"):
    """Save embedding results in formatted table.

    Args:
        df_emb_all: DataFrame with columns [Slice, Embedding, ARI, NMI, Sample].
        spot_counts: dict {dataset_name: [n_obs_per_slice]}.
        slice_ids: dict {dataset_name: [slice_id, ...]}.
        output_path: base path without extension.
    """
    datasets = list(dict.fromkeys(df_emb_all["Sample"]))  # preserve order

    # Find max number of slices across all datasets
    max_slices = max(len(slice_ids.get(ds, [])) for ds in datasets)

    # Build output rows
    rows_out = []
    for ds in datasets:
        df_ds = df_emb_all[df_emb_all["Sample"] == ds]
        methods = list(dict.fromkeys(df_ds["Embedding"]))  # preserve order
        slices = slice_ids.get(ds, sorted(df_ds["Slice"].unique()))
        spots = spot_counts.get(ds, [])
        spots_str = " + ".join(str(s) for s in spots) + " spots" if spots else ""

        for mi, method in enumerate(methods):
            row = {}
            # Dataset column: name on first row, spots on second, blank on rest
            if mi == 0:
                row["Dataset"] = ds
                row["_new_group"] = True
            elif mi == 1:
                row["Dataset"] = spots_str
            else:
                row["Dataset"] = ""

            row["Method"] = method

            # Per-slice ARI/NMI
            df_m = df_ds[df_ds["Embedding"] == method]
            aris, nmis = [], []
            for si, sl in enumerate(slices):
                df_sl = df_m[df_m["Slice"] == sl]
                if len(df_sl) > 0:
                    a_val = float(df_sl["ARI"].values[0])
                    n_val = float(df_sl["NMI"].values[0])
                    row[f"ARI_s{si + 1}"] = a_val
                    row[f"NMI_s{si + 1}"] = n_val
                    aris.append(a_val)
                    nmis.append(n_val)

            row["ARI(mean)"] = np.mean(aris) if aris else np.nan
            row["NMI(mean)"] = np.mean(nmis) if nmis else np.nan

            rows_out.append(row)

    # Column order: Dataset, Method, ARI(mean), NMI(mean), per-slice pairs
    cols = ["Dataset", "Method", "ARI(mean)", "NMI(mean)"]
    for si in range(max_slices):
        cols.extend([f"ARI_s{si + 1}", f"NMI_s{si + 1}"])
    df_out = pd.DataFrame(rows_out)
    df_out = df_out.drop(columns=["_new_group"], errors="ignore")
    existing_cols = [c for c in cols if c in df_out.columns]
    df_out = df_out[existing_cols]

    # ---- Console output ----
    print("\n" + "=" * 120)
    print("Embedding Evaluation Results (ARI / NMI)")
    print("=" * 120)

    # Column widths
    w_ds = max(16, max(len(str(r.get("Dataset", ""))) for r in rows_out) + 2)
    w_mt = max(28, max(len(str(r.get("Method", ""))) for r in rows_out) + 2)
    w_val = 9

    # Header
    hdr = f"{'Dataset':<{w_ds}} {'Method':<{w_mt}} {'ARI(m)':>{w_val}} {'NMI(m)':>{w_val}}"
    for si in range(max_slices):
        hdr += f" {'ARI_s' + str(si + 1):>{w_val}} {'NMI_s' + str(si + 1):>{w_val}}"
    print(hdr)
    print("-" * len(hdr))

    cur_ds = None
    for r in rows_out:
        ds_str = str(r.get("Dataset", ""))
        mt_str = str(r.get("Method", ""))
        new_group = r.get("_new_group", False)

        # Separator between dataset groups
        if new_group and cur_ds is not None:
            print("-" * len(hdr))
        if new_group:
            cur_ds = ds_str

        line = f"{ds_str:<{w_ds}} {mt_str:<{w_mt}}"
        line += f" {r.get('ARI(mean)', 0):>{w_val}.4f} {r.get('NMI(mean)', 0):>{w_val}.4f}"
        for si in range(max_slices):
            ak = f"ARI_s{si + 1}"
            nk = f"NMI_s{si + 1}"
            a_v = r.get(ak, np.nan)
            n_v = r.get(nk, np.nan)
            if np.isnan(a_v):
                line += f" {'':>{w_val}} {'':>{w_val}}"
            else:
                line += f" {a_v:>{w_val}.4f} {n_v:>{w_val}.4f}"
        print(line)

    print("=" * len(hdr))

    # ---- Save CSV ----
    df_out.to_csv(f"{output_path}.csv", index=False)
    print(f"\nResults saved to {output_path}.csv")

    # ---- Save XLSX ----
    try:
        df_out.to_excel(f"{output_path}.xlsx", index=False)
        print(f"Results saved to {output_path}.xlsx")
    except ImportError:
        print("  (openpyxl not installed, skipping .xlsx)")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run embedding evaluation")
    parser.add_argument("--no_paste", action="store_true", default=True)
    parser.add_argument("--no_paste2", action="store_true", default=True)
    parser.add_argument("--no_gpsa", action="store_true", default=True)
    parser.add_argument("--no_spateo", action="store_true", default=True)
    parser.add_argument("--no_stalign", action="store_true", default=True)
    parser.add_argument("--no_stagate", action="store_true", help="Skip STAGATE embedding")
    parser.add_argument("--no_staligner", action="store_true", help="Skip STAligner embedding")
    parser.add_argument("--no_graphst", action="store_true", help="Skip GraphST embedding")
    parser.add_argument("--no_spiral", action="store_true", help="Skip SPIRAL embedding")
    parser.add_argument("--no_seurat", action="store_true", help="Skip Seurat CCA embedding")
    parser.add_argument("--no_inr", action="store_true", help="Skip INST-Align INR training and embedding eval")
    parser.add_argument("--sample_groups", type=int, nargs="+", default=[0, 1, 2],
                        help="Which DLPFC sample groups to run (0-indexed). Default: all 3")
    parser.add_argument("--dlpfc_pairs", nargs="+", default=None,
                        help="Slice index pairs per group, e.g. '0,2' '0,1' '0,2'. "
                             "Default: use all 4 slices per group.")
    parser.add_argument("--run_embryo", action="store_true",
                        help="Also run embedding eval on MouseEmbryo dataset")
    parser.add_argument("--no_dlpfc", action="store_true",
                        help="Skip DLPFC embedding eval in Part 2 (useful for embryo-only runs)")
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip Part 1 (alignment), only run Part 2 (embedding eval)")
    parser.add_argument("--clustering", nargs="+",
                        default=["mclust", "leiden_spatial"],
                        choices=["mclust", "kmeans", "leiden", "leiden_spatial"],
                        help="Clustering methods for embedding evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = PipelineConfig(data_dir="./Data")

    # Parse dlpfc_pairs: selects which slices per group for BOTH Part 1 and Part 2
    # e.g. ["0,1", "0,1", "0,1"] → [[0,1], [0,1], [0,1]]
    # IMPORTANT: use consecutive indices (e.g. 0,1 or 1,2) for meaningful alignment
    dlpfc_pairs = None
    if args.dlpfc_pairs:
        dlpfc_pairs = [[int(x) for x in p.split(",")] for p in args.dlpfc_pairs]

    # ================================================================
    # Part 1: Train INST-Align models used for INR embeddings
    # ================================================================
    sample_indices = args.sample_groups
    inr_models = {}
    norm_params = {}
    deform_models = {}
    rigid_coords_norm = {}

    if args.no_inr:
        print("\n[Skipping INST-Align INR training: --no_inr]")
    elif not args.skip_alignment:
        print("\n" + "#" * 70)
        print("# Part 1: Train INST-Align models for INR embeddings")
        print("#" * 70)

        layer_groups = []
        sample_id_groups = []
        dataset_folders = []
        for gi, idx in enumerate(sample_indices):
            group_full = DLPFC_SAMPLE_GROUPS[idx]
            if dlpfc_pairs and gi < len(dlpfc_pairs):
                sel = dlpfc_pairs[gi]
                group = [group_full[i] for i in sel]
            else:
                group = list(group_full)
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

        df_align, inr_models, norm_params, deform_models, rigid_coords_norm = run_spatial_alignment_all(
            layer_groups, config, device=device,
            label_key="original_domain", label_map=DLPFC_LABEL_MAP,
            run_paste=not args.no_paste,
            run_paste2=not args.no_paste2,
            run_gpsa=not args.no_gpsa,
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

        df_align.to_csv("embedding_alignment_cache.csv", index=False)
        print(f"\nAlignment cache saved to embedding_alignment_cache.csv")
    else:
        print("\n[Skipping Part 1: Alignment]")


    # ================================================================
    # Part 2: Embedding Evaluation (ARI, NMI)
    # ================================================================
    print("\n" + "#" * 70)
    print("# Part 2: Embedding Evaluation (ARI, NMI)")
    print("#" * 70)

    import spateo as st

    all_emb_dfs = []
    spot_counts = {}   # {dataset_name: [n_obs_per_slice]}
    slice_ids = {}     # {dataset_name: [slice_id, ...]}

    # dlpfc_pairs already parsed above (before Part 1)

    if args.no_dlpfc:
        sample_indices = []

    for gi, idx in enumerate(sample_indices):
        group_full = DLPFC_SAMPLE_GROUPS[idx]
        folder = f"DLPFC_sample{idx + 1}"
        ds_name = f"DLPFC_sample{idx + 1}"

        # Select specific slices if dlpfc_pairs provided
        if dlpfc_pairs and gi < len(dlpfc_pairs):
            sel = dlpfc_pairs[gi]
            group = [group_full[i] for i in sel]
        else:
            group = list(group_full)
        print(f"\n  --- {ds_name} ({group}) ---")

        # Load raw slices (for integration methods)
        slices_raw = []
        for sample_id in group:
            path = f"./Data/{folder}/original_data/{sample_id}.h5ad"
            slices_raw.append(sc.read_h5ad(path))

        spot_counts[ds_name] = [a.n_obs for a in slices_raw]
        slice_ids[ds_name] = list(group)

        # Preprocessed slices (for PCA / INR)
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
        st.align.group_pca(slices, pca_key="X_pca")

        # Collect trained INR models, deform models, and normalization params
        inr_list = []
        norm_list = []
        deform_list = []
        rigid_coords_list = []
        for pi in range(len(group) - 1):
            inr_list.append(inr_models.get((gi, pi), None))
            norm_list.append(norm_params.get((gi, pi), None))
            deform_list.append(deform_models.get((gi, pi), None))
            rigid_coords_list.append(rigid_coords_norm.get((gi, pi), None))

        df_emb = evaluate_embeddings_dlpfc(
            slices, slices_raw, group,
            inr_models_for_group=inr_list,
            norm_params_for_group=norm_list,
            deform_models_for_group=deform_list,
            rigid_coords_for_group=rigid_coords_list,
            device=device,
            run_stagate=not args.no_stagate,
            run_staligner=not args.no_staligner,
            run_graphst=not args.no_graphst,
            run_spiral=not args.no_spiral,
            run_seurat=not args.no_seurat,
            clustering_methods=tuple(args.clustering),
        )
        df_emb["Sample"] = ds_name
        all_emb_dfs.append(df_emb)

        # Free memory between sample groups to prevent OOM
        del slices_raw, slices
        _free_gpu()

    # ================================================================
    # Part 2b: MouseEmbryo Embedding Evaluation
    # ================================================================
    if args.run_embryo:
        print("\n" + "#" * 70)
        print("# Part 2b: MouseEmbryo Embedding Evaluation (ARI, NMI)")
        print("#" * 70)

        embryo_label_key = "cellbin_SpatialDomain"

        # Load raw slices
        embryo_raw = []
        for si in [1, 2]:
            path = f"./Data/MouseEmbryo/sample_data/slices{si}.h5ad"
            a = sc.read_h5ad(path)
            print(f"  Loaded MouseEmbryo slice{si}: {a.shape}")
            embryo_raw.append(a)

        # Preprocess (normalize, log1p, HVG, PCA)
        embryo_slices = []
        for adata in embryo_raw:
            a = adata.copy()
            if "counts" not in a.layers:
                a.layers["counts"] = a.X.copy()
            sc.pp.normalize_total(a)
            sc.pp.log1p(a)
            if "highly_variable" not in a.var.columns:
                sc.pp.highly_variable_genes(a, n_top_genes=2000)
            embryo_slices.append(a)
        st.align.group_pca(embryo_slices, pca_key="X_pca")

        sample_ids_embryo = ["embryo_s1", "embryo_s2"]
        spot_counts["MouseEmbryo"] = [a.n_obs for a in embryo_raw]
        slice_ids["MouseEmbryo"] = sample_ids_embryo

        # Compute rad_cutoff for STAligner from median NN distance
        from sklearn.neighbors import NearestNeighbors as _NN_rc
        _coords0 = embryo_raw[0].obsm["spatial"]
        _nn_rc = _NN_rc(n_neighbors=2).fit(_coords0)
        _d_rc, _ = _nn_rc.kneighbors(_coords0)
        embryo_rad_cutoff = float(np.median(_d_rc[:, 1])) * 3.0
        print(f"  STAligner rad_cutoff for embryo: {embryo_rad_cutoff:.1f}")

        # Train INR on MouseEmbryo pair (phase 1 + phase 2)
        embryo_inr_list = [None]
        embryo_norm_list = [None]
        embryo_deform_list = [None]
        embryo_rigid_coords_list = [None]
        if not args.no_inr:
            print("  Training INR on MouseEmbryo...")
            # Use PCA-based ICP for embryo (may have rotation)
            saved_icp_mode = config.icp.mode
            config.icp.mode = "pca"
            try:
                _, _, _t_inr, embryo_expr_inr, embryo_norm_ms, embryo_deform, embryo_rigid_coords = run_ours(
                    embryo_raw[0].copy(), embryo_raw[1].copy(),
                    config, device=device,
                    label_key=embryo_label_key,
                )
                embryo_inr_list = [embryo_expr_inr]
                embryo_norm_list = [embryo_norm_ms]
                embryo_deform_list = [embryo_deform]
                embryo_rigid_coords_list = [embryo_rigid_coords]
                print(f"  INR training done ({_t_inr:.1f}s)")
            except Exception as e:
                print(f"  INR training failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                config.icp.mode = saved_icp_mode  # restore for other datasets

        df_emb_embryo = evaluate_embeddings_dlpfc(
            embryo_slices, embryo_raw, sample_ids_embryo,
            inr_models_for_group=embryo_inr_list,
            norm_params_for_group=embryo_norm_list,
            deform_models_for_group=embryo_deform_list,
            rigid_coords_for_group=embryo_rigid_coords_list,
            label_key=embryo_label_key,
            device=device,
            run_stagate=not args.no_stagate,
            run_staligner=not args.no_staligner,
            run_graphst=not args.no_graphst,
            run_spiral=not args.no_spiral,
            run_seurat=not args.no_seurat,
            staligner_rad_cutoff=embryo_rad_cutoff,
            clustering_methods=tuple(args.clustering),
        )
        df_emb_embryo["Sample"] = "MouseEmbryo"
        all_emb_dfs.append(df_emb_embryo)

        del embryo_raw, embryo_slices
        _free_gpu()

    # ================================================================
    # Print & save all results
    # ================================================================
    if all_emb_dfs:
        df_emb_all = pd.concat(all_emb_dfs, ignore_index=True)
        save_embedding_results(df_emb_all, spot_counts, slice_ids)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run the paper ablation study for INST-Align.

Variants:
  Full           — complete pipeline (Phase1 pretrain + Phase2 joint alignment)
  w/o Phase2     — Phase1 only, skip Phase2 (no non-rigid deformation)
  w/o Phase1     — skip Phase1 pretrain, Phase2 from random INR
  w/o Jacobian   — full pipeline but lam_jacobian=0

Output:
  supplementary/ablation_{dataset}_results.csv
  supplementary/ablation_{dataset}_embedding.csv

Usage::

    python run_ablation_insta.py --dataset DLPFC_sample1
    python run_ablation_insta.py --dataset MouseEmbryo
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA as skPCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

import spateo as st

from insta.config import SLICE_ORDER, PipelineConfig
from insta.trainer import apply_model, train
from insta.metrics import (
    chamfer_distance,
)
from insta.model import (
    DeformationNet,
    UnifiedCostMatcher,
    adaptive_icp,
    build_joint_models,
)
from insta.utils import coords_to_pi, mapping_accuracy_paste, normalize_coordinates

warnings.filterwarnings("ignore")


# Label key per dataset
LABEL_KEYS = {
    "MouseEmbryo": "cellbin_SpatialDomain",
    "DLPFC": "original_domain",
}

DATASET_OVERRIDES = {
    "MouseEmbryo": {
        "epochs": 500,
        "batch_size": 4000,
        "lr": 3e-4,
        "grad_clip": 2.0,
        "lam_jacobian": 0.01,
        "lam_deform_mag": 0.0,
        "tau_min": 0.005,
        "mode": "pca",
        "inr_pretrain_epochs": 150,
        "freeze_inr_phase2": True,
        "lam_recon_phase2": 0.0,
        "warmup_fraction": 0.0,
        "scheduler_patience": 9999,
    },
}


def _get_label_key(dataset: str) -> str:
    for prefix, key in LABEL_KEYS.items():
        if dataset.startswith(prefix):
            return key
    return "original_domain"


# ============================================================================
# Ablation variant definitions
# ============================================================================

VARIANTS = OrderedDict([
    ("Full", {
        "desc": "Phase1 pretrain + Phase2 joint alignment (complete pipeline)",
    }),
    ("w/o Phase2", {
        "desc": "Phase1 pretrain only, skip Phase2 (no non-rigid deformation)",
        "epochs": 0,
        # DeformNet stays identity-init -> alignment = rigid ICP only
        # INR embedding reflects Phase1 pretrain quality without joint training
    }),
    ("w/o Phase1", {
        "desc": "Skip Phase1 pretrain, Phase2 from random INR (inr_pretrain_epochs=0)",
        "inr_pretrain_epochs": 0,
    }),
    ("w/o Jacobian", {
        "desc": "Full pipeline but no Jacobian regularization (lam_jacobian=0)",
        "lam_jacobian": 0.0,
    }),
])


# ============================================================================
# Config helpers
# ============================================================================


def make_config(dataset: str, data_dir: str = "./Data") -> PipelineConfig:
    """Create config with dataset-specific overrides."""
    cfg = PipelineConfig(data_dir=data_dir)
    key = None
    for prefix in DATASET_OVERRIDES:
        if dataset.startswith(prefix) and (key is None or len(prefix) > len(key)):
            key = prefix
    if key is None:
        return cfg

    for field_name, value in DATASET_OVERRIDES[key].items():
        if hasattr(cfg.train, field_name):
            setattr(cfg.train, field_name, value)
        elif hasattr(cfg.joint, field_name):
            setattr(cfg.joint, field_name, value)
        elif hasattr(cfg.matcher, field_name):
            setattr(cfg.matcher, field_name, value)
        elif hasattr(cfg.icp, field_name):
            setattr(cfg.icp, field_name, value)
    return cfg


def apply_variant(base_config: PipelineConfig, variant_name: str) -> Tuple[PipelineConfig, bool]:
    """Return (modified_config, use_inr_flag) for a given variant."""
    cfg = copy.deepcopy(base_config)
    settings = VARIANTS[variant_name]
    use_inr = settings.get("use_inr", True)

    if "inr_pretrain_epochs" in settings:
        cfg.joint.inr_pretrain_epochs = settings["inr_pretrain_epochs"]
    if "epochs" in settings:
        cfg.train.epochs = settings["epochs"]
    if "lam_jacobian" in settings:
        cfg.joint.lam_jacobian = settings["lam_jacobian"]

    return cfg, use_inr


# ============================================================================
# Data loading
# ============================================================================


def load_and_preprocess(dataset: str, data_dir: str = "./Data"):
    """Load and preprocess a single pair of slices."""
    order = SLICE_ORDER.get(dataset, ["slices1", "slices2"])

    # Find data path
    path = os.path.join(data_dir, dataset, "sample_data")
    if not os.path.exists(path):
        path = os.path.join(data_dir, dataset, "original_data")

    s1 = sc.read_h5ad(os.path.join(path, f"{order[0]}.h5ad"))
    s2 = sc.read_h5ad(os.path.join(path, f"{order[1]}.h5ad"))
    print(f"Loaded {dataset}: {order[0]} ({s1.n_obs}) + {order[1]} ({s2.n_obs})")

    # Preprocess
    for ad in [s1, s2]:
        if "counts" not in ad.layers:
            ad.layers["counts"] = ad.X.copy()
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
        if "highly_variable" not in ad.var.columns:
            sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    st.align.group_pca([s1, s2], pca_key="X_pca")

    return s1, s2


# ============================================================================
# Single-pair alignment
# ============================================================================


def align_pair(
    slice1, slice2,
    config: PipelineConfig,
    device: str,
    use_inr: bool = True,
) -> Dict[str, Any]:
    """Align one pair. Returns dict with coords, models, and timing."""
    start = time.time()

    coords1 = slice1.obsm[config.spatial_key]
    coords2 = slice2.obsm[config.spatial_key]
    [c1_norm, c2_norm], mean, std = normalize_coordinates([coords1, coords2])

    # ICP rigid alignment
    R, t, angle, rmse = adaptive_icp(
        c1_norm, c2_norm, config.icp, verbose=False,
        emb_A=slice1.obsm[config.pca_key].astype(np.float32),
        emb_B=slice2.obsm[config.pca_key].astype(np.float32),
    )
    c2_rigid = ((R @ c2_norm.T).T + t).astype(np.float32)

    # ICP Only: skip deformation entirely
    if config.train.epochs == 0 and not use_inr:
        c2_rigid_denorm = c2_rigid * std + mean
        elapsed = time.time() - start
        return {
            "c2_rigid": c2_rigid_denorm,
            "c2_final": c2_rigid_denorm,
            "elapsed": elapsed,
            "expr_inr_s1": None,
            "expr_inr_s2": None,
            "deform": None,
            "c1_norm": c1_norm,
            "c2_rigid_norm": c2_rigid,
            "mean": mean, "std": std,
        }

    # GPU tensors
    x1 = torch.tensor(c1_norm.astype(np.float32), device=device)
    x2 = torch.tensor(c2_rigid, device=device)
    emb1 = torch.tensor(slice1.obsm[config.pca_key].astype(np.float32), device=device)
    emb2 = torch.tensor(slice2.obsm[config.pca_key].astype(np.float32), device=device)

    # DeformationNet + Matcher
    model = DeformationNet(config.model).to(device)
    matcher = UnifiedCostMatcher(config.matcher)

    # INR components
    inr_kwargs: dict = {}
    expr_inr_s1, expr_inr_s2, decoder = None, None, None
    if use_inr:
        hvg1_mask = slice1.var["highly_variable"].values
        hvg2_mask = slice2.var["highly_variable"].values
        X1_hvg = slice1.X[:, hvg1_mask]
        X2_hvg = slice2.X[:, hvg2_mask]
        if sp.issparse(X1_hvg):
            X1_hvg = X1_hvg.toarray()
        if sp.issparse(X2_hvg):
            X2_hvg = X2_hvg.toarray()
        hvg1 = torch.tensor(X1_hvg.astype(np.float32), device=device)
        hvg2 = torch.tensor(X2_hvg.astype(np.float32), device=device)

        jcfg = copy.deepcopy(config.joint)
        jcfg.n_output = hvg1.shape[1]
        models = build_joint_models(jcfg, device=device)
        expr_inr_s1 = models["expr_inr_s1"]
        expr_inr_s2 = models["expr_inr_s2"]
        decoder = models["decoder"]

        inr_kwargs = dict(
            expr_inr_s1=expr_inr_s1,
            expr_inr_s2=expr_inr_s2,
            decoder=decoder,
            hvg1=hvg1, hvg2=hvg2,
        )

    # Train
    result = train(
        model, matcher, x1, emb1, x2, emb2,
        config.train, config.joint,
        **inr_kwargs,
    )

    # Apply deformation & denormalize
    model.eval()
    x2_def = apply_model(model, x2)
    c2_final = x2_def.cpu().numpy() * std + mean
    c2_rigid_denorm = c2_rigid * std + mean

    elapsed = time.time() - start

    return {
        "c2_rigid": c2_rigid_denorm,
        "c2_final": c2_final,
        "elapsed": elapsed,
        "expr_inr_s1": expr_inr_s1,
        "expr_inr_s2": expr_inr_s2,
        "deform": model,
        "c1_norm": c1_norm,
        "c2_rigid_norm": c2_rigid,
        "mean": mean, "std": std,
    }


# ============================================================================
# Metrics
# ============================================================================


def compute_alignment_metrics(
    coords1, coords2_aligned, labels1, labels2,
    label_map=None,
) -> Dict[str, float]:
    """Compute ablation alignment metrics from Table 3."""
    v1 = np.array([str(l) not in ("NA", "nan", "None", "") for l in labels1])
    v2 = np.array([str(l) not in ("NA", "nan", "None", "") for l in labels2])
    c1f, c2f = coords1[v1], coords2_aligned[v2]
    l1f, l2f = labels1[v1], labels2[v2]

    pi = coords_to_pi(c1f, c2f)
    acc_ot = mapping_accuracy_paste(pd.Series(l1f), pd.Series(l2f), pi, label_map)
    cham = chamfer_distance(coords1, coords2_aligned)

    return {
        "Accuracy": acc_ot,
        "Chamfer": cham,
    }


# ============================================================================
# Embedding evaluation (ARI/NMI)
# ============================================================================

_MCLUST_LOADED = False


def _mclust_R(embedding, n_clusters, model_names="EEE", random_seed=666):
    global _MCLUST_LOADED
    try:
        import rpy2.robjects as robjects
        if not _MCLUST_LOADED:
            robjects.r.library("mclust")
            _MCLUST_LOADED = True
        nr, nc = embedding.shape
        r_vec = robjects.FloatVector(embedding.flatten(order='F'))
        r_mat = robjects.r['matrix'](r_vec, nrow=nr, ncol=nc)
        robjects.globalenv['emb.mat'] = r_mat
        robjects.globalenv['nc.val'] = robjects.IntVector([n_clusters])
        robjects.globalenv['mn.val'] = robjects.StrVector([model_names])
        robjects.r('set.seed(%d)' % random_seed)
        res = robjects.r('Mclust(emb.mat, G=nc.val, modelNames=mn.val)')
        if res == robjects.NULL:
            raise RuntimeError("mclust returned NULL")
        return np.array(res.rx2('classification')).astype(int) - 1
    except Exception as e:
        print(f"    [mclust fallback to sklearn GMM: {e}]")
        gmm = GaussianMixture(
            n_components=n_clusters, covariance_type="tied",
            n_init=10, random_state=random_seed, max_iter=300,
        )
        return gmm.fit_predict(embedding)


def _spatial_refine(pred, spatial_coords, n_clusters, radius):
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


def get_refine_radius(spatial_coords):
    nn = NearestNeighbors(n_neighbors=2).fit(spatial_coords)
    d, _ = nn.kneighbors(spatial_coords)
    return float(np.median(d[:, 1])) * 3.0


def compute_ari_nmi(embedding, labels, n_clusters,
                    spatial_coords=None, refine_radius=None):
    valid = np.array([str(l).strip() not in ("", "nan", "NA") for l in labels])
    emb = embedding[valid]
    lab = labels[valid]
    unique_labels = np.unique(lab)
    if len(unique_labels) < 2:
        return 0.0, 0.0
    n_c = min(n_clusters, len(unique_labels))

    n_pca = min(20, emb.shape[1], emb.shape[0] - 1)
    if emb.shape[1] > n_pca:
        emb_for_clust = skPCA(n_components=n_pca, random_state=42).fit_transform(emb)
    else:
        emb_for_clust = emb

    pred = _mclust_R(emb_for_clust, n_c)

    if refine_radius is not None and spatial_coords is not None:
        coords_valid = spatial_coords[valid]
        pred = _spatial_refine(pred, coords_valid, n_c, refine_radius)

    ari = adjusted_rand_score(lab, pred)
    nmi = normalized_mutual_info_score(lab, pred)
    return ari, nmi


def eval_inr_embedding(inr_model, coords_norm, labels, spatial_coords,
                       n_clusters, device, deform_model=None):
    """Evaluate INR embedding quality via ARI/NMI.

    For the source slice, optionally apply deformation first, then pass
    through the *reference* INR (expr_inr_s1), matching the embedding protocol.
    """
    inr_model.eval()
    with torch.no_grad():
        ct = torch.tensor(coords_norm.astype(np.float32), device=device)
        if deform_model is not None:
            deform_model.eval()
            alpha_def = torch.tensor(float(deform_model.n_freqs), device=device)
            ct = deform_model(ct, alpha_def)
        alpha = torch.tensor(float(inr_model.n_freqs), device=device)
        emb = inr_model(ct, alpha).cpu().numpy()
    refine_r = get_refine_radius(spatial_coords)
    return compute_ari_nmi(emb, labels, n_clusters,
                           spatial_coords=spatial_coords,
                           refine_radius=refine_r)


# ============================================================================
# Main
# ============================================================================


def main(dataset: str, data_dir: str = "./Data"):
    outdir = "supplementary"
    os.makedirs(outdir, exist_ok=True)
    output = f"{outdir}/ablation_{dataset}_results.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    s1_orig, s2_orig = load_and_preprocess(dataset, data_dir)
    label_key = _get_label_key(dataset)
    print(f"Label key: {label_key}")

    labels1 = np.asarray(s1_orig.obs[label_key])
    labels2 = np.asarray(s2_orig.obs[label_key])
    coords1 = s1_orig.obsm["spatial"]

    # Count valid labels
    all_labels = set()
    for labs in [labels1, labels2]:
        all_labels.update(l for l in labs if str(l).strip() not in ("", "nan", "NA"))
    n_clusters = len(all_labels)
    print(f"Unique labels: {n_clusters}")

    base_config = make_config(dataset, data_dir)

    align_rows = []
    emb_rows = []

    for variant_name, variant_info in VARIANTS.items():
        print(f"\n{'#' * 70}")
        print(f"# {variant_name}: {variant_info['desc']}")
        print(f"{'#' * 70}")

        cfg, use_inr = apply_variant(base_config, variant_name)

        try:
            res = align_pair(
                s1_orig.copy(), s2_orig.copy(),
                cfg, device, use_inr=use_inr,
            )

            # Alignment metrics
            metrics = compute_alignment_metrics(
                coords1, res["c2_final"], labels1, labels2,
            )

            align_rows.append({
                "Variant": variant_name,
                "Time": round(res["elapsed"], 1),
                **metrics,
            })

            print(f"  Align: OT={metrics['Accuracy']:.4f}  "
                  f"Chamfer={metrics['Chamfer']:.4f}  "
                  f"({res['elapsed']:.1f}s)")

            # Embedding ARI/NMI (only for INR variants)
            # Use expr_inr_s1 (reference INR)
            # for BOTH slices.  For source slice, apply DeformNet first.
            if use_inr and res["expr_inr_s1"] is not None:
                ari_s0, nmi_s0 = eval_inr_embedding(
                    res["expr_inr_s1"], res["c1_norm"], labels1, coords1,
                    n_clusters, device,
                )
                # Source slice: expr_inr_s1(DeformNet(c2_rigid_norm))
                ari_s1, nmi_s1 = eval_inr_embedding(
                    res["expr_inr_s1"], res["c2_rigid_norm"], labels2,
                    s2_orig.obsm["spatial"], n_clusters, device,
                    deform_model=res["deform"],
                )

                emb_rows.append({
                    "Variant": variant_name,
                    "ARI_s0": round(ari_s0, 4),
                    "NMI_s0": round(nmi_s0, 4),
                    "ARI_s1": round(ari_s1, 4),
                    "NMI_s1": round(nmi_s1, 4),
                })
                print(f"  Emb:  s0 ARI={ari_s0:.4f} NMI={nmi_s0:.4f}  "
                      f"s1 ARI={ari_s1:.4f} NMI={nmi_s1:.4f}")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Incremental save
        pd.DataFrame(align_rows).to_csv(output, index=False)

    # ======================================================================
    # Summary
    # ======================================================================

    df_align = pd.DataFrame(align_rows)
    df_align.to_csv(output, index=False)

    print(f"\n{'=' * 70}")
    print(f"Ablation Results — {dataset}")
    print(f"{'=' * 70}")

    metric_cols = ["Accuracy", "Chamfer"]
    header = f"{'Variant':22s}"
    for m in metric_cols:
        header += f" | {m:>12s}"
    header += f" | {'Time':>6s}"
    print(header)
    print("-" * len(header))

    for _, row in df_align.iterrows():
        line = f"{row['Variant']:22s}"
        for m in metric_cols:
            line += f" | {row[m]:12.4f}"
        line += f" | {row['Time']:6.1f}"
        print(line)

    print(f"\nSaved to {output}")

    # Embedding
    if emb_rows:
        df_emb = pd.DataFrame(emb_rows)
        emb_path = output.replace(".csv", "_embedding.csv")
        df_emb.to_csv(emb_path, index=False)

        print(f"\n{'=' * 70}")
        print(f"Embedding Quality — {dataset}")
        print(f"{'=' * 70}")
        print(f"{'Variant':22s} | {'ARI_s0':>8s} | {'NMI_s0':>8s} | "
              f"{'ARI_s1':>8s} | {'NMI_s1':>8s}")
        print("-" * 62)

        for _, row in df_emb.iterrows():
            print(f"{row['Variant']:22s} | {row['ARI_s0']:8.4f} | {row['NMI_s0']:8.4f} | "
                  f"{row['ARI_s1']:8.4f} | {row['NMI_s1']:8.4f}")

        print(f"\nSaved to {emb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run INST-Align ablation study")
    parser.add_argument("--dataset", default="DLPFC_sample1")
    parser.add_argument("--data_dir", default="./Data")
    args = parser.parse_args()
    main(args.dataset, args.data_dir)

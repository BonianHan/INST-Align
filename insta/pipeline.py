"""End-to-end alignment pipeline.

Command-line usage::

    python -m insta --dataset STARMap
    python -m insta --dataset STARMap --epochs 300 --use_expr_inr
    python -m insta --datasets STARMap DLPFC_sample1 BaristaSeq

Library usage::

    from insta.pipeline import run, align_pair
    aligned, metrics = run(PipelineConfig(dataset="STARMap"))

Output:
    Per-slice h5ad files saved to ``insta/inr/result/{dataset}/``
    Each file retains original spatial (X, Y) and adds inr_X, inr_Y in .obs
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from insta.config import SLICE_ORDER, DLPFC_SAMPLE_GROUPS, PipelineConfig
from insta.metrics import compute_istbench_metrics
from insta.model import (
    DeformationNet, UnifiedCostMatcher, adaptive_icp,
    build_joint_models, build_knn_graph,
)
from insta.trainer import TrainResult, apply_model, train
from insta.utils import (
    coords_to_pi,
    detect_grid_spacing,
    denormalize_coordinates,
    mapping_accuracy_paste,
    normalize_coordinates,
)


# ============================================================================
# Data loading
# ============================================================================


def load_slices(
    dataset: str,
    data_dir: str,
    slice_names: List[str],
) -> List[ad.AnnData]:
    """Load h5ad slices from ``{data_dir}/{dataset}/sample_data/``."""
    sample_dir = os.path.join(data_dir, dataset, "sample_data")
    slices = []
    for name in slice_names:
        path = os.path.join(sample_dir, f"{name}.h5ad")
        adata = sc.read_h5ad(path)
        print(f"  {name}: {adata.shape[0]} cells, {adata.shape[1]} genes")
        slices.append(adata)
    return slices


# ============================================================================
# Preprocessing
# ============================================================================


def preprocess_slices(
    slices: List[ad.AnnData],
    n_top_genes: int = 2000,
    pca_key: str = "X_pca",
) -> List[ad.AnnData]:
    """Filter, normalize, HVG selection, and joint PCA."""
    import spateo as st

    processed = []
    for adata in slices:
        a = adata.copy()
        sc.pp.filter_cells(a, min_genes=1)
        sc.pp.filter_genes(a, min_cells=1)
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a)
        sc.pp.log1p(a)
        try:
            sc.pp.highly_variable_genes(a, n_top_genes=min(n_top_genes, a.shape[1]))
        except Exception:
            a.var["highly_variable"] = True
        processed.append(a)

    st.align.group_pca(processed, pca_key=pca_key)
    return processed


# ============================================================================
# Single-pair alignment
# ============================================================================


def align_pair(
    ref_slice: ad.AnnData,
    src_slice: ad.AnnData,
    config: PipelineConfig,
    device: str,
) -> Tuple[np.ndarray, Optional[TrainResult]]:
    """Align *src_slice* to *ref_slice* using two-phase training.

    Normalization is computed pairwise (ref + src only), not globally.

    Args:
        ref_slice: Reference (target) AnnData.
        src_slice: Source AnnData.
        config: Pipeline configuration.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(coords_aligned, train_result_or_None)``.  ``train_result``
        is ``None`` when ICP alone was sufficient.
    """
    coords1_raw = ref_slice.obsm[config.spatial_key]
    coords2_raw = src_slice.obsm[config.spatial_key]
    [coords_ref, coords_src], mean, std = normalize_coordinates([coords1_raw, coords2_raw])

    spacing_x, spacing_y, is_grid, origin = detect_grid_spacing(coords_ref)
    if is_grid:
        print(f"  Grid detected: spacing=({spacing_x:.4f}, {spacing_y:.4f})")
    else:
        print(f"  Non-grid data: continuous coordinates")

    # Adaptive ICP rigid alignment (with expression-guided rotation selection)
    emb_ref_np = ref_slice.obsm[config.pca_key].astype(np.float32)
    emb_src_np = src_slice.obsm[config.pca_key].astype(np.float32)
    R, t, angle, rmse = adaptive_icp(
        coords_ref, coords_src, config.icp, verbose=True,
        emb_A=emb_ref_np, emb_B=emb_src_np,
    )
    coords_rigid = ((R @ coords_src.T).T + t).astype(np.float32)
    print(f"  ICP: angle={angle:.1f}, RMSE={rmse:.4f}")

    if rmse < config.icp.icp_threshold:
        print(f"  ICP error < {config.icp.icp_threshold}, skipping neural deformation")
        return denormalize_coordinates(coords_rigid, mean, std), None

    print(f"  Running neural deformation (two-phase INR)...")

    x_ref = torch.tensor(coords_ref.astype(np.float32), device=device)
    x2 = torch.tensor(coords_rigid, device=device)
    emb_ref = torch.tensor(emb_ref_np, device=device)
    emb_src = torch.tensor(emb_src_np, device=device)

    # HVG expression for reconstruction target (log-normalized)
    import scipy.sparse as sp
    hvg_mask_ref = ref_slice.var["highly_variable"].values
    hvg_mask_src = src_slice.var["highly_variable"].values
    X_ref_hvg = ref_slice.X[:, hvg_mask_ref]
    X_src_hvg = src_slice.X[:, hvg_mask_src]
    if sp.issparse(X_ref_hvg):
        X_ref_hvg = X_ref_hvg.toarray()
    if sp.issparse(X_src_hvg):
        X_src_hvg = X_src_hvg.toarray()
    hvg_ref = torch.tensor(X_ref_hvg.astype(np.float32), device=device)
    hvg_src = torch.tensor(X_src_hvg.astype(np.float32), device=device)

    # DeformNet (spatial only)
    model = DeformationNet(config.model).to(device)
    matcher = UnifiedCostMatcher(config.matcher)

    # Joint components (dual ExprINR + shared decoder)
    jcfg = config.joint
    jcfg.n_output = hvg_ref.shape[1]      # HVG count for decoder output
    models = build_joint_models(jcfg, device=device)

    result = train(
        model, matcher, x_ref, emb_ref, x2, emb_src,
        config.train, jcfg,
        expr_inr_s1=models["expr_inr_s1"],
        expr_inr_s2=models["expr_inr_s2"],
        decoder=models["decoder"],
        hvg1=hvg_ref,
        hvg2=hvg_src,
    )

    model.eval()
    x2_def = apply_model(model, x2)
    coords_final = denormalize_coordinates(x2_def.cpu().numpy(), mean, std)

    return coords_final, result


# ============================================================================
# Full pipeline (library API)
# ============================================================================


def run(config: Optional[PipelineConfig] = None) -> Tuple[List[ad.AnnData], pd.DataFrame]:
    """Run the full alignment pipeline.

    Returns:
        ``(aligned_slices, metrics_df)``.
    """
    if config is None:
        config = PipelineConfig()

    # Resolve device
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    print(f"Device: {device}")

    # Slice names
    if config.dataset in SLICE_ORDER:
        slice_names = SLICE_ORDER[config.dataset]
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}. Add to SLICE_ORDER in config.py")

    # Load and preprocess
    print(f"\nLoading {config.dataset}...")
    slices_raw = load_slices(config.dataset, config.data_dir, slice_names)
    print("Preprocessing...")
    slices = preprocess_slices(slices_raw, config.n_top_genes, config.pca_key)

    # First slice = reference
    aligned_slices = [slices[0].copy()]
    aligned_slices[0].obsm["spatial_aligned"] = slices[0].obsm[config.spatial_key].copy()

    # Pairwise alignment (each pair normalizes independently)
    print(f"\n{'=' * 60}")
    print(f"Aligning {len(slices)} slices to {slice_names[0]} (reference)")
    print(f"{'=' * 60}")

    for i in range(1, len(slices)):
        print(f"\n{'=' * 50}")
        print(f"Aligning {slice_names[i]} \u2192 {slice_names[0]}")
        print(f"{'=' * 50}")

        coords_aligned, _ = align_pair(
            slices[0], slices[i], config, device,
        )

        aligned = slices[i].copy()
        aligned.obsm["spatial_aligned"] = coords_aligned
        aligned_slices.append(aligned)

    # Metrics
    print(f"\n{'=' * 60}")
    print("Computing metrics")
    print(f"{'=' * 60}")
    metrics_df = compute_istbench_metrics(aligned_slices, slice_names, config.label_key)
    print(metrics_df.to_string(index=False))
    print(f"\nMean Accuracy: {metrics_df['Accuracy'].mean():.4f}")

    # Save
    save_results(aligned_slices, metrics_df, slice_names, config)

    return aligned_slices, metrics_df


# ============================================================================
# Saving — standard format
# ============================================================================


def save_results(
    aligned_slices: List[ad.AnnData],
    metrics_df: pd.DataFrame,
    slice_names: List[str],
    config: PipelineConfig,
) -> None:
    """Save aligned h5ad and metrics CSV."""
    output_dir = os.path.join(config.output_dir, config.dataset, "INR_Align")
    os.makedirs(output_dir, exist_ok=True)

    for s in aligned_slices:
        if "spatial_aligned" in s.obsm:
            s.obsm["spatial"] = s.obsm["spatial_aligned"]

    adata_concat = ad.concat(aligned_slices, label="slice_name", keys=slice_names)
    adata_concat.write_h5ad(os.path.join(output_dir, "Spatial_correct_data.h5ad"))
    metrics_df.to_csv(os.path.join(output_dir, "Accuracy.csv"), index=False)
    print(f"\n\u2705 Results saved to: {output_dir}")


# ============================================================================
# Saving — INR result format (per-slice h5ad with inr_X, inr_Y)
# ============================================================================


def save_inr_results(
    aligned_slices: List[ad.AnnData],
    slice_names: List[str],
    config: PipelineConfig,
) -> str:
    """Save per-slice h5ad files with INR-aligned coordinates.

    Output directory: ``insta/inr/result/{dataset}/``

    Returns:
        Output directory path.
    """
    result_dir = os.path.join("insta", "inr", "result", config.dataset)
    os.makedirs(result_dir, exist_ok=True)

    for i, (adata, name) in enumerate(zip(aligned_slices, slice_names)):
        out = adata.copy()

        if "spatial_aligned" in out.obsm:
            aligned_coords = out.obsm["spatial_aligned"]
        else:
            aligned_coords = out.obsm["spatial"]

        out.obs["inr_X"] = aligned_coords[:, 0]
        out.obs["inr_Y"] = aligned_coords[:, 1]

        out_path = os.path.join(result_dir, f"{name}.h5ad")
        out.write_h5ad(out_path)
        print(f"  Saved {out_path} ({out.shape[0]} cells)")

    print(f"\u2705 INR results saved to: {result_dir}")
    return result_dir


# ============================================================================
# Train a single dataset
# ============================================================================


def train_dataset(dataset: str, config: PipelineConfig, device: str) -> None:
    """Train INR alignment on a single dataset and save results."""
    print(f"\n{'#' * 60}")
    print(f"# Training: {dataset}")
    print(f"{'#' * 60}")

    config.dataset = dataset

    # Determine data format
    is_dlpfc_original = dataset.startswith("DLPFC_sample")

    if is_dlpfc_original:
        # DLPFC uses original_data format
        sample_idx = int(dataset.replace("DLPFC_sample", "")) - 1
        if sample_idx >= len(DLPFC_SAMPLE_GROUPS):
            print(f"  ERROR: {dataset} not found in DLPFC_SAMPLE_GROUPS")
            return
        group = DLPFC_SAMPLE_GROUPS[sample_idx]
        folder = f"DLPFC_sample{sample_idx + 1}"

        # Load slices
        import spateo as st
        slices = []
        slice_names = group  # e.g. ["151507", "151508", "151509", "151510"]
        for sample_id in group:
            path = os.path.join(config.data_dir, folder, "original_data", f"{sample_id}.h5ad")
            adata = sc.read_h5ad(path)
            slices.append(adata)
            print(f"  Loaded {sample_id}: {adata.shape}")

        # Preprocess with the same settings used by the experiment runners.
        for ad_ in slices:
            if "counts" not in ad_.layers:
                ad_.layers["counts"] = ad_.X.copy()
            sc.pp.normalize_total(ad_)
            sc.pp.log1p(ad_)
            if "highly_variable" not in ad_.var.columns:
                sc.pp.highly_variable_genes(ad_, n_top_genes=config.n_top_genes)
        st.align.group_pca(slices, pca_key=config.pca_key)

    else:
        # sample_data format
        if dataset not in SLICE_ORDER:
            print(f"  ERROR: {dataset} not in SLICE_ORDER")
            return
        slice_names = SLICE_ORDER[dataset]
        print(f"  Slice order: {slice_names}")
        slices = load_slices(dataset, config.data_dir, slice_names)
        slices = preprocess_slices(slices, config.n_top_genes, config.pca_key)

    # Reference slice (no alignment needed)
    aligned_slices = [slices[0].copy()]
    aligned_slices[0].obsm["spatial_aligned"] = slices[0].obsm[config.spatial_key].copy()

    # Pairwise alignment (each pair normalizes independently)
    for i in range(1, len(slices)):
        print(f"\n{'=' * 50}")
        print(f"  Aligning {slice_names[i]} -> {slice_names[0]}")
        print(f"{'=' * 50}")

        coords_aligned, result = align_pair(
            slices[0], slices[i], config, device,
        )

        aligned = slices[i].copy()
        aligned.obsm["spatial_aligned"] = coords_aligned
        aligned_slices.append(aligned)

        if result is not None:
            print(f"  Training time: {result.training_time:.1f}s, best match loss: {result.best_match_loss:.6f}")

    # Save results
    save_inr_results(aligned_slices, slice_names, config)


# ============================================================================
# CLI
# ============================================================================


ALL_DATASETS = list(SLICE_ORDER.keys())


def main() -> None:
    """CLI entry point: align one or more datasets."""
    from insta.config import add_pipeline_args, config_from_args, print_config

    parser = argparse.ArgumentParser(
        description="INST-Align: Spatial transcriptomics alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Run-specific args
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Multiple datasets to align sequentially",
    )
    # All pipeline config args
    add_pipeline_args(parser)

    args = parser.parse_args()
    config = config_from_args(args)
    print_config(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Determine what to run
    datasets = args.datasets
    if datasets is None:
        # Single dataset mode using config.dataset
        run(config)
    else:
        # Multi-dataset mode
        for ds in datasets:
            train_dataset(ds, config, device)


if __name__ == "__main__":
    main()

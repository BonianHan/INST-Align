"""End-to-end alignment pipeline.

Command-line usage::

    python -m inr_align --dataset STARMap
    python -m inr_align --dataset STARMap --epochs 300 --use_expr_inr
    python -m inr_align --datasets STARMap DLPFC_sample1 BaristaSeq

Library usage::

    from inr_align.run import run, align_pair
    aligned, metrics = run(PipelineConfig(dataset="STARMap"))

Output:
    Per-slice h5ad files saved to ``inr_align/inr/result/{dataset}/``
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
import scipy.sparse
import torch
import torch.nn.functional as F

from inr_align.config import SLICE_ORDER, DLPFC_SAMPLE_GROUPS, PipelineConfig
from inr_align.loss import compute_P_matrix
from inr_align.metrics import compute_istbench_metrics, mapping_accuracy_paste, coords_to_pi
from inr_align.model import (
    DeformationNet, ExprField, GeneDecoder, UnifiedCostMatcher, adaptive_icp,
    normalize_expression,
)
from inr_align.train import TrainResult, apply_model, train
from inr_align.utils import detect_grid_spacing, denormalize_coordinates, normalize_coordinates


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


def _prepare_expr_field(
    ref_slice: ad.AnnData,
    src_slice: ad.AnnData,
    config: PipelineConfig,
    device: str,
) -> Tuple[Optional[ExprField], Optional[torch.Tensor]]:
    """Create a fresh ExprField and prepare source expression for joint training.

    Returns:
        ``(expr_field, expr2_gt)`` or ``(None, None)`` if disabled.
    """
    if not config.use_expr_field:
        return None, None

    ef_config = config.expr_field

    # Shared HVG set
    hvg_mask = ref_slice.var["highly_variable"].values

    expr_arrays = []
    for s in [ref_slice, src_slice]:
        raw = s[:, hvg_mask].X
        if scipy.sparse.issparse(raw):
            raw = raw.toarray()
        expr_arrays.append(raw.astype(np.float32))

    # Select top-variance genes
    n_hvg = min(ef_config.n_hvg, expr_arrays[0].shape[1])
    pooled_var = sum(np.var(e, axis=0) for e in expr_arrays)
    top_idx = np.argsort(pooled_var)[-n_hvg:]

    # Global normalization stats
    all_expr = np.vstack([e[:, top_idx] for e in expr_arrays])
    all_expr_t = torch.tensor(all_expr, device=device)
    _, norm_stats = normalize_expression(all_expr_t, ef_config.norm_method)

    # Source expression (normalized)
    expr2_sub = torch.tensor(expr_arrays[1][:, top_idx], device=device)
    expr2_gt, _ = normalize_expression(expr2_sub, ef_config.norm_method, stats=norm_stats)

    # Fresh ExprField (no pretrain — will be trained jointly)
    expr_field = ExprField(ef_config, n_genes=n_hvg, n_slices=1).to(device)
    print(f"  ExprField: {n_hvg} genes, joint training (lam_expr={ef_config.lam_expr})")

    return expr_field, expr2_gt


def align_pair(
    ref_slice: ad.AnnData,
    src_slice: ad.AnnData,
    mean: np.ndarray,
    std: np.ndarray,
    grid_info: dict,
    config: PipelineConfig,
    device: str,
    splane_emb_src: Optional[np.ndarray] = None,
    pretrained_expr_field: Optional[ExprField] = None,
) -> Tuple[np.ndarray, Optional[TrainResult]]:
    """Align *src_slice* to *ref_slice*.

    Args:
        ref_slice: Reference (target) AnnData.
        src_slice: Source AnnData.
        mean: Global coordinate mean.
        std: Global coordinate std.
        grid_info: Dict with ``spacing_x``, ``spacing_y``, ``is_grid``,
            ``origin``.
        config: Pipeline configuration.
        device: ``"cuda"`` or ``"cpu"``.
        splane_emb_src: ``(N_src, E)`` pre-computed Splane embeddings for
            source slice (optional). Enables KL + gene recon losses.
        pretrained_expr_field: Pre-trained ExprField (optional, legacy).

    Returns:
        ``(coords_aligned, train_result_or_None)``.  ``train_result``
        is ``None`` when ICP alone was sufficient.
    """
    coords_ref = (ref_slice.obsm[config.spatial_key] - mean) / std
    coords_src = (src_slice.obsm[config.spatial_key] - mean) / std

    # Adaptive ICP rigid alignment (with expression-guided rotation selection)
    emb_ref_np = ref_slice.obsm[config.pca_key].astype(np.float32)
    emb_src_np = src_slice.obsm[config.pca_key].astype(np.float32)
    R, t, angle, rmse = adaptive_icp(
        coords_ref, coords_src, config.icp, verbose=True,
        emb_A=emb_ref_np, emb_B=emb_src_np,
    )
    coords_rigid = ((R @ coords_src.T).T + t).astype(np.float32)
    print(f"  ICP: angle={angle:.1f}\u00b0, RMSE={rmse:.4f}")

    if rmse < config.icp.icp_threshold:
        print(f"  \u2713 ICP error < {config.icp.icp_threshold}, skipping neural deformation")
        return denormalize_coordinates(coords_rigid, mean, std), None

    print(f"  \u2192 Running neural deformation...")

    x_ref = torch.tensor(coords_ref.astype(np.float32), device=device)
    x2 = torch.tensor(coords_rigid, device=device)
    emb_ref = torch.tensor(ref_slice.obsm[config.pca_key].astype(np.float32), device=device)
    emb2 = torch.tensor(src_slice.obsm[config.pca_key].astype(np.float32), device=device)

    is_grid = grid_info["is_grid"]

    # Determine embedding dimension
    has_splane = splane_emb_src is not None
    emb_dim = config.model.emb_dim if has_splane else 0

    model = DeformationNet(
        config.model,
        emb_dim=emb_dim,
    ).to(device)

    matcher = UnifiedCostMatcher(config.matcher)

    # Splane KL
    splane_emb2_t = None
    if has_splane:
        splane_emb2_t = torch.tensor(splane_emb_src, device=device)
        print(f"  Splane embeddings: {splane_emb2_t.shape}")

    # Gene reconstruction
    gene_decoder = None
    gene_expr2_gt = None
    slice_ids2 = None
    if has_splane and config.train.lam_recon > 0:
        from inr_align.benchmark import _prepare_gene_expr
        gene_expr2_gt = _prepare_gene_expr(src_slice, n_hvg=config.expr_field.n_hvg, device=device)
        if gene_expr2_gt is not None:
            n_genes = gene_expr2_gt.shape[1]
            gd = config.gene_decoder
            gene_decoder = GeneDecoder(
                emb_dim=emb_dim, batch_dim=gd.batch_dim, hidden=gd.hidden,
                layers=gd.layers, n_genes=n_genes, n_slices=2,
            ).to(device)
            slice_ids2 = torch.ones(src_slice.shape[0], dtype=torch.long, device=device)
            print(f"  GeneDecoder: emb_dim={emb_dim}, n_genes={n_genes}")

    # ExprField (legacy joint training, created fresh per pair)
    expr_field, expr2_gt = _prepare_expr_field(ref_slice, src_slice, config, device)

    result = train(
        model, matcher, x_ref, emb_ref, x2, emb2, config.train,
        expr_field=expr_field,
        expr2_gt=expr2_gt,
        lam_expr=config.expr_field.lam_expr if config.use_expr_field else 0.0,
        splane_emb2=splane_emb2_t,
        gene_decoder=gene_decoder,
        gene_expr2_gt=gene_expr2_gt,
        slice_ids2=slice_ids2,
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

    # Global normalization
    all_coords = [s.obsm[config.spatial_key] for s in slices]
    _, mean, std = normalize_coordinates(all_coords)

    # Grid detection on reference
    coords_ref_norm = (slices[0].obsm[config.spatial_key] - mean) / std
    sx, sy, is_grid, origin = detect_grid_spacing(coords_ref_norm)
    grid_info = {"spacing_x": sx, "spacing_y": sy, "is_grid": is_grid, "origin": origin}
    print(f"Grid mode: {is_grid}")

    # First slice = reference
    aligned_slices = [slices[0].copy()]
    aligned_slices[0].obsm["spatial_aligned"] = slices[0].obsm[config.spatial_key].copy()

    # Pairwise alignment
    print(f"\n{'=' * 60}")
    print(f"Aligning {len(slices)} slices to {slice_names[0]} (reference)")
    print(f"{'=' * 60}")

    for i in range(1, len(slices)):
        print(f"\n{'=' * 50}")
        print(f"Aligning {slice_names[i]} \u2192 {slice_names[0]}")
        print(f"{'=' * 50}")

        coords_aligned, _ = align_pair(
            slices[0], slices[i], mean, std, grid_info, config, device,
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

    Output directory: ``inr_align/inr/result/{dataset}/``

    Returns:
        Output directory path.
    """
    result_dir = os.path.join("inr_align", "inr", "result", config.dataset)
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

        # Preprocess (same as benchmark run_ours)
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

    # Global normalization
    all_coords = [s.obsm[config.spatial_key] for s in slices]
    _, mean, std = normalize_coordinates(all_coords)

    # Grid detection on reference
    coords_ref_norm = (slices[0].obsm[config.spatial_key] - mean) / std
    sx, sy, is_grid, origin = detect_grid_spacing(coords_ref_norm)
    grid_info = {"spacing_x": sx, "spacing_y": sy, "is_grid": is_grid, "origin": origin}
    if is_grid:
        print(f"  Grid detected: spacing=({sx:.4f}, {sy:.4f})")
    else:
        print(f"  Non-grid data: continuous coordinates")

    # Load Splane embeddings (if available)
    splane_embs = {}
    if is_dlpfc_original:
        splane_path = os.path.join(config.data_dir, folder, "splane_embeddings.npz")
        if os.path.exists(splane_path):
            splane_data = np.load(splane_path)
            for sid in slice_names:
                key = f"emb_{sid}"
                if key in splane_data:
                    splane_embs[sid] = splane_data[key].astype(np.float32)
            print(f"  Splane embeddings loaded: {list(splane_embs.keys())}")
        else:
            print(f"  [WARN] Splane embeddings not found at {splane_path}")

    # Reference slice (no alignment needed)
    aligned_slices = [slices[0].copy()]
    aligned_slices[0].obsm["spatial_aligned"] = slices[0].obsm[config.spatial_key].copy()

    # Pairwise alignment
    for i in range(1, len(slices)):
        print(f"\n{'=' * 50}")
        print(f"  Aligning {slice_names[i]} -> {slice_names[0]}")
        print(f"{'=' * 50}")

        splane_src = splane_embs.get(slice_names[i], None)

        coords_aligned, result = align_pair(
            slices[0], slices[i], mean, std, grid_info, config, device,
            splane_emb_src=splane_src,
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
    from inr_align.config import add_pipeline_args, config_from_args, print_config

    parser = argparse.ArgumentParser(
        description="INR-Align: Spatial transcriptomics alignment",
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

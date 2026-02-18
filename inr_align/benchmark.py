"""Multi-method benchmark: PASTE / Spateo / SPACEL / STalign / Ours / No-align.

Compares all methods on the same dataset using seven metrics:

1. **OT Accuracy** — PASTE-style transport-plan weighted label match.
2. **NN Accuracy** — Bidirectional nearest-neighbour label match (iSTBench-style).
3. **Ratio** — Many-to-one collapse measure.
4. **CLC** — Contextual Label Consistency.
5. **LISI** — Local Inverse Simpson's Index (domain consistency).
6. **Silhouette** — Domain separation in aligned space.
7. **Chamfer** — Symmetric Chamfer distance (geometric alignment).

Run directly::

    python -m inr_align.benchmark

Or import::

    from inr_align.benchmark import benchmark_all, print_summary
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import seaborn as sns
import spateo as st
import torch

import scipy.sparse

from inr_align.config import DLPFC_SAMPLE_GROUPS, PipelineConfig
from inr_align.loss import compute_P_matrix
from inr_align.metrics import (
    calculate_clc,
    chamfer_distance,
    compute_lisi,
    compute_silhouette,
    coords_to_pi,
    mapping_accuracy_nn_bidi,
    mapping_accuracy_paste,
    sparse_P_to_dense_pi,
)
from inr_align.model import (
    DeformationNet, GeneDecoder, UnifiedCostMatcher, adaptive_icp,
    normalize_expression, ExprField,
)
from inr_align.train import apply_model, train
from inr_align.utils import detect_grid_spacing, normalize_coordinates
from inr_align.loss1 import align_pair_joint
from inr_align.model1 import JointConfig


# ============================================================================
# DLPFC label map (PASTE convention)
# ============================================================================

DLPFC_LABEL_MAP = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6, "WM": 7}


# ============================================================================
# Individual method runners — return aligned coordinates + time
# ============================================================================


def run_paste_baseline(
    slice1,
    slice2,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """PASTE pairwise alignment.

    Returns:
        ``(pi, elapsed_time)``.  PASTE produces a transport plan,
        not explicit aligned coordinates.
    """
    import paste as pst

    pi0 = pst.match_spots_using_spatial_heuristic(
        slice1.obsm["spatial"], slice2.obsm["spatial"], use_ot=True
    )
    start = time.time()
    pi = pst.pairwise_align(
        slice1, slice2,
        alpha=alpha, G_init=pi0, norm=True, verbose=False,
        use_gpu=True, backend=ot.backend.TorchBackend(),
    )
    elapsed = time.time() - start
    return pi, elapsed


def run_spateo_baseline(
    slice1,
    slice2,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Spateo morpho_align (rigid + nonrigid).

    Returns:
        ``(target_coords, coords_rigid, coords_nonrigid, elapsed_time)``.
    """
    s1 = slice1.copy()
    s2 = slice2.copy()

    start = time.time()
    aligned_slices, _ = st.align.morpho_align(
        models=[s1, s2],
        verbose=False,
        spatial_key="spatial",
        key_added="align_spatial",
        device=device,
        dissimilarity="cos",
    )
    elapsed = time.time() - start

    target_coords = aligned_slices[0].obsm["align_spatial"]
    coords_rigid = aligned_slices[1].obsm["align_spatial_rigid"]
    coords_nonrigid = aligned_slices[1].obsm["align_spatial_nonrigid"]

    return target_coords, coords_rigid, coords_nonrigid, elapsed


def run_spacel_baseline(
    slice1,
    slice2,
    label_key: str = "original_domain",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """SPACEL Scube.align (graph-based alignment).

    Runs in a separate ``spacel`` conda environment via subprocess,
    because SPACEL requires ``torch<=1.13`` which is incompatible with
    the main environment.

    Returns:
        ``(coords1_aligned, coords2_aligned, elapsed_time)``.
    """
    import os
    import subprocess
    import tempfile

    # Save slices to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        s1_path = os.path.join(tmpdir, "s1.h5ad")
        s2_path = os.path.join(tmpdir, "s2.h5ad")
        out_path = os.path.join(tmpdir, "result.npz")

        slice1.write_h5ad(s1_path)
        slice2.write_h5ad(s2_path)

        # Find spacel_runner.py relative to this file
        runner = os.path.join(os.path.dirname(os.path.dirname(__file__)), "spacel_runner.py")

        cmd = [
            "conda", "run", "-n", "spacel", "--no-capture-output",
            "python", runner,
            "--slice1", s1_path,
            "--slice2", s2_path,
            "--label_key", label_key,
            "--output", out_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"SPACEL subprocess failed:\n{result.stderr}")

        data = np.load(out_path)
        coords1 = data["coords1"]
        coords2 = data["coords2"]
        elapsed = float(data["elapsed"])

    return coords1, coords2, elapsed


def run_stalign_baseline(
    slice1,
    slice2,
    device: str = "cuda",
    dx: int = 30,
) -> Tuple[np.ndarray, float]:
    """STalign LDDMM diffeomorphic registration.

    Returns:
        ``(coords2_aligned, elapsed_time)``.
    """
    from STalign import STalign as STalign_module

    coords1 = slice1.obsm["spatial"]
    coords2 = slice2.obsm["spatial"]

    # Target = slice1, Source = slice2
    xJ = np.array(coords1[:, 0])
    yJ = np.array(coords1[:, 1])
    XJ, YJ, J, _ = STalign_module.rasterize(xJ, yJ, dx=dx)

    xI = np.array(coords2[:, 0])
    yI = np.array(coords2[:, 1])
    XI, YI, I, _ = STalign_module.rasterize(xI, yI, dx=dx)

    start = time.time()
    params = {"niter": 10000, "device": device, "epV": 50}
    out = STalign_module.LDDMM([YI, XI], I, [YJ, XJ], J, **params)
    A, v, xv = out["A"], out["v"], out["xv"]

    dtype = A.dtype
    points_np = np.stack([yI, xI], axis=1)
    points_tensor = torch.tensor(points_np, dtype=dtype).to(device)
    tpointsI = STalign_module.transform_points_source_to_target(xv, v, A, points_tensor)
    if tpointsI.is_cuda:
        tpointsI = tpointsI.cpu()
    elapsed = time.time() - start

    coords2_aligned = tpointsI[:, [1, 0]].numpy()
    return coords2_aligned, elapsed


def _load_splane_embeddings(
    slice_obj,
    sample_id: str,
    data_dir: str = "./Data",
    dataset_folder: str = "DLPFC_sample1",
) -> Optional[np.ndarray]:
    """Try to load pre-computed Splane embeddings for a slice.

    Looks for ``{data_dir}/{dataset_folder}/splane_embeddings.npz``
    with key ``emb_{sample_id}``.

    Returns:
        ``(N, D)`` embedding array or ``None`` if not found.
    """
    import os
    splane_path = os.path.join(data_dir, dataset_folder, "splane_embeddings.npz")
    if not os.path.exists(splane_path):
        return None
    data = np.load(splane_path)
    key = f"emb_{sample_id}"
    if key in data:
        return data[key].astype(np.float32)
    return None


def _prepare_gene_expr(
    slice_obj,
    n_hvg: int = 2000,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """Extract normalized HVG expression from a slice for gene reconstruction.

    Returns:
        ``(N, n_hvg)`` tensor on device, or ``None`` on failure.
    """
    try:
        if "highly_variable" in slice_obj.var.columns:
            hvg_mask = slice_obj.var["highly_variable"].values
            raw = slice_obj[:, hvg_mask].X
        else:
            raw = slice_obj.X

        if scipy.sparse.issparse(raw):
            raw = raw.toarray()
        raw = raw.astype(np.float32)

        # Take top n_hvg by variance
        if raw.shape[1] > n_hvg:
            var = np.var(raw, axis=0)
            top_idx = np.argsort(var)[-n_hvg:]
            raw = raw[:, top_idx]

        # Per-gene z-score
        mean = raw.mean(axis=0)
        std_val = raw.std(axis=0) + 1e-8
        raw_norm = (raw - mean) / std_val

        return torch.tensor(raw_norm, device=device)
    except Exception as e:
        print(f"  [WARN] Failed to prepare gene expression: {e}")
        return None


def run_ours(
    slice1,
    slice2,
    config: PipelineConfig,
    device: str = "cuda",
    label_key: str = "original_domain",
    splane_emb2_np: Optional[np.ndarray] = None,
    dataset_folder: str = "DLPFC_sample1",
    sample_id2: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Our method: adaptive_icp + INR deformation.

    Returns:
        ``(coords2_rigid_denorm, coords2_final, elapsed_time)``.
    """
    # Preprocessing
    for ad_ in [slice1, slice2]:
        if "counts" not in ad_.layers:
            ad_.layers["counts"] = ad_.X.copy()
        sc.pp.normalize_total(ad_)
        sc.pp.log1p(ad_)
        if "highly_variable" not in ad_.var.columns:
            sc.pp.highly_variable_genes(ad_, n_top_genes=config.n_top_genes)
    st.align.group_pca([slice1, slice2], pca_key=config.pca_key)

    start = time.time()

    coords1 = slice1.obsm[config.spatial_key]
    coords2 = slice2.obsm[config.spatial_key]
    [coords1_norm, coords2_norm], mean, std = normalize_coordinates([coords1, coords2])

    spacing_x, spacing_y, is_grid, origin = detect_grid_spacing(coords1_norm)
    if is_grid:
        print(f"  Grid detected: spacing=({spacing_x:.4f}, {spacing_y:.4f})")
    else:
        print(f"  Non-grid data: continuous coordinates")

    # Adaptive ICP (with expression-guided rotation selection)
    R, t, angle, rmse = adaptive_icp(
        coords1_norm, coords2_norm, config.icp, verbose=True,
        emb_A=slice1.obsm[config.pca_key].astype(np.float32),
        emb_B=slice2.obsm[config.pca_key].astype(np.float32),
    )
    coords2_rigid = ((R @ coords2_norm.T).T + t).astype(np.float32)

    # GPU tensors
    x1 = torch.tensor(coords1_norm.astype(np.float32), device=device)
    x2 = torch.tensor(coords2_rigid, device=device)
    emb1 = torch.tensor(slice1.obsm[config.pca_key].astype(np.float32), device=device)
    emb2 = torch.tensor(slice2.obsm[config.pca_key].astype(np.float32), device=device)

    # --- Splane embeddings ---
    splane_emb2 = None
    if splane_emb2_np is not None:
        splane_emb2 = torch.tensor(splane_emb2_np, device=device)
        print(f"  Splane embeddings loaded: {splane_emb2.shape}")
    elif sample_id2 is not None:
        sp = _load_splane_embeddings(slice2, sample_id2, config.data_dir, dataset_folder)
        if sp is not None:
            splane_emb2 = torch.tensor(sp, device=device)
            print(f"  Splane embeddings loaded: {splane_emb2.shape}")
        else:
            print(f"  [WARN] Splane embeddings not found for {sample_id2}")

    # Determine if embedding head should be active
    has_splane = splane_emb2 is not None
    emb_dim = config.model.emb_dim if has_splane else 0

    # Model (with embedding head if Splane available)
    model = DeformationNet(
        config.model,
        emb_dim=emb_dim,
    ).to(device)

    matcher = UnifiedCostMatcher(config.matcher)

    # --- Gene reconstruction ---
    gene_decoder = None
    gene_expr2_gt = None
    slice_ids2 = None
    if has_splane and config.train.lam_recon > 0:
        gene_expr2_gt = _prepare_gene_expr(slice2, n_hvg=config.expr_field.n_hvg, device=device)
        if gene_expr2_gt is not None:
            n_genes = gene_expr2_gt.shape[1]
            gd = config.gene_decoder
            gene_decoder = GeneDecoder(
                emb_dim=emb_dim,
                batch_dim=gd.batch_dim,
                hidden=gd.hidden,
                layers=gd.layers,
                n_genes=n_genes,
                n_slices=2,
            ).to(device)
            # Source slice = slice_id 1
            slice_ids2 = torch.ones(slice2.shape[0], dtype=torch.long, device=device)
            print(f"  GeneDecoder: emb_dim={emb_dim}, n_genes={n_genes}")

    # ExprField (legacy joint training)
    expr_field = None
    expr2_gt = None
    if config.use_expr_field:
        from inr_align.run import _prepare_expr_field
        expr_field, expr2_gt = _prepare_expr_field(slice1, slice2, config, device)

    result = train(
        model, matcher, x1, emb1, x2, emb2, config.train,
        expr_field=expr_field,
        expr2_gt=expr2_gt,
        lam_expr=config.expr_field.lam_expr if config.use_expr_field else 0.0,
        splane_emb2=splane_emb2,
        gene_decoder=gene_decoder,
        gene_expr2_gt=gene_expr2_gt,
        slice_ids2=slice_ids2,
    )

    # Apply deformation
    model.eval()

    # Apply deformation and denormalize
    x2_def = apply_model(model, x2)
    coords2_final = x2_def.cpu().numpy() * std + mean

    coords2_rigid_denorm = coords2_rigid * std + mean

    elapsed = time.time() - start

    return coords2_rigid_denorm, coords2_final, elapsed


def run_ours_joint(
    slice1,
    slice2,
    config: PipelineConfig,
    jcfg: Optional[JointConfig] = None,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Our method with decoupled joint architecture (Encoder+Decoder+Discriminator).

    Drop-in replacement for ``run_ours`` using the new joint training loop.

    Returns:
        ``(coords2_rigid_denorm, coords2_final, elapsed_time)``.
    """
    if jcfg is None:
        jcfg = JointConfig()
    return align_pair_joint(slice1, slice2, config, jcfg, device)


# ============================================================================
# Full benchmark
# ============================================================================


def benchmark_all(
    layer_groups: List[List],
    config: Optional[PipelineConfig] = None,
    device: str = "cuda",
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
    run_paste: bool = True,
    run_spateo: bool = True,
    run_spacel: bool = True,
    run_stalign: bool = True,
    sample_id_groups: Optional[List[List[str]]] = None,
    dataset_folders: Optional[List[str]] = None,
    run_joint: bool = False,
    joint_config: Optional[JointConfig] = None,
) -> pd.DataFrame:
    """Run all methods on all sample groups.

    Args:
        layer_groups: ``layer_groups[j][i]`` is an AnnData for sample
            group *j*, slice *i*.
        config: Pipeline config (for our method's hyper-parameters).
        device: CUDA or CPU.
        label_key: Label column name.
        label_map: Optional label -> int mapping for accuracy.
            ``None`` uses generic label equality.
        run_paste: Whether to include PASTE baseline.
        run_spateo: Whether to include Spateo baseline.
        run_spacel: Whether to include SPACEL baseline.
        run_stalign: Whether to include STalign baseline.
        sample_id_groups: ``sample_id_groups[j][i]`` is the sample ID string
            for group *j*, slice *i*.  Used to load Splane embeddings.
        dataset_folders: ``dataset_folders[j]`` is the dataset folder name
            (e.g. ``"DLPFC_sample1"``) for group *j*.

    Returns:
        DataFrame with columns ``[Sample, Pair, Method, Time, Accuracy,
        Accuracy_NN, Ratio, CLC, LISI, Silhouette, Chamfer]``.
    """
    if config is None:
        config = PipelineConfig()

    rows = []

    for j in range(len(layer_groups)):
        for i in range(len(layer_groups[j]) - 1):
            print(f"\n{'=' * 60}")
            print(f"Sample {j}, Pair {i}")
            print(f"{'=' * 60}")

            s1 = layer_groups[j][i].copy()
            s2 = layer_groups[j][i + 1].copy()

            _coords1 = s1.obsm["spatial"]
            _coords2 = s2.obsm["spatial"]
            _l1 = np.asarray(s1.obs[label_key])
            _l2 = np.asarray(s2.obs[label_key])

            def _make_row(method, t, c2_aligned, c1_ref=None):
                """Compute all 7 metrics and build result row."""
                c1 = c1_ref if c1_ref is not None else _coords1
                pi = coords_to_pi(c1, c2_aligned)
                acc_ot = mapping_accuracy_paste(s1.obs[label_key], s2.obs[label_key], pi, label_map)
                acc_nn, ratio = mapping_accuracy_nn_bidi(_l1, _l2, c1, c2_aligned)
                clc_val = calculate_clc(_l1, _l2, c1, c2_aligned)
                lisi = compute_lisi(c1, c2_aligned, _l1, _l2)
                sil = compute_silhouette(c1, c2_aligned, _l1, _l2)
                cham = chamfer_distance(c1, c2_aligned)
                row = {
                    "Sample": j, "Pair": i, "Method": method, "Time": t,
                    "Accuracy": acc_ot, "Accuracy_NN": acc_nn, "Ratio": ratio,
                    "CLC": clc_val, "LISI": lisi, "Silhouette": sil, "Chamfer": cham,
                }
                print(f"  {method:16s} OT={acc_ot:.4f}  NN={acc_nn:.4f}  "
                      f"Rat={ratio:.4f}  CLC={clc_val:.4f}  "
                      f"LISI={lisi:.3f}  Sil={sil:.4f}  Cham={cham:.4f}")
                return row

            # --- No-align ---
            rows.append(_make_row("No-align", 0.0, _coords2))

            # --- PASTE ---
            if run_paste:
                try:
                    pi_paste, t_p = run_paste_baseline(s1.copy(), s2.copy(), alpha=0.1)
                    # PASTE has no explicit aligned coords; compute NN metrics on original
                    row_p = _make_row("PASTE", t_p, _coords2)
                    # Override OT accuracy with PASTE's pi-based value
                    row_p["Accuracy"] = mapping_accuracy_paste(
                        s1.obs[label_key], s2.obs[label_key], pi_paste, label_map
                    )
                    rows.append(row_p)
                except Exception as e:
                    print(f"  PASTE failed: {e}")

            # --- SPACEL ---
            if run_spacel:
                try:
                    c1_sc, c2_sc, t_sc = run_spacel_baseline(s1.copy(), s2.copy(), label_key)
                    rows.append(_make_row("SPACEL", t_sc, c2_sc, c1_ref=c1_sc))
                except Exception as e:
                    print(f"  SPACEL failed: {e}")

            # --- STalign ---
            if run_stalign:
                try:
                    c2_st, t_st = run_stalign_baseline(s1.copy(), s2.copy(), device)
                    rows.append(_make_row("STalign", t_st, c2_st))
                except Exception as e:
                    print(f"  STalign failed: {e}")

            # --- Spateo ---
            if run_spateo:
                try:
                    target_sp, c_rigid_sp, c_nonrigid_sp, t_sp = run_spateo_baseline(
                        s1.copy(), s2.copy(), device
                    )
                    rows.append(_make_row("Spateo_Rigid", t_sp, c_rigid_sp, c1_ref=target_sp))
                    rows.append(_make_row("Spateo_Nonrigid", t_sp, c_nonrigid_sp, c1_ref=target_sp))
                except Exception as e:
                    print(f"  Spateo failed: {e}")

            # --- Ours ---
            try:
                _sid2 = None
                _dfolder = None
                if sample_id_groups is not None and j < len(sample_id_groups):
                    _sid2 = sample_id_groups[j][i + 1]
                if dataset_folders is not None and j < len(dataset_folders):
                    _dfolder = dataset_folders[j]

                c2_rigid_o, c2_final_o, t_o = run_ours(
                    s1.copy(), s2.copy(), config, device, label_key,
                    dataset_folder=_dfolder or "DLPFC_sample1",
                    sample_id2=_sid2,
                )
                rows.append(_make_row("INSTA-Rigid", t_o, c2_rigid_o))
                rows.append(_make_row("INSTA-Nonrigid", t_o, c2_final_o))
            except Exception as e:
                print(f"  Ours failed: {e}")
                import traceback
                traceback.print_exc()

            # --- Ours (Joint: Encoder + Decoder + Discriminator) ---
            if run_joint:
                try:
                    jcfg = joint_config if joint_config is not None else JointConfig()
                    c2_rigid_j, c2_final_j, t_j = run_ours_joint(
                        s1.copy(), s2.copy(), config, jcfg, device,
                    )
                    rows.append(_make_row("INSTA-Joint-Rigid", t_j, c2_rigid_j))
                    rows.append(_make_row("INSTA-Joint-Nonrigid", t_j, c2_final_j))
                except Exception as e:
                    print(f"  Ours (Joint) failed: {e}")
                    import traceback
                    traceback.print_exc()

    return pd.DataFrame(rows)


# ============================================================================
# Summary and visualization
# ============================================================================


def print_summary(df: pd.DataFrame) -> None:
    """Print per-pair, per-sample, and overall accuracy summaries."""

    has_nn = "Accuracy_NN" in df.columns

    # ------ Per-Pair (OT) ------
    pivot = df.pivot_table(index=["Sample", "Pair"], columns="Method", values="Accuracy")
    print("\n" + "=" * 80)
    print("=== Per-Pair OT Accuracy ===")
    print("=" * 80)
    print(pivot.to_string(float_format="{:.4f}".format))

    if has_nn:
        pivot_nn = df.pivot_table(index=["Sample", "Pair"], columns="Method", values="Accuracy_NN")
        print("\n" + "=" * 80)
        print("=== Per-Pair NN Accuracy ===")
        print("=" * 80)
        print(pivot_nn.to_string(float_format="{:.4f}".format))

    # ------ Per-Sample Mean ------
    sample_mean = df.groupby(["Sample", "Method"])["Accuracy"].mean().unstack("Method")
    print("\n" + "=" * 80)
    print("=== Per-Sample Mean (OT) ===")
    print("=" * 80)
    print(sample_mean.to_string(float_format="{:.4f}".format))

    if has_nn:
        sample_mean_nn = df.groupby(["Sample", "Method"])["Accuracy_NN"].mean().unstack("Method")
        print("\n" + "=" * 80)
        print("=== Per-Sample Mean (NN) ===")
        print("=" * 80)
        print(sample_mean_nn.to_string(float_format="{:.4f}".format))

    # ------ Overall Mean +/- Std for all metrics ------
    metric_defs = [
        ("Accuracy", "OT Accuracy", "higher=better"),
        ("Accuracy_NN", "NN Accuracy", "higher=better"),
        ("Ratio", "Ratio", "lower=better"),
        ("CLC", "CLC", "higher=better"),
        ("LISI", "LISI", "lower=better"),
        ("Silhouette", "Silhouette", "higher=better"),
        ("Chamfer", "Chamfer", "lower=better"),
    ]

    for col, name, direction in metric_defs:
        if col not in df.columns:
            continue
        overall = df.groupby("Method")[col].agg(["mean", "std"])
        print("\n" + "=" * 80)
        print(f"=== Overall Mean +/- Std ({name}, {direction}) ===")
        print("=" * 80)
        for method, row in overall.iterrows():
            print(f"  {method:20s}: {row['mean']:.4f} +/- {row['std']:.4f}")


def plot_comparison(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Grouped bar chart comparing all methods across metrics."""

    method_order = [
        "No-align", "PASTE", "SPACEL", "STalign",
        "Spateo_Rigid", "Spateo_Nonrigid",
        "INSTA-Rigid", "INSTA-Nonrigid",
    ]
    present = [m for m in method_order if m in df["Method"].unique()]

    # Per-method colours
    palette = {
        "No-align": "#999999", "PASTE": "#e6a532", "SPACEL": "#5ba355",
        "STalign": "#d35b5b", "Spateo_Rigid": "#7caed6", "Spateo_Nonrigid": "#4a86b8",
        "INSTA-Rigid": "#d98cd9", "INSTA-Nonrigid": "#9933cc",
    }

    # Build list of metrics present in the DataFrame
    metrics = []
    titles = []
    if "Accuracy" in df.columns:
        metrics.append("Accuracy"); titles.append("OT Acc \u2191")
    if "Accuracy_NN" in df.columns:
        metrics.append("Accuracy_NN"); titles.append("NN Acc \u2191")
    if "CLC" in df.columns:
        metrics.append("CLC"); titles.append("CLC \u2191")
    if "Silhouette" in df.columns:
        metrics.append("Silhouette"); titles.append("Silhouette \u2191")
    if "Ratio" in df.columns:
        metrics.append("Ratio"); titles.append("Ratio \u2193")
    if "LISI" in df.columns:
        metrics.append("LISI"); titles.append("LISI \u2193")
    if "Chamfer" in df.columns:
        metrics.append("Chamfer"); titles.append("Chamfer \u2193")

    n_metrics = len(metrics)
    n_methods = len(present)

    # Compute per-method means for each metric
    means = df.groupby("Method")[metrics].mean()

    # Single figure with grouped bars: x-axis = metrics, grouped by method
    fig, ax = plt.subplots(figsize=(2 + 1.8 * n_metrics, 5))

    bar_width = 0.8 / n_methods
    x_base = np.arange(n_metrics)

    for mi, method in enumerate(present):
        if method not in means.index:
            continue
        vals = [means.loc[method, m] for m in metrics]
        offset = (mi - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x_base + offset, vals, width=bar_width,
                       color=palette.get(method, "#888888"),
                       label=method.replace("_", " "), edgecolor="white", linewidth=0.5)
        # Value labels on top of bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x_base)
    ax.set_xticklabels(titles, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_ylim(0, min(ax.get_ylim()[1] + 0.1, 1.05))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\u2705 Plot saved to {save_path}")
    plt.show()


# ============================================================================
# Generic dataset benchmark (sample_data format)
# ============================================================================


def benchmark_dataset(
    dataset: str,
    config: PipelineConfig,
    device: str = "cuda",
    label_key: str = "original_domain",
    run_paste: bool = True,
    run_spateo: bool = True,
    run_spacel: bool = True,
    run_stalign: bool = True,
) -> pd.DataFrame:
    """Run all methods on a single dataset (sample_data format).

    Loads slices from ``{data_dir}/{dataset}/sample_data/``, preprocesses,
    and runs all methods on each consecutive pair.

    Returns:
        DataFrame with all metric columns.
    """
    from inr_align.config import SLICE_ORDER
    from inr_align.run import load_slices, preprocess_slices

    if dataset not in SLICE_ORDER:
        print(f"  WARNING: {dataset} not in SLICE_ORDER, skipping.")
        return pd.DataFrame()

    slice_names = SLICE_ORDER[dataset]
    print(f"\n  Loading {dataset}: {slice_names}")
    try:
        slices_raw = load_slices(dataset, config.data_dir, slice_names)
        slices = preprocess_slices(slices_raw, config.n_top_genes, config.pca_key)
    except Exception as e:
        print(f"  Failed to load {dataset}: {e}")
        return pd.DataFrame()

    # Check label_key exists
    if label_key not in slices[0].obs.columns:
        # Try common alternatives
        for alt in ["original_domain", "cell_type", "celltype", "cluster", "label"]:
            if alt in slices[0].obs.columns:
                label_key = alt
                break
        else:
            print(f"  WARNING: No label column found in {dataset}, skipping.")
            return pd.DataFrame()

    # Wrap as single group for benchmark_all
    layer_groups = [slices]
    df = benchmark_all(
        layer_groups, config, device=device,
        label_key=label_key, label_map=None,
        run_paste=run_paste, run_spateo=run_spateo,
        run_spacel=run_spacel, run_stalign=run_stalign,
    )
    # Rename Sample column -> Dataset
    df["Dataset"] = dataset
    df = df.drop(columns=["Sample"])
    return df


# ============================================================================
# Entry point
# ============================================================================


def _load_dlpfc_layer_groups(
    data_dir: str = "./Data",
    sample_groups: Optional[List[List[str]]] = None,
) -> List[List]:
    """Load DLPFC original_data for benchmark."""
    if sample_groups is None:
        sample_groups = DLPFC_SAMPLE_GROUPS

    layer_groups = []
    for i, group in enumerate(sample_groups):
        slices = []
        folder = f"DLPFC_sample{i + 1}"
        for sample_id in group:
            path = f"{data_dir}/{folder}/original_data/{sample_id}.h5ad"
            adata = sc.read_h5ad(path)
            slices.append(adata)
            print(f"  Loaded {sample_id}: {adata.shape}")
        layer_groups.append(slices)
    return layer_groups


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    config = PipelineConfig(dataset="DLPFC_sample1", data_dir="./Data")
    # -------- Tune hyperparameters here --------
    # config.train.epochs = 100
    # config.train.lam_jacobian = 0.005

    print("Loading DLPFC data...")
    layer_groups = _load_dlpfc_layer_groups(config.data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    df = benchmark_all(layer_groups, config, device=device)
    print_summary(df)
    plot_comparison(df, "benchmark_results.png")

    # Save CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("\n\u2705 Results saved to benchmark_results.csv")

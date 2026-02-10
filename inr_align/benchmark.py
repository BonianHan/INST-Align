"""Multi-method benchmark: PASTE / Spateo / SPACEL / STalign / Ours / No-align.

Compares all methods on the same dataset using two accuracy metrics:

1. **OT Accuracy** — PASTE-style transport-plan weighted label match.
2. **NN Accuracy** — Bidirectional nearest-neighbour label match (iSTBench-style).

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
from inr_align.metrics import coords_to_pi, mapping_accuracy_nn_bidi, mapping_accuracy_paste, sparse_P_to_dense_pi
from inr_align.model import (
    DeformationNet, UnifiedCostMatcher, adaptive_icp,
    normalize_expression, ExprField,
)
from inr_align.train import apply_model, train
from inr_align.utils import detect_grid_spacing, normalize_coordinates


# ============================================================================
# DLPFC label map (PASTE convention)
# ============================================================================

DLPFC_LABEL_MAP = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6, "WM": 7}


# ============================================================================
# Individual method runners
# ============================================================================


def run_no_align_baseline(
    slice1,
    slice2,
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """No alignment — raw spatial coordinates → OT pi → accuracy.

    Returns:
        ``(acc_ot, acc_nn, ratio, elapsed_time)``.
    """
    start = time.time()
    coords1 = slice1.obsm["spatial"]
    coords2 = slice2.obsm["spatial"]
    pi = coords_to_pi(coords1, coords2)
    elapsed = time.time() - start

    labels1 = slice1.obs[label_key]
    labels2 = slice2.obs[label_key]
    acc_ot = mapping_accuracy_paste(labels1, labels2, pi, label_map)
    acc_nn, ratio = mapping_accuracy_nn_bidi(
        np.asarray(labels1), np.asarray(labels2), coords1, coords2,
    )
    return acc_ot, acc_nn, ratio, elapsed


def run_paste_baseline(
    slice1,
    slice2,
    alpha: float = 0.1,
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """PASTE pairwise alignment.

    Returns:
        ``(acc_ot, acc_nn, ratio, elapsed_time)``.
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

    labels1 = slice1.obs[label_key]
    labels2 = slice2.obs[label_key]
    acc_ot = mapping_accuracy_paste(labels1, labels2, pi, label_map)

    # PASTE aligns via pi, not explicit coordinates; use original coords for NN
    coords1 = slice1.obsm["spatial"]
    coords2 = slice2.obsm["spatial"]
    acc_nn, ratio = mapping_accuracy_nn_bidi(
        np.asarray(labels1), np.asarray(labels2), coords1, coords2,
    )
    return acc_ot, acc_nn, ratio, elapsed


def run_spateo_baseline(
    slice1,
    slice2,
    device: str = "cuda",
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """Spateo morpho_align (rigid + nonrigid).

    Returns:
        ``((acc_ot, acc_nn, ratio), (acc_ot, acc_nn, ratio), elapsed_time)``.
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

    # aligned_slices[1] is the aligned source
    coords_rigid = aligned_slices[1].obsm["align_spatial_rigid"]
    coords_nonrigid = aligned_slices[1].obsm["align_spatial_nonrigid"]
    target_coords = aligned_slices[0].obsm["align_spatial"]

    pi_rigid = coords_to_pi(target_coords, coords_rigid)
    pi_nonrigid = coords_to_pi(target_coords, coords_nonrigid)

    labels1 = slice1.obs[label_key]
    labels2 = slice2.obs[label_key]
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)

    acc_rigid_ot = mapping_accuracy_paste(labels1, labels2, pi_rigid, label_map)
    acc_nonrigid_ot = mapping_accuracy_paste(labels1, labels2, pi_nonrigid, label_map)

    acc_rigid_nn, ratio_rigid = mapping_accuracy_nn_bidi(l1, l2, target_coords, coords_rigid)
    acc_nonrigid_nn, ratio_nonrigid = mapping_accuracy_nn_bidi(l1, l2, target_coords, coords_nonrigid)

    return (acc_rigid_ot, acc_rigid_nn, ratio_rigid), (acc_nonrigid_ot, acc_nonrigid_nn, ratio_nonrigid), elapsed


def run_spacel_baseline(
    slice1,
    slice2,
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """SPACEL Scube.align (graph-based alignment).

    Runs in a separate ``spacel`` conda environment via subprocess,
    because SPACEL requires ``torch<=1.13`` which is incompatible with
    the main environment.

    Returns:
        ``(acc_ot, acc_nn, ratio, elapsed_time)``.
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

    pi = coords_to_pi(coords1, coords2)

    labels1 = slice1.obs[label_key]
    labels2 = slice2.obs[label_key]
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)

    acc_ot = mapping_accuracy_paste(labels1, labels2, pi, label_map)
    acc_nn, ratio = mapping_accuracy_nn_bidi(l1, l2, coords1, coords2)

    return acc_ot, acc_nn, ratio, elapsed


def run_stalign_baseline(
    slice1,
    slice2,
    device: str = "cuda",
    dx: int = 30,
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """STalign LDDMM diffeomorphic registration.

    Returns:
        ``(acc_ot, acc_nn, ratio, elapsed_time)``.
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

    pi = coords_to_pi(coords1, coords2_aligned)

    labels1 = slice1.obs[label_key]
    labels2 = slice2.obs[label_key]
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)

    acc_ot = mapping_accuracy_paste(labels1, labels2, pi, label_map)
    acc_nn, ratio = mapping_accuracy_nn_bidi(l1, l2, coords1, coords2_aligned)

    return acc_ot, acc_nn, ratio, elapsed


def run_ours(
    slice1,
    slice2,
    config: PipelineConfig,
    device: str = "cuda",
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
    pretrained_expr_field=None,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float]:
    """Our method: adaptive_icp + INR deformation.

    Args:
        pretrained_expr_field: Pre-trained ``PretrainedExprField``
            (from :func:`~inr_align.run.pretrain_expr_field_pipeline`).

    Returns:
        ``((acc_ot, acc_nn, ratio), (acc_ot, acc_nn, ratio), elapsed_time)``.
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
        print(f"  Grid detected: spacing=({spacing_x:.4f}, {spacing_y:.4f}), snap_to_grid=True")
    else:
        print(f"  Non-grid data: continuous coordinates, snap_to_grid=False")

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

    # Model
    model = DeformationNet(
        config.model,
        grid_mode=is_grid,
        grid_spacing=[spacing_x, spacing_y] if is_grid else None,
        grid_origin=origin,
    ).to(device)

    matcher = UnifiedCostMatcher(config.matcher)

    # ExprField (canonical consistency)
    expr_field_model = pretrained_expr_field.model if pretrained_expr_field is not None else None

    result = train(
        model, matcher, x1, emb1, x2, emb2, config.train,
        expr_field=expr_field_model,
        lam_canonical=config.expr_field.lam_canonical if config.use_expr_field else 0.0,
    )

    # Apply
    model.eval()
    x2_def = apply_model(model, x2, snap_to_grid=is_grid)
    coords2_final = x2_def.cpu().numpy() * std + mean
    coords2_rigid_denorm = coords2_rigid * std + mean

    elapsed = time.time() - start

    # PI matrices
    # 1. Rigid: spatial EMD
    pi_rigid = coords_to_pi(coords1, coords2_rigid_denorm)

    # 2. Spatial-only: spatial EMD after deformation
    pi_spatial = coords_to_pi(coords1, coords2_final)

    labels1 = slice1.obs[label_key]
    labels2 = slice2.obs[label_key]
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)

    acc_rigid_ot = mapping_accuracy_paste(labels1, labels2, pi_rigid, label_map)
    acc_spatial_ot = mapping_accuracy_paste(labels1, labels2, pi_spatial, label_map)

    acc_rigid_nn, ratio_rigid = mapping_accuracy_nn_bidi(l1, l2, coords1, coords2_rigid_denorm)
    acc_spatial_nn, ratio_spatial = mapping_accuracy_nn_bidi(l1, l2, coords1, coords2_final)

    return (acc_rigid_ot, acc_rigid_nn, ratio_rigid), (acc_spatial_ot, acc_spatial_nn, ratio_spatial), elapsed


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
) -> pd.DataFrame:
    """Run all methods on all sample groups.

    Args:
        layer_groups: ``layer_groups[j][i]`` is an AnnData for sample
            group *j*, slice *i*.
        config: Pipeline config (for our method's hyper-parameters).
        device: CUDA or CPU.
        label_key: Label column name.
        label_map: Optional label → int mapping for accuracy.
            ``None`` uses generic label equality.
        run_paste: Whether to include PASTE baseline.
        run_spateo: Whether to include Spateo baseline.
        run_spacel: Whether to include SPACEL baseline.
        run_stalign: Whether to include STalign baseline.

    Returns:
        DataFrame with columns ``[Sample, Pair, Method, Time, Accuracy, Accuracy_NN, Ratio]``.
        ``Accuracy`` is OT-based (PASTE-style), ``Accuracy_NN`` is bidirectional NN,
        ``Ratio`` is ``abs(log2(min(N1,N2)/n_unique))`` measuring collapse.
    """
    if config is None:
        config = PipelineConfig()

    rows = []

    for j in range(len(layer_groups)):
        # Pre-process group for ExprField joint pre-training
        _pretrained_ef = None
        if config.use_expr_field:
            from inr_align.run import pretrain_expr_field_pipeline
            from inr_align.utils import normalize_coordinates as _norm_coords

            _expr_group_slices = [s.copy() for s in layer_groups[j]]
            for ad_ in _expr_group_slices:
                if "counts" not in ad_.layers:
                    ad_.layers["counts"] = ad_.X.copy()
                sc.pp.normalize_total(ad_)
                sc.pp.log1p(ad_)
                if "highly_variable" not in ad_.var.columns:
                    sc.pp.highly_variable_genes(ad_, n_top_genes=config.n_top_genes)
            st.align.group_pca(_expr_group_slices, pca_key=config.pca_key)

            all_coords = [s.obsm[config.spatial_key] for s in _expr_group_slices]
            coords_norm_list, mean_, std_ = _norm_coords(all_coords)
            _pretrained_ef = pretrain_expr_field_pipeline(
                _expr_group_slices, coords_norm_list, config, device,
            )

        for i in range(len(layer_groups[j]) - 1):
            print(f"\n{'=' * 60}")
            print(f"Sample {j}, Pair {i}")
            print(f"{'=' * 60}")

            s1 = layer_groups[j][i].copy()
            s2 = layer_groups[j][i + 1].copy()

            # --- No-align ---
            acc_ot_na, acc_nn_na, ratio_na, t_na = run_no_align_baseline(s1, s2, label_key, label_map)
            rows.append({"Sample": j, "Pair": i, "Method": "No-align", "Time": t_na, "Accuracy": acc_ot_na, "Accuracy_NN": acc_nn_na, "Ratio": ratio_na})
            print(f"  No-align:      OT={acc_ot_na:.4f}  NN={acc_nn_na:.4f}  Ratio={ratio_na:.4f}")

            # --- PASTE ---
            if run_paste:
                try:
                    acc_ot_p, acc_nn_p, ratio_p, t_p = run_paste_baseline(s1.copy(), s2.copy(), alpha=0.1, label_key=label_key, label_map=label_map)
                    rows.append({"Sample": j, "Pair": i, "Method": "PASTE", "Time": t_p, "Accuracy": acc_ot_p, "Accuracy_NN": acc_nn_p, "Ratio": ratio_p})
                    print(f"  PASTE:         OT={acc_ot_p:.4f}  NN={acc_nn_p:.4f}  Ratio={ratio_p:.4f}")
                except Exception as e:
                    print(f"  PASTE failed: {e}")

            # --- SPACEL ---
            if run_spacel:
                try:
                    acc_ot_sc, acc_nn_sc, ratio_sc, t_sc = run_spacel_baseline(s1.copy(), s2.copy(), label_key, label_map)
                    rows.append({"Sample": j, "Pair": i, "Method": "SPACEL", "Time": t_sc, "Accuracy": acc_ot_sc, "Accuracy_NN": acc_nn_sc, "Ratio": ratio_sc})
                    print(f"  SPACEL:        OT={acc_ot_sc:.4f}  NN={acc_nn_sc:.4f}  Ratio={ratio_sc:.4f}")
                except Exception as e:
                    print(f"  SPACEL failed: {e}")

            # --- STalign ---
            if run_stalign:
                try:
                    acc_ot_st, acc_nn_st, ratio_st, t_st = run_stalign_baseline(s1.copy(), s2.copy(), device, label_key=label_key, label_map=label_map)
                    rows.append({"Sample": j, "Pair": i, "Method": "STalign", "Time": t_st, "Accuracy": acc_ot_st, "Accuracy_NN": acc_nn_st, "Ratio": ratio_st})
                    print(f"  STalign:       OT={acc_ot_st:.4f}  NN={acc_nn_st:.4f}  Ratio={ratio_st:.4f}")
                except Exception as e:
                    print(f"  STalign failed: {e}")

            # --- Spateo ---
            if run_spateo:
                try:
                    (acc_sr_ot, acc_sr_nn, ratio_sr), (acc_snr_ot, acc_snr_nn, ratio_snr), t_s = run_spateo_baseline(s1.copy(), s2.copy(), device, label_key, label_map)
                    rows.append({"Sample": j, "Pair": i, "Method": "Spateo_Rigid", "Time": t_s, "Accuracy": acc_sr_ot, "Accuracy_NN": acc_sr_nn, "Ratio": ratio_sr})
                    rows.append({"Sample": j, "Pair": i, "Method": "Spateo_Nonrigid", "Time": t_s, "Accuracy": acc_snr_ot, "Accuracy_NN": acc_snr_nn, "Ratio": ratio_snr})
                    print(f"  Spateo rigid:  OT={acc_sr_ot:.4f}  NN={acc_sr_nn:.4f}  Ratio={ratio_sr:.4f}")
                    print(f"  Spateo nonrig: OT={acc_snr_ot:.4f}  NN={acc_snr_nn:.4f}  Ratio={ratio_snr:.4f}")
                except Exception as e:
                    print(f"  Spateo failed: {e}")

            # --- Ours ---
            try:
                (acc_r_ot, acc_r_nn, ratio_r), (acc_sp_ot, acc_sp_nn, ratio_sp), t_o = run_ours(
                    s1.copy(), s2.copy(), config, device, label_key, label_map,
                    pretrained_expr_field=_pretrained_ef,
                )
                rows.append({"Sample": j, "Pair": i, "Method": "Ours_Rigid", "Time": t_o, "Accuracy": acc_r_ot, "Accuracy_NN": acc_r_nn, "Ratio": ratio_r})
                rows.append({"Sample": j, "Pair": i, "Method": "Ours_Spatial", "Time": t_o, "Accuracy": acc_sp_ot, "Accuracy_NN": acc_sp_nn, "Ratio": ratio_sp})
                print(f"  Ours rigid:    OT={acc_r_ot:.4f}  NN={acc_r_nn:.4f}  Ratio={ratio_r:.4f}")
                print(f"  Ours spatial:  OT={acc_sp_ot:.4f}  NN={acc_sp_nn:.4f}  Ratio={ratio_sp:.4f}")
            except Exception as e:
                print(f"  Ours failed: {e}")
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

    # ------ Overall Mean ± Std ------
    overall = df.groupby("Method")["Accuracy"].agg(["mean", "std"])
    print("\n" + "=" * 80)
    print("=== Overall Mean ± Std (OT) ===")
    print("=" * 80)
    for method, row in overall.iterrows():
        print(f"  {method:20s}: {row['mean']:.4f} ± {row['std']:.4f}")

    if has_nn:
        overall_nn = df.groupby("Method")["Accuracy_NN"].agg(["mean", "std"])
        print("\n" + "=" * 80)
        print("=== Overall Mean ± Std (NN) ===")
        print("=" * 80)
        for method, row in overall_nn.iterrows():
            print(f"  {method:20s}: {row['mean']:.4f} ± {row['std']:.4f}")

    if "Ratio" in df.columns:
        overall_ratio = df.groupby("Method")["Ratio"].agg(["mean", "std"])
        print("\n" + "=" * 80)
        print("=== Overall Mean ± Std (Ratio, lower=better) ===")
        print("=" * 80)
        for method, row in overall_ratio.iterrows():
            print(f"  {method:20s}: {row['mean']:.4f} ± {row['std']:.4f}")


def plot_comparison(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Box-plot comparison of all methods (OT and NN accuracy side by side)."""

    has_nn = "Accuracy_NN" in df.columns

    # Consistent ordering
    method_order = [
        "No-align", "PASTE", "SPACEL", "STalign",
        "Spateo_Rigid", "Spateo_Nonrigid",
        "Ours_Rigid", "Ours_Spatial",
    ]
    present = [m for m in method_order if m in df["Method"].unique()]

    if has_nn:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        # OT accuracy
        sns.boxplot(data=df[df["Method"].isin(present)], x="Method", y="Accuracy", order=present, ax=axes[0])
        sns.stripplot(data=df[df["Method"].isin(present)], x="Method", y="Accuracy",
                      order=present, color="black", size=4, alpha=0.6, ax=axes[0])
        axes[0].set_title("OT Accuracy (PASTE-style)", fontsize=14)
        axes[0].set_ylabel("Mapping Accuracy (OT)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=30)

        # NN accuracy
        sns.boxplot(data=df[df["Method"].isin(present)], x="Method", y="Accuracy_NN", order=present, ax=axes[1])
        sns.stripplot(data=df[df["Method"].isin(present)], x="Method", y="Accuracy_NN",
                      order=present, color="black", size=4, alpha=0.6, ax=axes[1])
        axes[1].set_title("NN Accuracy (Bidirectional)", fontsize=14)
        axes[1].set_ylabel("Mapping Accuracy (NN)")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df[df["Method"].isin(present)], x="Method", y="Accuracy", order=present, ax=ax)
        sns.stripplot(data=df[df["Method"].isin(present)], x="Method", y="Accuracy",
                      order=present, color="black", size=4, alpha=0.6, ax=ax)
        ax.set_title("Alignment Accuracy: Method Comparison", fontsize=14)
        ax.set_ylabel("Mapping Accuracy")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

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
        DataFrame with columns ``[Dataset, Pair, Method, Time, Accuracy, Accuracy_NN, Ratio]``.
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
    # Rename Sample column → Dataset
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

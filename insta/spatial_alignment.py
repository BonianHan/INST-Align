"""Multi-method Experiment 1 runner: baselines / INST-Align / no-align.

Compares all methods on the same dataset using the paper's alignment metrics:

1. **OT Accuracy** — PASTE-style transport-plan weighted label match.
2. **NN Accuracy** — Bidirectional nearest-neighbour label match (iSTBench-style).
3. **Chamfer** — Symmetric Chamfer distance (geometric alignment).

Run directly::

    python run_spatial_alignment.py

Or import::

    from insta.spatial_alignment import run_spatial_alignment_all, print_summary
"""

from __future__ import annotations

import signal
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

from insta.config import DLPFC_SAMPLE_GROUPS, JointConfig, PipelineConfig
from insta.metrics import chamfer_distance, mapping_accuracy_nn_bidi
from insta.model import (
    DeformationNet, UnifiedCostMatcher, adaptive_icp,
    build_joint_models, build_knn_graph, ExprINR,
)
from insta.trainer import apply_model, train
from insta.utils import (
    coords_to_pi,
    detect_grid_spacing,
    mapping_accuracy_paste,
    normalize_coordinates,
    sparse_P_to_dense_pi,
)


# ============================================================================
# DLPFC label map (PASTE convention)
# ============================================================================

DLPFC_LABEL_MAP = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6, "WM": 7}


# ============================================================================
# Timeout helper (SIGALRM, Linux only)
# ============================================================================

class _MethodTimeout(Exception):
    pass


def _run_with_timeout(func, *args, timeout: int = 300, method_name: str = "", **kwargs):
    """Run func(*args, **kwargs) with a hard SIGALRM timeout.

    Returns ``(result, error_str)``.  On timeout or exception, result is None.
    """
    def _handler(signum, frame):
        raise _MethodTimeout(f"{method_name} timed out after {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result, None
    except _MethodTimeout as e:
        return None, str(e)
    except Exception as e:
        signal.alarm(0)
        return None, str(e)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


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
    import torch
    backend = ot.backend.TorchBackend()
    use_gpu = torch.cuda.is_available()
    start = time.time()
    pi = pst.pairwise_align(
        slice1, slice2,
        alpha=alpha, G_init=pi0, norm=True, verbose=False,
        use_gpu=use_gpu, backend=backend,
    )
    elapsed = time.time() - start
    return pi, elapsed


def run_paste2_baseline(
    slice1,
    slice2,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """PASTE2 partial pairwise alignment (BenchmarkST).

    Returns:
        ``(pi, elapsed_time)``.  PASTE2 produces a transport plan.
    """
    from paste2.model_selection import select_overlap_fraction
    from paste2.PASTE2 import partial_pairwise_align

    start = time.time()
    s = select_overlap_fraction(slice1, slice2, alpha=alpha)
    pi = partial_pairwise_align(
        slice1, slice2, s, alpha=alpha,
        armijo=False, dissimilarity="kl", norm=True,
        return_obj=False, verbose=False,
    )
    elapsed = time.time() - start
    return pi, elapsed


def run_gpsa_baseline(
    slice1,
    slice2,
    n_spatial_dims: int = 2,
    n_latent_gps: int = 5,
    m_X_per_view: int = 200,
    m_G: int = 200,
    n_epochs: int = 500,
    lr: float = 1e-2,
    device: str = "cuda",
) -> Tuple[np.ndarray, float]:
    """GPSA Gaussian Process Spatial Alignment (BenchmarkST).

    Returns:
        ``(coords2_aligned, elapsed_time)``.
    """
    from gpsa import VariationalGPSA, rbf_kernel

    coords1 = slice1.obsm["spatial"].astype(np.float32)
    coords2 = slice2.obsm["spatial"].astype(np.float32)

    # Standardize spatial coordinates (GPSA is sensitive to input scale)
    coords_all_raw = np.concatenate([coords1, coords2], axis=0)
    sp_mean = coords_all_raw.mean(axis=0)
    sp_std = coords_all_raw.std(axis=0) + 1e-8
    coords1_s = (coords1 - sp_mean) / sp_std
    coords2_s = (coords2 - sp_mean) / sp_std

    # Combine expression data (use PCA if available, else raw)
    pca_key = "X_pca"
    if pca_key in slice1.obsm and pca_key in slice2.obsm:
        Y1 = slice1.obsm[pca_key].astype(np.float32)
        Y2 = slice2.obsm[pca_key].astype(np.float32)
        n_features = min(Y1.shape[1], Y2.shape[1])
        Y1 = Y1[:, :n_features]
        Y2 = Y2[:, :n_features]
    else:
        import scipy.sparse as sp
        X1 = slice1.X.toarray() if sp.issparse(slice1.X) else np.array(slice1.X)
        X2 = slice2.X.toarray() if sp.issparse(slice2.X) else np.array(slice2.X)
        n_features = min(X1.shape[1], X2.shape[1])
        Y1 = X1[:, :n_features].astype(np.float32)
        Y2 = X2[:, :n_features].astype(np.float32)

    # Standardize features
    Y_all = np.concatenate([Y1, Y2], axis=0)
    y_mean = Y_all.mean(axis=0)
    y_std = Y_all.std(axis=0) + 1e-8
    Y1 = (Y1 - y_mean) / y_std
    Y2 = (Y2 - y_mean) / y_std

    # GPSA expects torch tensors
    spatial_all = torch.tensor(np.concatenate([coords1_s, coords2_s], axis=0))
    outputs_all = torch.tensor(np.concatenate([Y1, Y2], axis=0))

    data_dict = {
        "expression": {
            "spatial_coords": spatial_all,
            "outputs": outputs_all,
            "n_samples_list": [len(coords1), len(coords2)],
        }
    }

    n_gps = min(n_latent_gps, n_features)
    n_min = min(len(coords1), len(coords2))

    model = VariationalGPSA(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=min(m_X_per_view, n_min),
        m_G=min(m_G, n_min),
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps={"expression": n_gps},
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        fixed_view_idx=0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_spatial = {"expression": spatial_all.to(device)}
    Y_dict = {"expression": outputs_all.to(device)}

    start = time.time()
    print_interval = max(1, n_epochs // 5)
    for ep in range(n_epochs):
        optimizer.zero_grad()
        G_means, G_samples, F_latent, F_observed = model.forward(
            X_spatial, model.view_idx, model.Ns,
        )
        loss = model.loss_fn(
            {"expression": {"outputs": Y_dict["expression"]}},
            F_observed,
        )
        loss.backward()
        optimizer.step()
        if ep % print_interval == 0:
            print(f"    GPSA epoch {ep}/{n_epochs}  loss={loss.item():.4f}")
    elapsed = time.time() - start

    # Extract aligned coordinates and convert back to original scale
    with torch.no_grad():
        model.eval()
        G_means, *_ = model.forward(
            X_spatial, model.view_idx, model.Ns,
        )
        aligned_coords_s = G_means["expression"].cpu().numpy()
        n1 = len(coords1)
        coords2_aligned = aligned_coords_s[n1:] * sp_std + sp_mean

    return coords2_aligned, elapsed


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


def run_stalign_baseline(
    slice1,
    slice2,
    device: str = "cuda",
    dx: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """STalign LDDMM diffeomorphic registration.

    Returns:
        ``(coords2_aligned, elapsed_time)``.
    """
    from STalign import STalign as STalign_module

    coords1 = slice1.obsm["spatial"].astype(np.float64)
    coords2 = slice2.obsm["spatial"].astype(np.float64)

    # Scale coordinates to ~1000-unit range so STalign's internal
    # velocity field parameters (a=500, step=250) work correctly.
    # Small ranges like DLPFC (~127) cause the velocity grid to
    # have only 1 point, triggering index-out-of-bounds errors.
    all_coords = np.vstack([coords1, coords2])
    max_range = max(
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(),
    )
    TARGET_RANGE = 2000.0
    scale = TARGET_RANGE / max(max_range, 1e-8)
    offset = all_coords.min(axis=0)

    coords1_s = (coords1 - offset) * scale
    coords2_s = (coords2 - offset) * scale

    # Auto-compute dx on the scaled coordinates
    if dx is None:
        dx = max(TARGET_RANGE / 20.0, 1.0)

    # Target = slice1, Source = slice2
    xJ = np.array(coords1_s[:, 0])
    yJ = np.array(coords1_s[:, 1])
    XJ, YJ, J, _ = STalign_module.rasterize(xJ, yJ, dx=dx)

    xI = np.array(coords2_s[:, 0])
    yI = np.array(coords2_s[:, 1])
    XI, YI, I, _ = STalign_module.rasterize(xI, yI, dx=dx)

    start = time.time()
    params = {"niter": 3000, "device": device, "epV": 50}
    out = STalign_module.LDDMM([YI, XI], I, [YJ, XJ], J, **params)
    A, v, xv = out["A"], out["v"], out["xv"]

    dtype = A.dtype
    points_np = np.stack([yI, xI], axis=1)
    points_tensor = torch.tensor(points_np, dtype=dtype).to(device)
    tpointsI = STalign_module.transform_points_source_to_target(xv, v, A, points_tensor)
    if tpointsI.is_cuda:
        tpointsI = tpointsI.cpu()
    elapsed = time.time() - start

    # Scale back to original coordinate space
    aligned_s = tpointsI[:, [1, 0]].numpy()  # (N, 2) in (x, y)
    coords2_aligned = aligned_s / scale + offset
    return coords2_aligned, elapsed


def run_ours(
    slice1,
    slice2,
    config: PipelineConfig,
    device: str = "cuda",
    label_key: str = "original_domain",
) -> Tuple[np.ndarray, np.ndarray, float, Optional["ExprINR"]]:
    """Our method: adaptive ICP + two-phase DeformNet with dual ExprINR.

    Returns:
        ``(coords2_rigid_denorm, coords2_final, elapsed_time, expr_inr_s1)``.
    """
    # Preprocessing — skip if already done by a dataset runner.
    already_preprocessed = all(
        "counts" in ad_.layers and config.pca_key in ad_.obsm
        for ad_ in [slice1, slice2]
    )
    if not already_preprocessed:
        for ad_ in [slice1, slice2]:
            if "counts" not in ad_.layers:
                ad_.layers["counts"] = ad_.X.copy()
            sc.pp.normalize_total(ad_)
            sc.pp.log1p(ad_)
            if "highly_variable" not in ad_.var.columns:
                sc.pp.highly_variable_genes(ad_, n_top_genes=config.n_top_genes)
        st.align.group_pca([slice1, slice2], pca_key=config.pca_key)
    else:
        print("  [run_ours] Data already preprocessed, skipping normalize/log1p/PCA")

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

    # HVG expression for reconstruction target (log-normalized)
    import scipy.sparse as sp
    hvg_mask1 = slice1.var["highly_variable"].values
    hvg_mask2 = slice2.var["highly_variable"].values
    X1_hvg = slice1.X[:, hvg_mask1]
    X2_hvg = slice2.X[:, hvg_mask2]
    if sp.issparse(X1_hvg):
        X1_hvg = X1_hvg.toarray()
    if sp.issparse(X2_hvg):
        X2_hvg = X2_hvg.toarray()
    hvg1 = torch.tensor(X1_hvg.astype(np.float32), device=device)
    hvg2 = torch.tensor(X2_hvg.astype(np.float32), device=device)

    # DeformNet (spatial only)
    model = DeformationNet(config.model).to(device)
    matcher = UnifiedCostMatcher(config.matcher)

    # Joint components (dual ExprINR + shared decoder)
    jcfg = config.joint
    jcfg.n_output = hvg1.shape[1]      # HVG count for decoder output
    models = build_joint_models(jcfg, device=device)

    result = train(
        model, matcher, x1, emb1, x2, emb2,
        config.train, jcfg,
        expr_inr_s1=models["expr_inr_s1"],
        expr_inr_s2=models["expr_inr_s2"],
        decoder=models["decoder"],
        hvg1=hvg1,
        hvg2=hvg2,
    )

    # Apply deformation and denormalize
    model.eval()
    x2_def = apply_model(model, x2)
    coords2_final = x2_def.cpu().numpy() * std + mean
    coords2_rigid_denorm = coords2_rigid * std + mean

    elapsed = time.time() - start
    return coords2_rigid_denorm, coords2_final, elapsed, result.expr_inr, (mean, std), result.deform, coords2_rigid


# ============================================================================
# Full spatial alignment experiment
# ============================================================================


def run_spatial_alignment_all(
    layer_groups: List[List],
    config: Optional[PipelineConfig] = None,
    device: str = "cuda",
    label_key: str = "original_domain",
    label_map: Optional[Dict] = None,
    run_paste: bool = True,
    run_paste2: bool = True,
    run_gpsa: bool = True,
    run_spateo: bool = True,
    run_stalign: bool = True,
    run_insta: bool = True,
    sample_id_groups: Optional[List[List[str]]] = None,
    dataset_folders: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run the spatial alignment experiment on all sample groups.

    Args:
        layer_groups: ``layer_groups[j][i]`` is an AnnData for sample
            group *j*, slice *i*.
        config: Pipeline config (for our method's hyper-parameters).
        device: CUDA or CPU.
        label_key: Label column name.
        label_map: Optional label -> int mapping for accuracy.
            ``None`` uses generic label equality.
        run_paste: Whether to include PASTE baseline.
        run_paste2: Whether to include PASTE2 baseline.
        run_gpsa: Whether to include GPSA baseline.
        run_spateo: Whether to include Spateo baseline.
        run_stalign: Whether to include STalign baseline.
        run_insta: Whether to include INST-Align.
        sample_id_groups: ``sample_id_groups[j][i]`` is the sample ID string
            for group *j*, slice *i*.  Used to load Splane embeddings.
        dataset_folders: ``dataset_folders[j]`` is the dataset folder name
            (e.g. ``"DLPFC_sample1"``) for group *j*.

    Returns:
        ``(DataFrame, encoders_dict)`` where DataFrame has columns
        ``[Sample, Pair, Method, Time, Accuracy, Accuracy_NN, Chamfer]``
        and ``encoders_dict`` maps
        ``(sample_idx, pair_idx)`` to the trained ``ExprINR``.
    """
    if config is None:
        config = PipelineConfig()

    rows = []
    inr_models: Dict[Tuple[int, int], ExprINR] = {}
    norm_params: Dict[Tuple[int, int], Tuple] = {}  # (mean, std) per pair
    deform_models: Dict[Tuple[int, int], DeformationNet] = {}
    rigid_coords_norm: Dict[Tuple[int, int], np.ndarray] = {}  # coords2_rigid in norm space

    for j in range(len(layer_groups)):
        for i in range(len(layer_groups[j]) - 1):
            print(f"\n{'=' * 60}")
            print(f"Sample {j}, Pair {i}")
            print(f"{'=' * 60}")

            s1 = layer_groups[j][i].copy()
            s2 = layer_groups[j][i + 1].copy()

            _coords1 = s1.obsm["spatial"]
            _coords2 = s2.obsm["spatial"]
            _l1_raw = np.asarray(s1.obs[label_key])
            _l2_raw = np.asarray(s2.obs[label_key])

            # Filter out NA/nan labels for metric computation
            _valid1 = np.array([str(l) not in ("NA", "nan", "None", "") for l in _l1_raw])
            _valid2 = np.array([str(l) not in ("NA", "nan", "None", "") for l in _l2_raw])
            n_filtered = (~_valid1).sum() + (~_valid2).sum()
            if n_filtered > 0:
                print(f"  Filtered {n_filtered} NA labels "
                      f"(s1: {(~_valid1).sum()}, s2: {(~_valid2).sum()})")

            def _make_row(method, t, c2_aligned, c1_ref=None):
                """Compute Experiment 1 metrics and build result row."""
                c1 = c1_ref if c1_ref is not None else _coords1
                # Use filtered labels/coords for label-based metrics
                c1_f = c1[_valid1]
                c2_f = c2_aligned[_valid2]
                l1_f = _l1_raw[_valid1]
                l2_f = _l2_raw[_valid2]
                l1_s1 = s1.obs[label_key].iloc[np.where(_valid1)[0]]
                l2_s2 = s2.obs[label_key].iloc[np.where(_valid2)[0]]
                pi = coords_to_pi(c1_f, c2_f)
                acc_ot = mapping_accuracy_paste(l1_s1, l2_s2, pi, label_map)
                acc_nn = mapping_accuracy_nn_bidi(l1_f, l2_f, c1_f, c2_f)
                # Chamfer uses all points (geometric, no labels)
                cham = chamfer_distance(c1, c2_aligned)
                row = {
                    "Sample": j, "Pair": i, "Method": method, "Time": t,
                    "N1": s1.shape[0], "N2": s2.shape[0],
                    "Accuracy": acc_ot, "Accuracy_NN": acc_nn, "Chamfer": cham,
                }
                print(f"  {method:16s} OT={acc_ot:.4f}  NN={acc_nn:.4f}  "
                      f"Cham={cham:.4f}")
                return row

            # --- No-align ---
            rows.append(_make_row("No-align", 0.0, _coords2))

            # --- PASTE ---
            if run_paste:
                result, err = _run_with_timeout(
                    run_paste_baseline, s1.copy(), s2.copy(),
                    timeout=300, method_name="PASTE", alpha=0.1,
                )
                if result is not None:
                    pi_paste, t_p = result
                    # pi shape: (n1, n2). Transpose so rows = slice2 points,
                    # then normalize rows to get weighted avg of slice1 coords.
                    pi_T = pi_paste.T  # (n2, n1)
                    pi_T_norm = pi_T / (pi_T.sum(1, keepdims=True) + 1e-12)
                    c2_paste_aligned = pi_T_norm @ _coords1  # (n2, 2)
                    row_p = _make_row("PASTE", t_p, c2_paste_aligned)
                    row_p["Accuracy"] = mapping_accuracy_paste(
                        s1.obs[label_key], s2.obs[label_key], pi_paste, label_map
                    )
                    rows.append(row_p)
                else:
                    print(f"  PASTE skipped: {err}")

            # --- PASTE2 ---
            if run_paste2:
                result, err = _run_with_timeout(
                    run_paste2_baseline, s1.copy(), s2.copy(),
                    timeout=300, method_name="PASTE2", alpha=0.1,
                )
                if result is not None:
                    pi_paste2, t_p2 = result
                    # pi shape: (n1, n2). Transpose for slice2 -> slice1 mapping.
                    pi2_T = pi_paste2.T  # (n2, n1)
                    pi2_T_norm = pi2_T / (pi2_T.sum(1, keepdims=True) + 1e-12)
                    c2_paste2_aligned = pi2_T_norm @ _coords1  # (n2, 2)
                    row_p2 = _make_row("PASTE2", t_p2, c2_paste2_aligned)
                    row_p2["Accuracy"] = mapping_accuracy_paste(
                        s1.obs[label_key], s2.obs[label_key], pi_paste2, label_map
                    )
                    rows.append(row_p2)
                else:
                    print(f"  PASTE2 skipped: {err}")

            # --- GPSA ---
            if run_gpsa:
                result, err = _run_with_timeout(
                    run_gpsa_baseline, s1.copy(), s2.copy(),
                    timeout=600, method_name="GPSA", device=device,
                )
                if result is not None:
                    c2_gpsa, t_gpsa = result
                    rows.append(_make_row("GPSA", t_gpsa, c2_gpsa))
                else:
                    print(f"  GPSA skipped: {err}")

            # --- STalign ---
            if run_stalign:
                result, err = _run_with_timeout(
                    run_stalign_baseline, s1.copy(), s2.copy(),
                    timeout=600, method_name="STalign", device=device,
                )
                if result is not None:
                    c2_st, t_st = result
                    rows.append(_make_row("STalign", t_st, c2_st))
                else:
                    print(f"  STalign skipped: {err}")

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

            # --- Ours (two-phase DeformNet + ExprINR) ---
            if run_insta:
                try:
                    c2_rigid_o, c2_final_o, t_o, expr_inr_o, norm_ms, deform_o, c2_rigid_norm_o = run_ours(
                        s1.copy(), s2.copy(), config, device, label_key,
                    )
                    rows.append(_make_row("INSTA-Rigid", t_o, c2_rigid_o))
                    rows.append(_make_row("INSTA-Nonrigid", t_o, c2_final_o))
                    if expr_inr_o is not None:
                        inr_models[(j, i)] = expr_inr_o
                        norm_params[(j, i)] = norm_ms
                        deform_models[(j, i)] = deform_o
                        rigid_coords_norm[(j, i)] = c2_rigid_norm_o
                except Exception as e:
                    print(f"  Ours failed: {e}")
                    import traceback
                    traceback.print_exc()

    return pd.DataFrame(rows), inr_models, norm_params, deform_models, rigid_coords_norm


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
        "No-align", "PASTE", "PASTE2", "GPSA", "STalign",
        "Spateo_Rigid", "Spateo_Nonrigid",
        "INSTA-Rigid", "INSTA-Nonrigid",
    ]
    present = [m for m in method_order if m in df["Method"].unique()]

    # Per-method colours
    palette = {
        "No-align": "#999999", "PASTE": "#e6a532", "PASTE2": "#c4882a",
        "GPSA": "#5cb85c", "STalign": "#d35b5b",
        "Spateo_Rigid": "#7caed6", "Spateo_Nonrigid": "#4a86b8",
        "INSTA-Rigid": "#d98cd9", "INSTA-Nonrigid": "#9933cc",
    }

    # Build list of metrics present in the DataFrame
    metrics = []
    titles = []
    if "Accuracy" in df.columns:
        metrics.append("Accuracy"); titles.append("OT Acc \u2191")
    if "Accuracy_NN" in df.columns:
        metrics.append("Accuracy_NN"); titles.append("NN Acc \u2191")
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
    ax.set_title("Spatial Alignment Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    ax.set_ylim(0, min(ax.get_ylim()[1] + 0.1, 1.05))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\u2705 Plot saved to {save_path}")
    plt.show()


# ============================================================================
# Generic dataset runner (sample_data format)
# ============================================================================


def run_spatial_alignment_dataset(
    dataset: str,
    config: PipelineConfig,
    device: str = "cuda",
    label_key: str = "original_domain",
    run_paste: bool = True,
    run_paste2: bool = True,
    run_gpsa: bool = True,
    run_spateo: bool = True,
    run_stalign: bool = True,
    run_insta: bool = True,
) -> pd.DataFrame:
    """Run the spatial alignment experiment on one dataset.

    Loads slices from ``{data_dir}/{dataset}/sample_data/``, preprocesses,
    and runs all methods on each consecutive pair.

    Returns:
        DataFrame with all metric columns.
    """
    from insta.config import SLICE_ORDER
    from insta.pipeline import load_slices, preprocess_slices

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
        for alt in ["original_domain", "cellbin_SpatialDomain", "cell_type", "celltype", "cluster", "label"]:
            if alt in slices[0].obs.columns:
                label_key = alt
                break
        else:
            print(f"  WARNING: No label column found in {dataset}, skipping.")
            return pd.DataFrame()

    # Wrap as single group for the all-dataset runner.
    layer_groups = [slices]
    df, *_ = run_spatial_alignment_all(
        layer_groups, config, device=device,
        label_key=label_key, label_map=None,
        run_paste=run_paste, run_paste2=run_paste2,
        run_gpsa=run_gpsa, run_spateo=run_spateo,
        run_stalign=run_stalign, run_insta=run_insta,
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
    """Load DLPFC original_data for spatial alignment evaluation."""
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

    df, *_ = run_spatial_alignment_all(layer_groups, config, device=device)
    print_summary(df)
    plot_comparison(df, "spatial_alignment_results.png")

    # Save CSV
    df.to_csv("spatial_alignment_results.csv", index=False)
    print("\nResults saved to spatial_alignment_results.csv")

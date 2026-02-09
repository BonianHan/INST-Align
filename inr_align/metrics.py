"""Evaluation metrics for spatial transcriptomics alignment.

Provides five families of metrics:

1. **NN accuracy** — nearest-neighbour label agreement.
2. **OT accuracy** — label overlap weighted by optimal transport plan.
3. **PASTE-style accuracy** — uses a precomputed transport plan ``pi``.
4. **CLC** — Contextual Label Consistency.
5. **iSTBench metrics** — Accuracy + Ratio across consecutive slices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import ot
import pandas as pd
import scipy.spatial
import scipy.spatial.distance
from numpy.typing import NDArray
from scipy.spatial import cKDTree


# ============================================================================
# 1. NN accuracy
# ============================================================================


def mapping_accuracy_nn(
    labels1: NDArray,
    labels2: NDArray,
    coords1: NDArray,
    coords2_aligned: NDArray,
    k: int = 1,
) -> float:
    """Nearest-neighbour label accuracy (unidirectional: source → target).

    For each point in *coords2_aligned*, find its nearest neighbour in
    *coords1* and check whether the labels match.
    """
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)
    tree = cKDTree(coords1)
    _, idx = tree.query(coords2_aligned, k=k)

    if k == 1:
        matched = l1[idx]
    else:
        matched = np.array([pd.Series(l1[i]).mode().iloc[0] for i in idx])

    return float(np.mean(l2 == matched))


def mapping_accuracy_nn_bidi(
    labels1: NDArray,
    labels2: NDArray,
    coords1: NDArray,
    coords2_aligned: NDArray,
) -> Tuple[float, float]:
    """Bidirectional nearest-neighbour label accuracy + collapse ratio.

    Computes NN label match in both directions (1→2 and 2→1) and
    returns the average.  This matches the iSTBench R-script metric.

    Also computes **Ratio** = ``abs(log2(min(N1, N2) / n_unique_matches))``
    which measures many-to-one collapse.  Lower is better (0 = perfect
    1-to-1 matching).

    Returns:
        ``(accuracy, ratio)``
    """
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2_aligned, dtype=np.float64)

    # Forward: for each point in c1, find nearest in c2
    tree2 = cKDTree(c2)
    _, idx_fwd = tree2.query(c1, k=1)
    match_fwd = float(np.mean(l1 == l2[idx_fwd]))

    # Backward: for each point in c2, find nearest in c1
    tree1 = cKDTree(c1)
    _, idx_bwd = tree1.query(c2, k=1)
    match_bwd = float(np.mean(l2 == l1[idx_bwd]))

    accuracy = (match_fwd + match_bwd) / 2.0

    # Ratio: many-to-one collapse measure (iSTBench)
    n_unique = len(np.unique(idx_fwd))
    n_min = min(len(c1), len(c2))
    ratio = float(np.abs(np.log2(n_min / max(n_unique, 1))))

    return accuracy, ratio


# ============================================================================
# 2. OT accuracy (with subsampling for large datasets)
# ============================================================================


def mapping_accuracy_ot(
    labels1: NDArray,
    labels2: NDArray,
    coords1: NDArray,
    coords2_aligned: NDArray,
    max_samples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """OT-based label accuracy with optional subsampling.

    Returns:
        Dict with keys ``accuracy``, ``upper_bound``, ``normalized``.
    """
    l1_all = np.asarray(labels1)
    l2_all = np.asarray(labels2)
    c1_all = np.asarray(coords1)
    c2_all = np.asarray(coords2_aligned)
    n1, n2 = len(l1_all), len(l2_all)

    rng = np.random.default_rng(seed)
    if n1 > max_samples or n2 > max_samples:
        idx1 = rng.choice(n1, min(n1, max_samples), replace=False)
        idx2 = rng.choice(n2, min(n2, max_samples), replace=False)
        c1, l1 = c1_all[idx1], l1_all[idx1]
        c2, l2 = c2_all[idx2], l2_all[idx2]
    else:
        c1, l1, c2, l2 = c1_all, l1_all, c2_all, l2_all

    m1, m2 = len(l1), len(l2)
    label_match = (l1[:, None] == l2[None, :]).astype(np.float32)
    spatial_dist = scipy.spatial.distance.cdist(c1, c2).astype(np.float32)
    a = np.ones(m1, dtype=np.float32) / m1
    b = np.ones(m2, dtype=np.float32) / m2

    pi_spatial = ot.emd(a, b, spatial_dist)
    acc = float(np.sum(pi_spatial * label_match))

    label_dist = (1.0 - label_match).astype(np.float32)
    pi_label = ot.emd(a, b, label_dist)
    upper = float(np.sum(pi_label * label_match))

    return {"accuracy": acc, "upper_bound": upper, "normalized": acc / (upper + 1e-8)}


# ============================================================================
# 3. PASTE-style mapping accuracy (uses pi directly)
# ============================================================================


def mapping_accuracy_paste(
    labels1,
    labels2,
    pi: NDArray,
    label_map: Optional[Dict] = None,
) -> float:
    """Label accuracy weighted by a precomputed transport plan *pi*.

    This is the metric used by the PASTE paper.  *pi* has shape
    ``(N1, N2)`` and its marginals should sum to ``1/N1`` and ``1/N2``.

    Args:
        labels1: Series or array of labels for slice 1.
        labels2: Series or array of labels for slice 2.
        pi: ``(N1, N2)`` transport plan.
        label_map: Optional mapping from label → int.  If ``None``,
            uses generic label equality.
    """
    if label_map is not None:
        # Use provided mapping (e.g. {'L1':1, ..., 'WM':7})
        if hasattr(labels1, "map"):
            l1_int = np.matrix(labels1.map(label_map)).T
            l2_int = np.matrix(labels2.map(label_map)).T
        else:
            l1_int = np.matrix([label_map[l] for l in labels1]).T
            l2_int = np.matrix([label_map[l] for l in labels2]).T
        same = scipy.spatial.distance_matrix(l1_int, l2_int) == 0
    else:
        # Generic: label equality
        l1 = np.asarray(labels1)
        l2 = np.asarray(labels2)
        same = l1[:, None] == l2[None, :]

    return float(np.sum(pi * same))


# ============================================================================
# 4. CLC (Contextual Label Consistency)
# ============================================================================


def calculate_clc(
    labels1: NDArray,
    labels2: NDArray,
    coords1: NDArray,
    coords2_aligned: NDArray,
    k_percent: float = 0.025,
) -> float:
    """Contextual Label Consistency.

    Checks whether the neighbourhood structure is preserved after
    alignment and labels are consistent.
    """
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)
    c1 = np.asarray(coords1)
    c2 = np.asarray(coords2_aligned)

    n1, n2 = len(c1), len(c2)
    k = max(1, int(((n1 + n2) / 2) * k_percent))

    tree1 = cKDTree(c1)
    _, idx_map = tree1.query(c2, k=1)

    label_match = l1[idx_map] == l2

    tree2 = cKDTree(c2)
    _, nbr_moving = tree2.query(c2, k=k + 1)
    nbr_moving = nbr_moving[:, 1:]

    _, nbr_fixed = tree1.query(c1, k=k + 1)
    nbr_fixed = nbr_fixed[:, 1:]

    scores = np.zeros(n2)
    for i in range(n2):
        if not label_match[i]:
            continue
        mapped_nbrs = set(idx_map[nbr_moving[i]])
        fixed_nbrs = set(nbr_fixed[idx_map[i]])
        overlap = len(mapped_nbrs & fixed_nbrs)
        scores[i] = overlap / k

    return float(scores.mean())


# ============================================================================
# 5. Comprehensive evaluation
# ============================================================================


def evaluate_alignment(
    slice1,
    slice2,
    coords2_aligned: NDArray,
    label_key: str = "original_domain",
    max_samples: int = 3000,
    seed: int = 42,
) -> Dict[str, float]:
    """Run all metrics on an aligned pair.

    Args:
        slice1: AnnData with ``obsm['spatial']`` and ``obs[label_key]``.
        slice2: AnnData with ``obs[label_key]``.
        coords2_aligned: ``(N2, 2)`` aligned source coordinates.
        label_key: Column name in ``.obs`` holding ground-truth labels.

    Returns:
        Dict with ``nn_accuracy``, ``ot_accuracy``, ``ot_upper_bound``,
        ``ot_normalized``, ``clc``.
    """
    coords1 = np.asarray(slice1.obsm["spatial"])
    coords2 = np.asarray(coords2_aligned)
    labels1 = np.asarray(slice1.obs[label_key])
    labels2 = np.asarray(slice2.obs[label_key])

    acc_nn = mapping_accuracy_nn(labels1, labels2, coords1, coords2)
    ot_res = mapping_accuracy_ot(labels1, labels2, coords1, coords2, max_samples=max_samples, seed=seed)
    clc = calculate_clc(labels1, labels2, coords1, coords2)

    return {
        "nn_accuracy": acc_nn,
        "ot_accuracy": ot_res["accuracy"],
        "ot_upper_bound": ot_res["upper_bound"],
        "ot_normalized": ot_res["normalized"],
        "clc": clc,
    }


# ============================================================================
# Transport-plan helpers
# ============================================================================


def coords_to_pi(coords1: NDArray, coords2: NDArray) -> NDArray:
    """Spatial-only OT transport plan via earth-mover distance."""
    n1, n2 = len(coords1), len(coords2)
    dist = scipy.spatial.distance.cdist(coords1, coords2)
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2
    return ot.emd(a, b, dist)


def sparse_P_to_dense_pi(
    topk_idx,
    weights,
    N1: int,
    N2: int,
) -> NDArray:
    """Convert sparse P-matrix to dense transport plan with Sinkhorn normalization.

    Args:
        topk_idx: ``(N2, K)`` tensor — top-k neighbour indices in slice 1.
        weights: ``(N2, K)`` tensor — softmax weights.
        N1: Number of spots in slice 1.
        N2: Number of spots in slice 2.

    Returns:
        ``(N1, N2)`` dense transport plan.
    """
    import torch

    if isinstance(topk_idx, torch.Tensor):
        topk_idx_np = topk_idx.cpu().numpy()
        weights_np = weights.cpu().numpy()
    else:
        topk_idx_np = np.asarray(topk_idx)
        weights_np = np.asarray(weights)

    K = topk_idx_np.shape[1]

    P_dense = np.zeros((N2, N1), dtype=np.float64)
    row_indices = np.repeat(np.arange(N2), K)
    col_indices = topk_idx_np.ravel()
    values = weights_np.ravel()
    np.add.at(P_dense, (row_indices, col_indices), values)

    pi = P_dense.T  # (N1, N2)

    # Sinkhorn normalization
    a = np.ones(N1) / N1
    b = np.ones(N2) / N2
    for _ in range(50):
        row_sums = pi.sum(axis=1, keepdims=True)
        pi = pi * (a[:, None] / np.maximum(row_sums, 1e-12))
        col_sums = pi.sum(axis=0, keepdims=True)
        pi = pi * (b[None, :] / np.maximum(col_sums, 1e-12))

    return pi


# ============================================================================
# iSTBench metrics
# ============================================================================


def compute_istbench_metrics(
    slices_list: list,
    slice_names: List[str],
    label_key: str = "original_domain",
) -> pd.DataFrame:
    """Compute iSTBench-style pairwise metrics across consecutive slices.

    Each slice in *slices_list* should have ``obsm['spatial_aligned']``
    (or ``obsm['spatial']`` as fallback).
    """
    results = []
    for i in range(len(slices_list) - 1):
        s1, s2 = slices_list[i], slices_list[i + 1]
        c1 = s1.obsm.get("spatial_aligned", s1.obsm["spatial"])
        c2 = s2.obsm.get("spatial_aligned", s2.obsm["spatial"])

        tree2 = cKDTree(c2)
        _, idx_fwd = tree2.query(c1, k=1)
        tree1 = cKDTree(c1)
        _, idx_bwd = tree1.query(c2, k=1)

        if label_key in s1.obs and label_key in s2.obs:
            d1 = s1.obs[label_key].values
            d2 = s2.obs[label_key].values
            match_fwd = np.mean(d1 == d2[idx_fwd])
            match_bwd = np.mean(d2 == d1[idx_bwd])
            accuracy = (match_fwd + match_bwd) / 2

            ot_met = mapping_accuracy_ot(
                s1.obs[label_key], s2.obs[label_key], c1, c2, label_key
            ) if False else {"nn_accuracy": float(np.nan), "accuracy": float(np.nan), "normalized": float(np.nan)}
            # Compute OT metrics properly
            ot_met = mapping_accuracy_ot(
                np.asarray(s1.obs[label_key]),
                np.asarray(s2.obs[label_key]),
                c1, c2,
            )
        else:
            accuracy = float(np.nan)
            ot_met = {"accuracy": float(np.nan), "normalized": float(np.nan)}

        n_unique = len(np.unique(idx_fwd))
        ratio = np.abs(np.log2(min(len(c1), len(c2)) / n_unique))

        nn_acc = mapping_accuracy_nn(
            np.asarray(s1.obs[label_key]) if label_key in s1.obs else np.array([]),
            np.asarray(s2.obs[label_key]) if label_key in s2.obs else np.array([]),
            c1, c2,
        ) if label_key in s1.obs else float(np.nan)

        results.append({
            "slice_pair": f"s{i + 1}-s{i + 2}",
            "Accuracy": accuracy,
            "Ratio": ratio,
            "NN_Acc": nn_acc,
            "OT_Acc": ot_met["accuracy"],
            "OT_Norm": ot_met["normalized"],
        })

    return pd.DataFrame(results)

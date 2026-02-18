"""Evaluation metrics for spatial transcriptomics alignment.

Provides six families of metrics:

1. **NN accuracy** — nearest-neighbour label agreement.
2. **OT accuracy** — label overlap weighted by optimal transport plan.
3. **PASTE-style accuracy** — uses a precomputed transport plan ``pi``.
4. **CLC** — Contextual Label Consistency.
5. **Gene-expression similarity** — NN-matched PCA cosine similarity.
6. **iSTBench metrics** — Accuracy + Ratio across consecutive slices.
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

    For each cell *i* in the moving slice (coords2), find its nearest
    match *i'* in the fixed slice (coords1).  If the labels agree, check
    what fraction of *i*'s spatial neighbours in coords2, once mapped to
    coords1, fall within *i'*'s spatial neighbourhood in coords1.

    Higher CLC means better preservation of local spatial context.
    """
    l1 = np.asarray(labels1)
    l2 = np.asarray(labels2)
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2_aligned, dtype=np.float64)

    n1, n2 = len(c1), len(c2)
    k = max(1, int(((n1 + n2) / 2) * k_percent))

    # Mapping: each cell in c2 → nearest cell in c1
    tree1 = cKDTree(c1)
    _, idx_map = tree1.query(c2, k=1)           # (n2,)

    label_match = (l1[idx_map] == l2)            # (n2,) bool

    # Neighbours in the moving slice (c2)
    tree2 = cKDTree(c2)
    _, nbr_moving = tree2.query(c2, k=k + 1)
    nbr_moving = nbr_moving[:, 1:]              # (n2, k)

    # Neighbours in the fixed slice (c1)
    _, nbr_fixed = tree1.query(c1, k=k + 1)
    nbr_fixed = nbr_fixed[:, 1:]               # (n1, k)

    # Vectorised: map neighbours of i in c2 → c1
    mapped_nbrs = idx_map[nbr_moving]           # (n2, k) — mapped neighbour indices
    fixed_nbrs = nbr_fixed[idx_map]             # (n2, k) — fixed-space neighbours of i'

    # Sort rows for fast set-intersection via searchsorted
    mapped_nbrs_sorted = np.sort(mapped_nbrs, axis=1)
    fixed_nbrs_sorted = np.sort(fixed_nbrs, axis=1)

    # Count overlap per row
    scores = np.zeros(n2)
    for i in range(n2):
        if not label_match[i]:
            continue
        # np.intersect1d is faster than set for sorted arrays
        overlap = np.intersect1d(mapped_nbrs_sorted[i], fixed_nbrs_sorted[i], assume_unique=False).size
        scores[i] = overlap / k

    return float(scores.mean())


# ============================================================================
# 5. Gene-expression similarity (NN-matched PCA cosine similarity)
# ============================================================================


def gene_expr_similarity(
    coords1: NDArray,
    coords2_aligned: NDArray,
    emb1: NDArray,
    emb2: NDArray,
) -> float:
    """Mean cosine similarity of NN-matched PCA embeddings after alignment.

    For each cell in *coords2_aligned*, find its nearest neighbour in
    *coords1*, then compute the cosine similarity between their PCA
    embeddings.  Higher means the aligned cells have more similar
    expression profiles.

    Args:
        coords1: ``(N1, 2)`` target spatial coordinates.
        coords2_aligned: ``(N2, 2)`` aligned source spatial coordinates.
        emb1: ``(N1, D)`` PCA embeddings for slice 1.
        emb2: ``(N2, D)`` PCA embeddings for slice 2.

    Returns:
        Mean cosine similarity in ``[0, 1]``.
    """
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2_aligned, dtype=np.float64)
    e1 = np.asarray(emb1, dtype=np.float64)
    e2 = np.asarray(emb2, dtype=np.float64)

    tree1 = cKDTree(c1)
    _, idx = tree1.query(c2, k=1)  # (N2,)

    # Cosine similarity per matched pair
    matched_e1 = e1[idx]  # (N2, D)
    dot = np.sum(matched_e1 * e2, axis=1)
    norm1 = np.linalg.norm(matched_e1, axis=1) + 1e-12
    norm2 = np.linalg.norm(e2, axis=1) + 1e-12
    cos_sim = dot / (norm1 * norm2)

    return float(np.mean(cos_sim))


# ============================================================================
# 6. LISI (Local Inverse Simpson's Index)
# ============================================================================


def compute_lisi(
    coords1: NDArray,
    coords2_aligned: NDArray,
    labels1: NDArray,
    labels2: NDArray,
    k: int = 30,
) -> float:
    """Domain LISI on combined aligned point cloud.

    Combines both slices into one point cloud and computes the Local
    Inverse Simpson's Index for domain labels.  If alignment is good,
    same domains from both slices overlap and local neighbourhoods are
    dominated by a single domain (LISI close to 1).

    Args:
        coords1: ``(N1, 2)`` reference spatial coordinates.
        coords2_aligned: ``(N2, 2)`` aligned source spatial coordinates.
        labels1: ``(N1,)`` domain labels for slice 1.
        labels2: ``(N2,)`` domain labels for slice 2.
        k: Number of neighbours for LISI computation.

    Returns:
        Median LISI score (**lower is better**, minimum 1.0).
    """
    combined_coords = np.vstack([
        np.asarray(coords1, dtype=np.float64),
        np.asarray(coords2_aligned, dtype=np.float64),
    ])
    combined_labels = np.concatenate([np.asarray(labels1), np.asarray(labels2)])

    # Filter invalid labels
    valid = np.array([str(l).strip() not in ("", "nan") for l in combined_labels])
    if valid.sum() < k + 1:
        return float("nan")

    coords = combined_coords[valid]
    labels = combined_labels[valid]
    n = len(coords)
    k_actual = min(k, n - 1)

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k_actual + 1)
    indices = indices[:, 1:]  # exclude self

    unique_labels, label_codes = np.unique(labels, return_inverse=True)
    nbr_codes = label_codes[indices]  # (N, k)

    # Simpson's index per cell: sum of squared label frequencies
    simpson = np.zeros(n)
    for c in range(len(unique_labels)):
        freq = np.mean(nbr_codes == c, axis=1)
        simpson += freq ** 2

    lisi = 1.0 / np.maximum(simpson, 1e-12)
    return float(np.median(lisi))


# ============================================================================
# 7. Silhouette Score
# ============================================================================


def compute_silhouette(
    coords1: NDArray,
    coords2_aligned: NDArray,
    labels1: NDArray,
    labels2: NDArray,
    max_samples: int = 5000,
    seed: int = 42,
) -> float:
    """Silhouette score on domain labels in combined aligned space.

    Combines both slices and computes the silhouette score for domain
    labels.  Higher score means better spatial separation of domains
    after alignment.

    Args:
        coords1: ``(N1, 2)`` reference spatial coordinates.
        coords2_aligned: ``(N2, 2)`` aligned source spatial coordinates.
        labels1: ``(N1,)`` domain labels for slice 1.
        labels2: ``(N2,)`` domain labels for slice 2.
        max_samples: Subsample to this size for speed.
        seed: Random seed for subsampling.

    Returns:
        Silhouette score in ``[-1, 1]`` (**higher is better**).
    """
    from sklearn.metrics import silhouette_score

    combined_coords = np.vstack([
        np.asarray(coords1, dtype=np.float64),
        np.asarray(coords2_aligned, dtype=np.float64),
    ])
    combined_labels = np.concatenate([np.asarray(labels1), np.asarray(labels2)])

    valid = np.array([str(l).strip() not in ("", "nan") for l in combined_labels])
    coords = combined_coords[valid]
    labels = combined_labels[valid]

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(coords) < 10:
        return float("nan")

    # Subsample for speed
    rng = np.random.default_rng(seed)
    if len(coords) > max_samples:
        idx = rng.choice(len(coords), max_samples, replace=False)
        coords = coords[idx]
        labels = labels[idx]

    return float(silhouette_score(coords, labels))


# ============================================================================
# 8. Chamfer Distance
# ============================================================================


def chamfer_distance(
    coords1: NDArray,
    coords2_aligned: NDArray,
) -> float:
    """Symmetric Chamfer distance between two point clouds.

    For each point in one cloud, finds the nearest point in the other
    and averages.  Measures geometric alignment quality.

    Args:
        coords1: ``(N1, 2)`` reference spatial coordinates.
        coords2_aligned: ``(N2, 2)`` aligned source spatial coordinates.

    Returns:
        Mean Chamfer distance (**lower is better**).
    """
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2_aligned, dtype=np.float64)

    tree1 = cKDTree(c1)
    tree2 = cKDTree(c2)

    d12, _ = tree2.query(c1, k=1)  # c1 -> nearest in c2
    d21, _ = tree1.query(c2, k=1)  # c2 -> nearest in c1

    return float((d12.mean() + d21.mean()) / 2)


# ============================================================================
# 9. Comprehensive evaluation
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
    lisi = compute_lisi(coords1, coords2, labels1, labels2)
    sil = compute_silhouette(coords1, coords2, labels1, labels2)
    chamfer = chamfer_distance(coords1, coords2)

    return {
        "nn_accuracy": acc_nn,
        "ot_accuracy": ot_res["accuracy"],
        "ot_upper_bound": ot_res["upper_bound"],
        "ot_normalized": ot_res["normalized"],
        "clc": clc,
        "lisi": lisi,
        "silhouette": sil,
        "chamfer": chamfer,
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

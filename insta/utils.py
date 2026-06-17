"""Utility helpers: grid detection, coordinate normalization, transport plans."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import ot
import scipy.spatial
import scipy.spatial.distance
from numpy.typing import NDArray


# ============================================================================
# Grid detection
# ============================================================================

def detect_grid_spacing(
    coords: NDArray,
    tol: float = 0.1,
) -> Tuple[Optional[float], Optional[float], bool, Optional[NDArray]]:
    """Detect whether *coords* lie on a regular grid.

    Args:
        coords: ``(N, 2)`` spatial coordinates.
        tol: Coefficient-of-variation threshold below which we declare
            a grid along that axis.

    Returns:
        ``(spacing_x, spacing_y, is_grid, origin)`` where *is_grid* is
        ``True`` when a grid is detected and the remaining values are
        ``None`` otherwise.
    """
    coords = np.asarray(coords)
    unique_x = np.sort(np.unique(coords[:, 0]))
    unique_y = np.sort(np.unique(coords[:, 1]))

    if len(unique_x) < 3 or len(unique_y) < 3:
        return None, None, False, None

    dx = np.diff(unique_x)
    dy = np.diff(unique_y)
    cv_x = dx.std() / (dx.mean() + 1e-8)
    cv_y = dy.std() / (dy.mean() + 1e-8)

    is_grid = (cv_x < tol) and (cv_y < tol)

    if is_grid:
        spacing_x = float(np.median(dx))
        spacing_y = float(np.median(dy))
        origin = np.array([unique_x[0], unique_y[0]])
        return spacing_x, spacing_y, True, origin
    else:
        return None, None, False, None


# ============================================================================
# Coordinate normalization
# ============================================================================

def normalize_coordinates(
    coords_list: List[NDArray],
) -> Tuple[List[NDArray], NDArray, NDArray]:
    """Global z-score normalization across multiple coordinate arrays.

    Args:
        coords_list: A list of ``(N_i, 2)`` arrays.

    Returns:
        ``(normalized_list, mean, std)`` where *mean* / *std* are shared
        across all arrays.
    """
    all_coords = np.vstack(coords_list)
    mean = all_coords.mean(axis=0)
    std = all_coords.std(axis=0) + 1e-8
    normalized = [(c - mean) / std for c in coords_list]
    return normalized, mean, std


def denormalize_coordinates(
    coords: NDArray,
    mean: NDArray,
    std: NDArray,
) -> NDArray:
    """Reverse :func:`normalize_coordinates`."""
    return coords * std + mean


# ============================================================================
# Transport-plan utilities
# ============================================================================

def mapping_accuracy_paste(
    labels1,
    labels2,
    pi: NDArray,
    label_map: Optional[Dict] = None,
) -> float:
    """Label accuracy weighted by a precomputed transport plan."""
    if label_map is not None:
        if hasattr(labels1, "map"):
            l1_int = np.matrix(labels1.map(label_map)).T
            l2_int = np.matrix(labels2.map(label_map)).T
        else:
            l1_int = np.matrix([label_map[l] for l in labels1]).T
            l2_int = np.matrix([label_map[l] for l in labels2]).T
        same = scipy.spatial.distance_matrix(l1_int, l2_int) == 0
    else:
        l1 = np.asarray(labels1)
        l2 = np.asarray(labels2)
        same = l1[:, None] == l2[None, :]

    return float(np.sum(pi * same))


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
    """Convert sparse P-matrix top-k weights to a dense transport plan."""
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

    pi = P_dense.T

    a = np.ones(N1) / N1
    b = np.ones(N2) / N2
    for _ in range(50):
        row_sums = pi.sum(axis=1, keepdims=True)
        pi = pi * (a[:, None] / np.maximum(row_sums, 1e-12))
        col_sums = pi.sum(axis=0, keepdims=True)
        pi = pi * (b[None, :] / np.maximum(col_sums, 1e-12))

    return pi


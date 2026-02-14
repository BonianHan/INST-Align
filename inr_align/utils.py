"""Utility helpers: grid detection, coordinate normalization."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
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
# Griddata post-processing (replaces snap_to_grid)
# ============================================================================


def griddata_resample(
    coords: NDArray,
    side_length: int = 200,
) -> Tuple[NDArray, NDArray]:
    """Resample irregular coordinates onto a regular grid via linear interpolation.

    Creates a ``side_length x side_length`` grid spanning the bounding box of
    *coords*, interpolates coordinate values onto that grid, and removes
    any NaN points (outside the convex hull).

    This replaces the old ``snap_to_grid`` approach which used greedy assignment
    and produced many-to-one collisions.

    Args:
        coords: ``(N, 2)`` spatial coordinates (already deformed/aligned).
        side_length: Grid resolution per side (default 200).

    Returns:
        ``(grid_coords, valid_mask)`` where ``grid_coords`` is ``(M, 2)``
        (M ≤ side_length²) and ``valid_mask`` is ``(side_length², )`` bool.
    """
    from scipy.interpolate import griddata

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Create regular grid
    gx = np.linspace(x_min, x_max, side_length)
    gy = np.linspace(y_min, y_max, side_length)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Interpolate — we're just checking which grid points are inside the hull
    # For coordinate resampling, grid_points ARE the new coordinates
    # We use griddata to check validity by interpolating a dummy field
    dummy = np.ones(len(coords))
    interp = griddata(coords, dummy, grid_points, method="linear")
    valid_mask = ~np.isnan(interp)

    return grid_points[valid_mask], valid_mask

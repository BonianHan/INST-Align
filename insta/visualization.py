"""Visualization helpers for INST-Align examples and figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_SLICE_COLORS = (
    "#C92A2A",
    "#1864AB",
    "#2B8A3E",
    "#E67700",
    "#7048E8",
    "#0CA678",
)

DLPFC_LAYER_ORDER = ("L1", "L2", "L3", "L4", "L5", "L6", "WM")
DLPFC_LAYER_COLORS = {
    "L1": "#F56867",
    "L2": "#FEB915",
    "L3": "#56B4E9",
    "L4": "#0072B2",
    "L5": "#009E73",
    "L6": "#CC79A7",
    "WM": "#999999",
}


def _coords(adata, spatial_key: str) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} not found in AnnData.obsm")
    return np.asarray(adata.obsm[spatial_key], dtype=np.float64)


def _save(fig, save_path: Optional[os.PathLike | str], dpi: int, tight: bool = True) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(path, dpi=dpi, bbox_inches=bbox, facecolor="white")


def center_translate(ref_coords: np.ndarray, src_coords: np.ndarray) -> np.ndarray:
    """Translate source coordinates so source and reference centroids match."""
    ref = np.asarray(ref_coords, dtype=np.float64)
    src = np.asarray(src_coords, dtype=np.float64)
    return src + (ref.mean(axis=0) - src.mean(axis=0))


def _normalize_pair(
    ref_coords: np.ndarray,
    src_coords: np.ndarray,
    padding: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.vstack([ref_coords, src_coords])
    cmin = coords.min(axis=0)
    cmax = coords.max(axis=0)
    span = float((cmax - cmin).max())
    if span == 0:
        span = 1.0
    center = (cmin + cmax) / 2.0
    scale = span * (1.0 + padding)
    return (ref_coords - center) / scale + 0.5, (src_coords - center) / scale + 0.5


def plot_pair_overlay(
    ref_coords: np.ndarray,
    src_coords: np.ndarray,
    ax: Optional[plt.Axes] = None,
    ref_label: str = "Reference",
    src_label: str = "Source",
    ref_color: str = DEFAULT_SLICE_COLORS[0],
    src_color: str = DEFAULT_SLICE_COLORS[1],
    point_size: float = 3,
    alpha: float = 0.8,
) -> plt.Axes:
    """Plot two spatial point clouds on the same normalized 2D axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ref = np.asarray(ref_coords, dtype=np.float64)
    src = np.asarray(src_coords, dtype=np.float64)
    ref_n, src_n = _normalize_pair(ref, src)

    ax.scatter(
        src_n[:, 0], src_n[:, 1],
        s=point_size, c=src_color, alpha=alpha,
        edgecolors="none", rasterized=True, label=src_label,
    )
    ax.scatter(
        ref_n[:, 0], ref_n[:, 1],
        s=point_size, c=ref_color, alpha=alpha,
        edgecolors="none", rasterized=True, label=ref_label,
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def plot_before_after(
    ref_slice,
    src_slice,
    aligned_src_coords: Optional[np.ndarray] = None,
    ref_spatial_key: str = "spatial",
    src_spatial_key: str = "spatial",
    aligned_spatial_key: str = "spatial_aligned",
    ref_label: str = "Slice 1",
    src_label: str = "Slice 2",
    save_path: Optional[os.PathLike | str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot center-translated source coordinates beside INST-Align output."""
    ref_coords = _coords(ref_slice, ref_spatial_key)
    src_coords = _coords(src_slice, src_spatial_key)
    if aligned_src_coords is None:
        aligned_src_coords = _coords(src_slice, aligned_spatial_key)
    aligned_src_coords = np.asarray(aligned_src_coords, dtype=np.float64)

    centered_src = center_translate(ref_coords, src_coords)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), facecolor="white")
    plot_pair_overlay(
        ref_coords, centered_src, ax=axes[0],
        ref_label=ref_label, src_label=src_label,
    )
    plot_pair_overlay(
        ref_coords, aligned_src_coords, ax=axes[1],
        ref_label=ref_label, src_label=src_label,
    )

    axes[0].set_title("Center translated", fontsize=13)
    axes[1].set_title("INST-Align", fontsize=13)

    legend = [
        Line2D([], [], marker="o", color="w", markerfacecolor=DEFAULT_SLICE_COLORS[0],
               markersize=9, label=ref_label, linestyle="None"),
        Line2D([], [], marker="o", color="w", markerfacecolor=DEFAULT_SLICE_COLORS[1],
               markersize=9, label=src_label, linestyle="None"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, save_path, dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def _category_colors(
    labels: np.ndarray,
    color_map: Optional[Mapping[str, str]] = None,
) -> Mapping[str, str]:
    if color_map is not None:
        return color_map

    unique = [str(x) for x in sorted(set(labels.astype(str)))]
    cmap = plt.get_cmap("tab20")
    return {label: cmap(i % cmap.N) for i, label in enumerate(unique)}


def plot_3d_stack(
    slices: Sequence,
    label_key: str = "original_domain",
    spatial_key: str = "spatial_aligned",
    slice_gap: float = 1.5,
    color_map: Optional[Mapping[str, str]] = None,
    label_order: Optional[Sequence[str]] = None,
    elev: float = 30,
    azim: float = -55,
    point_size: float = 16,
    save_path: Optional[os.PathLike | str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot aligned slices as a stacked 3D reconstruction."""
    if len(slices) == 0:
        raise ValueError("slices must contain at least one AnnData object")

    coords_list = []
    for adata in slices:
        key = spatial_key if spatial_key in adata.obsm else "spatial"
        coords_list.append(_coords(adata, key))

    global_center = np.vstack(coords_list).mean(axis=0)
    all_labels = []
    for i, adata in enumerate(slices):
        if label_key in adata.obs:
            all_labels.extend(np.asarray(adata.obs[label_key]).astype(str))
        else:
            all_labels.extend([f"slice_{i + 1}"] * adata.n_obs)
    all_labels_arr = np.asarray(all_labels)
    colors = _category_colors(all_labels_arr, color_map)

    if label_order is None:
        if color_map is not None:
            label_order = list(color_map.keys())
        else:
            label_order = [str(x) for x in sorted(set(all_labels_arr))]

    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    for i in range(len(slices) - 1, -1, -1):
        adata = slices[i]
        coords = coords_list[i] - global_center
        z = np.full(len(coords), i * slice_gap)
        if label_key in adata.obs:
            labels = np.asarray(adata.obs[label_key]).astype(str)
        else:
            labels = np.full(adata.n_obs, f"slice_{i + 1}")

        for label in label_order:
            mask = labels == label
            if np.any(mask):
                ax.scatter(
                    coords[mask, 0], coords[mask, 1], z[mask],
                    c=[colors.get(str(label), "#777777")],
                    s=point_size, alpha=0.95, edgecolors="none",
                    depthshade=True, rasterized=True,
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis.line.set_visible(False)

    if len(label_order) <= 15:
        handles = [
            Line2D([], [], marker="o", color="w",
                   markerfacecolor=colors.get(str(label), "#777777"),
                   markersize=8, label=str(label), linestyle="None")
            for label in label_order
            if str(label) in colors
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    _save(fig, save_path, dpi, tight=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax

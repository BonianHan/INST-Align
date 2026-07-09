"""Visualization helpers for INST-Align examples and figures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

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


PathLike = Union[os.PathLike, str]


MOUSE_EMBRYO_OVERRIDES = {
    "epochs": 500,
    "batch_size": 4000,
    "lr": 3e-4,
    "grad_clip": 2.0,
    "lam_jacobian": 0.01,
    "lam_deform_mag": 0.0,
    "tau_min": 0.005,
    "mode": "pca",
    "inr_pretrain_epochs": 150,
    "freeze_inr_phase2": True,
    "lam_recon_phase2": 0.0,
    "warmup_fraction": 0.0,
    "scheduler_patience": 9999,
}


def _save(fig, save_path: Optional[PathLike], dpi: int, tight: bool = True) -> None:
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


def _apply_overrides(config, overrides: Mapping[str, object]) -> None:
    for name, value in overrides.items():
        if hasattr(config.train, name):
            setattr(config.train, name, value)
        elif hasattr(config.joint, name):
            setattr(config.joint, name, value)
        elif hasattr(config.matcher, name):
            setattr(config.matcher, name, value)
        elif hasattr(config.icp, name):
            setattr(config.icp, name, value)


def _preprocess_figure_slices(slices: Sequence, n_top_genes: int = 2000) -> None:
    import scanpy as sc
    import spateo as st

    for adata in slices:
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if "highly_variable" not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    st.align.group_pca(list(slices), pca_key="X_pca")


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
    save_path: Optional[PathLike] = None,
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
    save_path: Optional[PathLike] = None,
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


def plot_mouseembryo_alignment(
    ref_coords: np.ndarray,
    center_translated_coords: np.ndarray,
    aligned_coords: np.ndarray,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    dpi: int = 600,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plot the MouseEmbryo qualitative panels used in the paper."""
    color_s1 = "#CC0000"
    color_s2 = "#0044CC"

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), facecolor="white")
    panels = [
        (axes[0], ref_coords, center_translated_coords, "(a)"),
        (axes[1], ref_coords, aligned_coords, "(b)"),
    ]

    for ax, ref, query, label in panels:
        ref = np.asarray(ref, dtype=np.float64)
        query = np.asarray(query, dtype=np.float64)
        all_coords = np.vstack([ref, query])
        cmin, cmax = all_coords.min(axis=0), all_coords.max(axis=0)
        span = float((cmax - cmin).max()) * 1.08
        if span == 0:
            span = 1.0
        center = (cmin + cmax) / 2.0

        ref_plot = (ref - center) / span + 0.5
        query_plot = (query - center) / span + 0.5

        ax.scatter(
            query_plot[:, 0], query_plot[:, 1],
            s=3, c=color_s2, alpha=0.8, edgecolors="none", rasterized=True,
        )
        ax.scatter(
            ref_plot[:, 0], ref_plot[:, 1],
            s=3, c=color_s1, alpha=0.8, edgecolors="none", rasterized=True,
        )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.5, -0.06, label,
            transform=ax.transAxes, fontsize=36, fontweight="bold",
            ha="center", va="top",
        )

    legend_elements = [
        Line2D([], [], marker="o", color="w", markerfacecolor=color_s1,
               markersize=28, label="Slice 1", linestyle="None"),
        Line2D([], [], marker="o", color="w", markerfacecolor=color_s2,
               markersize=28, label="Slice 2", linestyle="None"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=2, frameon=False, fontsize=32, handletextpad=0.6, columnspacing=3.0,
    )
    plt.subplots_adjust(wspace=0.04, bottom=0.08)
    _save(fig, save_path, dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axes


def make_mouseembryo_figure(
    data_dir: str = "Data",
    save_dir: str = "assets",
    device: Optional[str] = None,
    save_name: str = "mouseembryo_figure.png",
    show: bool = True,
    epochs: Optional[int] = None,
    pretrain_epochs: Optional[int] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Run INST-Align on MouseEmbryo and draw paper panels (a) and (b)."""
    import scanpy as sc
    import torch

    from insta.config import PipelineConfig
    from insta.pipeline import align_pair

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_dir = os.path.join(data_dir, "MouseEmbryo", "sample_data")
    s1 = sc.read_h5ad(os.path.join(sample_dir, "slices1.h5ad"))
    s2 = sc.read_h5ad(os.path.join(sample_dir, "slices2.h5ad"))
    print(f"Loaded MouseEmbryo: slices1 ({s1.n_obs}) + slices2 ({s2.n_obs})")

    _preprocess_figure_slices([s1, s2])

    config = PipelineConfig(dataset="MouseEmbryo", data_dir=data_dir)
    _apply_overrides(config, MOUSE_EMBRYO_OVERRIDES)
    config.icp.icp_threshold = -1.0
    if epochs is not None:
        config.train.epochs = epochs
    if pretrain_epochs is not None:
        config.joint.inr_pretrain_epochs = pretrain_epochs

    coords1 = s1.obsm["spatial"].copy()
    coords2 = s2.obsm["spatial"].copy()
    coords2_center = center_translate(coords1, coords2)

    aligned_coords, _ = align_pair(s1, s2, config, device)

    save_path = os.path.join(save_dir, save_name)
    return plot_mouseembryo_alignment(
        coords1, coords2_center, aligned_coords,
        save_path=save_path, show=show,
    )


def align_dlpfc_sample3_consecutive(
    data_dir: str = "Data",
    device: Optional[str] = None,
    epochs: Optional[int] = None,
    pretrain_epochs: Optional[int] = None,
):
    """Align DLPFC Sample 3 consecutive slices for the paper 3D figure."""
    import scanpy as sc
    import torch

    from insta.config import DLPFC_SAMPLE_GROUPS, PipelineConfig
    from insta.pipeline import align_pair

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_idx = 2
    group = DLPFC_SAMPLE_GROUPS[sample_idx]
    folder = f"DLPFC_sample{sample_idx + 1}"

    slices = []
    for sid in group:
        path = os.path.join(data_dir, folder, "original_data", f"{sid}.h5ad")
        adata = sc.read_h5ad(path)
        print(f"Loaded {sid}: {adata.shape}")
        slices.append(adata)

    _preprocess_figure_slices(slices)

    config = PipelineConfig(dataset="DLPFC_sample3", data_dir=data_dir)
    config.icp.icp_threshold = -1.0
    if epochs is not None:
        config.train.epochs = epochs
    if pretrain_epochs is not None:
        config.joint.inr_pretrain_epochs = pretrain_epochs

    aligned = [slices[0].copy()]
    aligned[0].obsm["spatial_aligned"] = slices[0].obsm["spatial"].copy()

    for i in range(len(slices) - 1):
        print(f"\nAligning {group[i + 1]} -> {group[i]}")
        coords_aligned, result = align_pair(slices[i], slices[i + 1], config, device)
        adata = slices[i + 1].copy()
        adata.obsm["spatial_aligned"] = coords_aligned
        aligned.append(adata)
        if result is not None:
            print(f"  Done ({result.training_time:.1f}s)")

    return aligned


def plot_dlpfc3_3d(
    slices: Sequence,
    label_key: str = "original_domain",
    spatial_key: str = "spatial_aligned",
    save_path: Optional[PathLike] = None,
    elev: float = 30,
    azim: float = -55,
    show: bool = True,
    dpi: int = 600,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the DLPFC Sample 3 3D reconstruction used as panel (c)."""
    all_coords = np.vstack([adata.obsm[spatial_key] for adata in slices])
    global_mean = all_coords.mean(axis=0)

    fig = plt.figure(figsize=(14, 16), facecolor="white")
    ax = fig.add_axes([0.02, 0.08, 0.96, 0.90], projection="3d")
    ax.view_init(elev=elev, azim=azim)

    z_spacing = 1.5
    for i in range(len(slices) - 1, -1, -1):
        adata = slices[i]
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64) - global_mean
        labels = np.asarray(adata.obs[label_key]).astype(str)
        z = np.full(len(coords), i * z_spacing)

        for layer in DLPFC_LAYER_ORDER:
            mask = labels == layer
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1], z[mask],
                c=DLPFC_LAYER_COLORS[layer], s=35, alpha=1.0,
                edgecolors="none", depthshade=True, rasterized=True,
            )

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.line.set_visible(False)
    ax.yaxis.line.set_visible(False)
    ax.zaxis.line.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")
    ax.grid(False)

    legend_elements = [
        Line2D([], [], marker="o", color="w",
               markerfacecolor=DLPFC_LAYER_COLORS[layer],
               markersize=32, label=layer, linestyle="None")
        for layer in DLPFC_LAYER_ORDER
    ]
    legend = ax.legend(
        handles=legend_elements, loc="upper right", ncol=1, frameon=True,
        fontsize=32, handletextpad=0.4, labelspacing=0.5, fancybox=True,
        framealpha=0.9, edgecolor="lightgray",
    )
    legend.get_frame().set_linewidth(0.5)
    fig.text(0.5, 0.02, "(c)", fontsize=36, fontweight="bold",
             ha="center", va="bottom")

    _save(fig, save_path, dpi, tight=False)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def make_dlpfc3_figure(
    data_dir: str = "Data",
    save_dir: str = "assets",
    device: Optional[str] = None,
    save_name: str = "dlpfc3_3d.png",
    show: bool = True,
    epochs: Optional[int] = None,
    pretrain_epochs: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Run consecutive DLPFC Sample 3 alignment and draw paper panel (c)."""
    aligned = align_dlpfc_sample3_consecutive(
        data_dir=data_dir, device=device,
        epochs=epochs, pretrain_epochs=pretrain_epochs,
    )
    return plot_dlpfc3_3d(
        aligned,
        save_path=os.path.join(save_dir, save_name),
        show=show,
    )

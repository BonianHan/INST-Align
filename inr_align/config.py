"""Configuration dataclasses for inr_align.

All hyperparameters are centralized here. To tune, modify the config
object before passing it to ``run()`` or ``benchmark_all()``.

The CLI is auto-generated from the dataclass fields so that every config
parameter can also be set from the command line.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ============================================================================
# Model (DeformationNet — spatial only)
# ============================================================================

@dataclass
class ModelConfig:
    """DeformationNet architecture — spatial deformation only.

    Architecture::

        (x, y) -> PE -> Trunk MLP -> SpatialHead -> (dx, dy)

    The SpatialHead outputs per-point displacements with identity-init.
    Jacobian regularization ensures smooth, volume-preserving deformation.
    Embedding is handled by the independent ExprINR (see JointConfig).
    """

    d: int = 2
    hidden: int = 128               # Trunk hidden width
    layers: int = 6                  # Trunk layers

    n_freqs: int = 6
    max_freq_log2: int = 5

    # --- Spatial head ---
    spatial_head_layers: int = 1     # Small: few layers, low capacity
    spatial_head_hidden: int = 64


# ============================================================================
# Matcher
# ============================================================================

@dataclass
class MatcherConfig:
    """Adaptive-temperature soft matcher."""

    tau_init: float = 0.1
    tau_min: float = 0.05
    tau_max: float = 1.0
    lambda_feat: float = 1.0
    ema_decay: float = 0.9


# ============================================================================
# Training
# ============================================================================

@dataclass
class TrainConfig:
    """Training loop hyper-parameters."""

    epochs: int = 200
    batch_size: int = 2500
    topk: int = 64
    lr: float = 1e-3
    weight_rev: float = 1.0
    full_reverse_interval: int = 1   # Full-coverage reverse loss every N epochs (0 = off)
    grad_clip: float = 1.0
    warmup_fraction: float = 0.3
    print_every: int = 10
    scheduler_patience: int = 50     # ReduceLROnPlateau: epochs without improvement before LR drop
    scheduler_factor: float = 0.5    # LR multiplied by this factor on plateau
    scheduler_min_lr: float = 1e-6   # minimum LR floor


# ============================================================================
# Joint (ExprINR + Decoder)
# ============================================================================

@dataclass
class JointConfig:
    """Hyperparameters for ExprINR / Decoder.

    Phase 1 — INR Pretrain (``inr_pretrain_epochs``):
        Two independent INRs learn spatial expression fields:
        coords1 → INR1 → emb → SharedDecoder(emb, batch=0) → expr1_hat
        coords2 → INR2 → emb → SharedDecoder(emb, batch=1) → expr2_hat

    Phase 2 — Joint Alignment (``TrainConfig.epochs``):
        INR2 frozen.  INR1 + decoder optionally continue training (recon at
        reduced weight), or fully frozen (``freeze_inr_phase2=True``).
        P-matrix uses INR1 for both sides (target coordinate space).
        DeformNet trained: matching + Jacobian + uniqueness.
    """

    # --- ExprINR (coords -> embedding) ---
    emb_dim: int = 64               # Bottleneck embedding dimension
    inr_hidden: int = 256           # INR backbone hidden width
    inr_layers: int = 4             # INR backbone layers
    inr_n_freqs: int = 6            # Fourier PE frequencies
    inr_max_freq_log2: int = 5      # Max frequency (log2) for PE

    # --- INR pretraining ---
    inr_pretrain_epochs: int = 300  # Phase 1 pretrain epochs
    inr_pretrain_lr: float = 1e-3   # Learning rate for INR pretraining
    freeze_inr_phase2: bool = True  # Freeze INR1+decoder in Phase 2 (preserve embedding quality)

    # --- Decoder (embedding + slice_id -> expression) ---
    n_output: int = 2000            # Reconstruction target dim (HVG count, set at runtime)
    decoder_hidden: int = 256
    decoder_layers: int = 2

    # --- Loss weights ---
    lam_match: float = 1.0
    lam_recon: float = 1.0          # Recon weight during Phase 1 pretrain
    lam_recon_phase2: float = 0.1   # Reduced recon weight during Phase 2 alignment
    lam_smooth: float = 0.01
    lam_jacobian: float = 1.0
    lam_uniqueness: float = 0.1
    lam_deform_mag: float = 0.1     # Deformation magnitude penalty: ||x_def - x||^2

    # --- Smoothing ---
    smooth_k: int = 6

    # --- Slices ---
    n_slices: int = 2


# ============================================================================
# Rigid alignment (ICP)
# ============================================================================

@dataclass
class ICPConfig:
    """Adaptive ICP alignment parameters.

    ``mode`` controls the rotation search strategy:

    - ``"adaptive"`` (default): PCA-guided 4-candidate search, falling back
      to full angular search if PCA is uncertain.  Uses expression-based
      re-ranking when embeddings are provided.
    - ``"icp_only"``: No rotation search at all.  Just runs ICP directly
      on the original coordinates (translation + small rotation only).
      Use this when slices are already roughly aligned.
    - ``"pca"``: PCA-guided search only (no full-search fallback).
    """

    mode: str = "icp_only"  # "adaptive" | "icp_only" | "pca"
    angle_step: int = 15
    icp_max_iter: int = 100
    pca_rmse_ratio: float = 1.1
    icp_threshold: float = 0.05


# ============================================================================
# Pipeline
# ============================================================================

@dataclass
class PipelineConfig:
    """Top-level configuration for run.py / benchmark.py."""

    dataset: str = "STARMap"
    data_dir: str = "./Data"
    output_dir: str = "./Results"
    label_key: str = "original_domain"
    spatial_key: str = "spatial"
    pca_key: str = "X_pca"
    n_top_genes: int = 2000
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    model: ModelConfig = field(default_factory=ModelConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    icp: ICPConfig = field(default_factory=ICPConfig)
    joint: JointConfig = field(default_factory=JointConfig)


# ============================================================================
# CLI — auto-generated from config dataclasses
# ============================================================================


def _add_dataclass_args(
    parser: argparse.ArgumentParser,
    dc_class: type,
    prefix: str = "",
    defaults: Optional[object] = None,
) -> None:
    """Add argparse arguments for every field in a dataclass."""
    obj = defaults if defaults is not None else dc_class()
    for f in dataclasses.fields(dc_class):
        name = f"--{prefix}{f.name}" if prefix else f"--{f.name}"
        default = getattr(obj, f.name)
        if isinstance(default, bool):
            parser.add_argument(name, type=lambda x: x.lower() in ("true", "1", "yes"), default=default, help=f"{prefix}{f.name} (default: {default})")
        elif isinstance(default, (int, float, str)):
            parser.add_argument(name, type=type(default), default=default, help=f"{prefix}{f.name} (default: {default})")


def _set_dataclass_from_args(
    obj: object,
    args: argparse.Namespace,
    prefix: str = "",
) -> None:
    """Set dataclass fields from argparse namespace."""
    for f in dataclasses.fields(obj):
        arg_name = f"{prefix}{f.name}" if prefix else f.name
        if hasattr(args, arg_name):
            setattr(obj, f.name, getattr(args, arg_name))


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add all PipelineConfig parameters to an argparse parser.

    Grouped by sub-config with prefixes to avoid name collisions::

        --dataset, --data_dir, ...        (PipelineConfig)
        --model_hidden, --model_layers    (ModelConfig)
        --matcher_tau_init, ...           (MatcherConfig)
        --train_epochs, ...               (TrainConfig)
        --icp_mode, ...                   (ICPConfig)
        --joint_decoder_hidden, ...       (JointConfig)
    """
    # Top-level PipelineConfig scalars
    g = parser.add_argument_group("Pipeline")
    g.add_argument("--dataset", type=str, default="STARMap", help="Dataset name")
    g.add_argument("--data_dir", type=str, default="./Data", help="Data directory")
    g.add_argument("--output_dir", type=str, default="./Results", help="Output directory")
    g.add_argument("--label_key", type=str, default="original_domain")
    g.add_argument("--spatial_key", type=str, default="spatial")
    g.add_argument("--pca_key", type=str, default="X_pca")
    g.add_argument("--n_top_genes", type=int, default=2000)
    g.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")

    # Sub-configs with prefixes
    g = parser.add_argument_group("Model (DeformationNet)")
    _add_dataclass_args(g, ModelConfig, prefix="model_")

    g = parser.add_argument_group("Matcher")
    _add_dataclass_args(g, MatcherConfig, prefix="matcher_")

    g = parser.add_argument_group("Training")
    _add_dataclass_args(g, TrainConfig, prefix="train_")

    g = parser.add_argument_group("ICP")
    _add_dataclass_args(g, ICPConfig, prefix="icp_")

    g = parser.add_argument_group("Joint (ExprINR+Decoder)")
    _add_dataclass_args(g, JointConfig, prefix="joint_")


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from a parsed argparse namespace."""
    config = PipelineConfig()
    # Top-level scalars
    for name in ("dataset", "data_dir", "output_dir", "label_key",
                 "spatial_key", "pca_key", "n_top_genes", "device"):
        if hasattr(args, name):
            setattr(config, name, getattr(args, name))

    # Sub-configs
    _set_dataclass_from_args(config.model, args, prefix="model_")
    _set_dataclass_from_args(config.matcher, args, prefix="matcher_")
    _set_dataclass_from_args(config.train, args, prefix="train_")
    _set_dataclass_from_args(config.icp, args, prefix="icp_")
    _set_dataclass_from_args(config.joint, args, prefix="joint_")
    return config


def print_config(config: PipelineConfig) -> None:
    """Print all non-default configuration values."""
    default = PipelineConfig()
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)

    def _print_section(name: str, obj, default_obj):
        lines = []
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            dval = getattr(default_obj, f.name)
            marker = " *" if val != dval else ""
            lines.append(f"  {f.name:24s} = {val}{marker}")
        if lines:
            print(f"\n  [{name}]")
            for l in lines:
                print(l)

    # Top-level
    print(f"\n  [Pipeline]")
    for name in ("dataset", "data_dir", "output_dir", "label_key",
                 "spatial_key", "pca_key", "n_top_genes", "device"):
        val = getattr(config, name)
        dval = getattr(default, name)
        marker = " *" if val != dval else ""
        print(f"  {name:24s} = {val}{marker}")

    _print_section("Model", config.model, default.model)
    _print_section("Matcher", config.matcher, default.matcher)
    _print_section("Training", config.train, default.train)
    _print_section("ICP", config.icp, default.icp)
    _print_section("Joint", config.joint, default.joint)
    print("\n" + "=" * 60)
    print("  (* = non-default)")
    print("=" * 60)


# ============================================================================
# Known slice orderings for benchmark datasets
# ============================================================================

SLICE_ORDER: Dict[str, List[str]] = {
    "BaristaSeq": ["slices1", "slices2", "slices3"],
    "STARMap": ["slices1", "slices3", "slices2"],
    "DLPFC_sample1": ["slices2", "slices3", "slices1", "slices4"],
    "DLPFC_sample2": ["slices2", "slices3", "slices1", "slices4"],
    "DLPFC_sample3": ["slices2", "slices4", "slices1", "slices3"],
    "MERFISH": ["slices1", "slices4", "slices2", "slices5", "slices3"],
    "MERFISH_Brain_S2": ["slices2", "slices1"],
    "MERFISH_Brain_S3": ["slices2", "slices1"],
    "MERFISH_Brain_S4": ["slices2", "slices1"],
    "MERFISH_Brain_S5": ["slices2", "slices1"],
    "MERFISH_Brain_S6": ["slices2", "slices1"],
    "MERFISH_Brain_S7": ["slices2", "slices1"],
    "MERFISH_Brain_S8": ["slices2", "slices1"],
    "MERFISH_Brain_S9": ["slices2", "slices1"],
    "MERFISH_Brain_S10": ["slices2", "slices1"],
    "MERFISH_Brain_S11": ["slices2", "slices1"],
    "MERFISH_Brain_S12": ["slices2", "slices1"],
    "Mouse": ["slices1", "slices2"],
    "TNBC": ["slices1", "slices2"],
}

# DLPFC sample groups for benchmark (original_data format, per-pair)
DLPFC_SAMPLE_GROUPS: List[List[str]] = [
    ["151507", "151508", "151509", "151510"],
    ["151669", "151670", "151671", "151672"],
    ["151673", "151674", "151675", "151676"],
]

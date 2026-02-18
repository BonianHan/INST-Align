"""Decoupled joint optimization — model components.

Architecture::

    Source coords → WarpNet → aligned_coords ──┐
                                                ├→ P matrix → matching_loss
    Source expr  → Encoder → embedding ────────┘
                      │
                      ├→ Decoder(emb, slice_id) → recon_loss
                      └→ GRL → Discriminator(emb) → adv_loss

Components:
  - ``ExprEncoder``:  MLP, expression → point-discriminative embedding.
  - ``ExprDecoder``:  MLP, embedding + slice_onehot → reconstructed expression.
  - ``SliceDiscriminator``: MLP with GRL, embedding → slice prediction.
  - ``GradientReversalLayer``: reverses gradients for adversarial training.

Backward-compatible: existing ``DeformationNet`` + ``UnifiedCostMatcher``
from ``model.py`` are unchanged.  New encoder embeddings drop into
``compute_P_matrix`` via the ``emb1`` / ``emb2`` arguments.

Usage::

    from model1 import ExprEncoder, ExprDecoder, SliceDiscriminator, JointConfig
    from model1 import build_joint_models
    from inr_align.model import DeformationNet, UnifiedCostMatcher
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from scipy.spatial import cKDTree


# ============================================================================
# Config
# ============================================================================


@dataclass
class JointConfig:
    """Hyperparameters for encoder / decoder / discriminator."""

    # --- Encoder ---
    n_input: int = 50               # PCA dimension
    emb_dim: int = 64               # Encoder output dimension
    encoder_hidden: int = 256
    encoder_layers: int = 2

    # --- Decoder ---
    n_output: int = 50              # Reconstruction target dim (= n_input for PCA)
    decoder_hidden: int = 256
    decoder_layers: int = 2

    # --- Discriminator ---
    disc_hidden: int = 128
    disc_layers: int = 2

    # --- Loss weights ---
    lam_match: float = 1.0
    lam_recon: float = 1.0
    lam_adv: float = 0.1
    lam_smooth: float = 0.01
    lam_jacobian: float = 0.015
    lam_uniqueness: float = 0.1

    # --- Adversarial schedule ---
    grl_max_lambda: float = 1.0     # Ramp target

    # --- Smoothing ---
    smooth_k: int = 6

    # --- Slices ---
    n_slices: int = 2


# ============================================================================
# Gradient Reversal Layer (Ganin et al., 2016)
# ============================================================================


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


class GradientReversalLayer(nn.Module):
    """Reverse gradients during backward pass.

    Between encoder and discriminator: minimising disc loss w.r.t.
    encoder parameters actually *maximises* it → encoder fools disc.
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lam)

    def set_lambda(self, lam: float) -> None:
        self.lam = lam


# ============================================================================
# Expression Encoder
# ============================================================================


class ExprEncoder(nn.Module):
    """MLP: per-point expression → embedding.

    No GCN (no neighbourhood averaging) → preserves point-level
    discriminability needed for P-matrix matching.

    Uses LayerNorm (not BatchNorm) for mixed-slice batch stability.
    """

    def __init__(
        self,
        n_input: int = 50,
        emb_dim: int = 64,
        hidden: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_d = n_input
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            in_d = hidden
        layers.append(nn.Linear(in_d, emb_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, n_input) → (N, emb_dim)``  (raw, NOT L2-normalised)."""
        return self.net(x)


# ============================================================================
# Expression Decoder
# ============================================================================


class ExprDecoder(nn.Module):
    """MLP: embedding + slice_onehot → reconstructed expression.

    Slice-onehot lets the decoder absorb batch effects so the
    encoder embedding stays batch-free.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        n_slices: int = 2,
        n_output: int = 50,
        hidden: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_slices = n_slices
        layers = []
        in_d = emb_dim + n_slices
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden), nn.ReLU()]
            in_d = hidden
        layers.append(nn.Linear(in_d, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor, slice_id: torch.Tensor) -> torch.Tensor:
        """``(N, emb_dim), (N,) → (N, n_output)``."""
        onehot = F.one_hot(slice_id, self.n_slices).float()
        return self.net(torch.cat([emb, onehot], dim=-1))


# ============================================================================
# Slice Discriminator
# ============================================================================


class SliceDiscriminator(nn.Module):
    """MLP with built-in GRL: embedding → slice logits.

    Two forward modes:

    - ``forward(emb)``: with GRL (use in joint encoder+disc optimiser step).
    - ``forward_no_grl(emb)``: without GRL (use if you do separate disc step).
    """

    def __init__(
        self,
        emb_dim: int = 64,
        n_slices: int = 2,
        hidden: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(lam=1.0)
        layers = []
        in_d = emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden), nn.ReLU()]
            in_d = hidden
        layers.append(nn.Linear(in_d, n_slices))
        self.classifier = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """With gradient reversal → encoder receives reversed grads."""
        return self.classifier(self.grl(emb))

    def forward_no_grl(self, emb: torch.Tensor) -> torch.Tensor:
        """Without GRL → normal disc training."""
        return self.classifier(emb)

    def set_grl_lambda(self, lam: float) -> None:
        self.grl.set_lambda(lam)


# ============================================================================
# k-NN graph for spatial smoothing
# ============================================================================


def build_knn_graph(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Pre-compute k-NN indices.  ``(N, 2) → (N, k)``."""
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    _, idx = tree.query(coords, k=k + 1)
    return idx[:, 1:]


# ============================================================================
# Factory
# ============================================================================


def build_joint_models(
    config: JointConfig,
    device: str = "cuda",
) -> Dict[str, nn.Module]:
    """Build encoder + decoder + discriminator from config.

    Returns dict with keys ``"encoder"``, ``"decoder"``, ``"discriminator"``.
    """
    enc = ExprEncoder(
        config.n_input, config.emb_dim,
        config.encoder_hidden, config.encoder_layers,
    ).to(device)

    dec = ExprDecoder(
        config.emb_dim, config.n_slices, config.n_output,
        config.decoder_hidden, config.decoder_layers,
    ).to(device)

    disc = SliceDiscriminator(
        config.emb_dim, config.n_slices,
        config.disc_hidden, config.disc_layers,
    ).to(device)

    return {"encoder": enc, "decoder": dec, "discriminator": disc}

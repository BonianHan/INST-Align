"""Neural network components and rigid alignment (PCA + adaptive ICP).

This module contains:
- ``NerfiesPositionalEncoding``: windowed Fourier features (Nerfies).
- ``DeformationNet``: residual MLP that predicts per-point displacements.
- ``UnifiedCostMatcher``: adaptive-temperature soft-assignment matcher.
- ``ExpressionINR``: implicit neural field for gene expression prediction (legacy).
- ``ExprField``: canonical expression field with batch correction (new).
- ``adaptive_icp``: PCA-guided + full-search ICP for arbitrary rotations.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from inr_align.config import ExprFieldConfig, ExpressionINRConfig, ICPConfig, ModelConfig, MatcherConfig

# ============================================================================
# Positional Encoding
# ============================================================================


class NerfiesPositionalEncoding(nn.Module):
    """Windowed Fourier positional encoding (Park et al., *Nerfies*, 2021).

    The coarse-to-fine *alpha* window lets the network learn from low
    to high frequencies progressively during training.
    """

    def __init__(self, d_in: int = 2, n_freqs: int = 6, max_freq_log2: int = 5):
        super().__init__()
        self.d_in = d_in
        self.n_freqs = n_freqs
        freqs = 2.0 ** torch.linspace(0, max_freq_log2, n_freqs)
        self.register_buffer("freqs", freqs)
        self.d_out = d_in + 2 * n_freqs * d_in

    def forward(self, x: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode *x* with Fourier features.

        Args:
            x: ``(N, d_in)`` input coordinates.
            alpha: Scalar controlling the coarse-to-fine window.  When
                ``None`` all frequencies are fully enabled.
        """
        angles = x.unsqueeze(-1) * self.freqs.view(1, 1, -1) * np.pi
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)

        if alpha is not None:
            j = torch.arange(self.n_freqs, device=x.device, dtype=x.dtype)
            window = (1 - torch.cos(np.pi * torch.clamp(alpha - j, 0, 1))) / 2
            sin_enc = sin_enc * window.view(1, 1, -1)
            cos_enc = cos_enc * window.view(1, 1, -1)

        return torch.cat(
            [x, sin_enc.reshape(x.shape[0], -1), cos_enc.reshape(x.shape[0], -1)],
            dim=-1,
        )


# ============================================================================
# Deformation Network
# ============================================================================


class DeformationNet(nn.Module):
    """Residual MLP that predicts per-point spatial displacements.

    The output is ``x + delta(x)`` so that the identity mapping is the
    default at initialization (last-layer weights near zero).

    Args:
        config: Architecture hyper-parameters.
        grid_mode: Whether the data lies on a regular grid.
        grid_spacing: ``(spacing_x, spacing_y)`` for grid snapping.
        grid_origin: ``(x0, y0)`` origin of the grid.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        grid_mode: bool = False,
        grid_spacing: Optional[list] = None,
        grid_origin: Optional[NDArray] = None,
    ):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.encoder = NerfiesPositionalEncoding(config.d, config.n_freqs, config.max_freq_log2)
        self.n_freqs = config.n_freqs
        self.grid_mode = grid_mode

        if grid_spacing is not None:
            self.register_buffer("grid_spacing", torch.tensor(grid_spacing, dtype=torch.float32))
        else:
            self.grid_spacing = None

        if grid_origin is not None:
            self.register_buffer("grid_origin", torch.tensor(grid_origin, dtype=torch.float32))
        else:
            self.grid_origin = None

        # Build MLP
        net = []
        in_dim = self.encoder.d_out
        for i in range(config.layers - 1):
            net.append(nn.Linear(in_dim if i == 0 else config.hidden, config.hidden))
            net.append(nn.ReLU())
        net.append(nn.Linear(config.hidden, config.d))
        self.net = nn.Sequential(*net)

        # Near-identity initialization
        with torch.no_grad():
            self.net[-1].weight.uniform_(-1e-4, 1e-4)
            self.net[-1].bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        snap_to_grid: bool = False,
    ) -> torch.Tensor:
        """Return deformed coordinates ``x + delta(x)``."""
        x_def = x + self.net(self.encoder(x, alpha))
        if snap_to_grid and self.grid_mode and self.grid_spacing is not None:
            x_def = self._snap_to_grid(x_def)
        return x_def

    # --------------------------------------------------------------------- #

    def _snap_to_grid(self, coords: torch.Tensor) -> torch.Tensor:
        relative = coords - self.grid_origin if self.grid_origin is not None else coords
        snapped = torch.round(relative / self.grid_spacing) * self.grid_spacing
        if self.grid_origin is not None:
            snapped = snapped + self.grid_origin
        return snapped


# ============================================================================
# Unified Cost Matcher
# ============================================================================


class UnifiedCostMatcher:
    """Adaptive-temperature matcher that balances spatial and feature costs.

    This is *not* an ``nn.Module``; it holds no learnable parameters.
    Its state (``tau``, ``spatial_scale``, ``feat_scale``) is updated via
    EMA during training.
    """

    def __init__(self, config: Optional[MatcherConfig] = None):
        if config is None:
            config = MatcherConfig()
        self.tau = config.tau_init
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        self.lambda_feat = config.lambda_feat
        self.ema_decay = config.ema_decay
        self.sinkhorn_iters = config.sinkhorn_iters
        self.spatial_scale: float = 1.0
        self.feat_scale: float = 1.0

    # ---- EM-style tau update --------------------------------------------- #

    def update_tau_em(self, P: torch.Tensor, C: torch.Tensor) -> None:
        with torch.no_grad():
            weighted_cost = (P * C).sum() / (P.sum() + 1e-8)
            tau_new = weighted_cost.item() / 2.0
            self.tau = self.ema_decay * self.tau + (1 - self.ema_decay) * tau_new
            self.tau = max(self.tau_min, min(self.tau_max, self.tau))

    def update_scales(self, spatial_dist_sq: torch.Tensor, feat_dist: torch.Tensor) -> None:
        with torch.no_grad():
            self.spatial_scale = 0.9 * self.spatial_scale + 0.1 * (spatial_dist_sq.mean().item() + 1e-8)
            self.feat_scale = 0.9 * self.feat_scale + 0.1 * (feat_dist.mean().item() + 1e-8)

    def reset(self) -> None:
        """Reset to initial state."""
        self.spatial_scale = 1.0
        self.feat_scale = 1.0


# ============================================================================
# Expression INR — coordinate-conditioned gene expression prediction
# ============================================================================


class ExpressionINR(nn.Module):
    """Implicit neural representation for spatial gene expression fields.

    Maps 2D spatial coordinates to predicted HVG gene expression values.
    Uses the same ``NerfiesPositionalEncoding`` as ``DeformationNet``.

    Architecture::

        coords (N, 2)  →  PE (N, d_pe)  →  MLP  →  ReLU  →  expression (N, n_genes)

    Args:
        config: Architecture and training hyper-parameters.
        n_genes: Number of output genes (determined at runtime).
    """

    def __init__(self, config: Optional[ExpressionINRConfig] = None, n_genes: int = 200):
        super().__init__()
        if config is None:
            config = ExpressionINRConfig()

        self.encoder = NerfiesPositionalEncoding(2, config.n_freqs, config.max_freq_log2)
        self.n_freqs = config.n_freqs
        self.n_genes = n_genes

        # Build MLP
        net = []
        in_dim = self.encoder.d_out
        for i in range(config.layers - 1):
            net.append(nn.Linear(in_dim if i == 0 else config.hidden, config.hidden))
            net.append(nn.ReLU())
        net.append(nn.Linear(config.hidden, n_genes))
        # No final activation — we use per-gene z-score normalization,
        # so output values can be negative.
        self.net = nn.Sequential(*net)

    def forward(
        self,
        x: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict gene expression at spatial coordinates *x*.

        Args:
            x: ``(N, 2)`` spatial coordinates.
            alpha: Coarse-to-fine window scalar (optional).

        Returns:
            ``(N, n_genes)`` predicted expression values.
        """
        return self.net(self.encoder(x, alpha))


# ============================================================================
# ExprField — canonical expression field with batch correction
# ============================================================================


class ExprField(nn.Module):
    """Canonical expression field with per-slice batch embeddings.

    Learns a shared spatial expression field across all slices.
    Each slice has a learnable batch embedding that absorbs slice-specific
    technical variation.  Setting ``batch_emb=0`` yields batch-corrected
    canonical predictions.

    Architecture::

        coords (N, 2)  →  PE (N, d_pe)
                              ↓
                        concat with batch_emb (N, batch_emb_dim)
                              ↓
                        backbone MLP (hidden layers)
                              ↓
                        bottleneck (N, latent_dim)   ← clustering embedding
                              ↓
                        head linear → expression (N, n_genes)

    Args:
        config: Architecture hyper-parameters.
        n_genes: Number of output genes (determined at runtime).
        n_slices: Number of slices (for batch embedding table).
    """

    def __init__(
        self,
        config: Optional[ExprFieldConfig] = None,
        n_genes: int = 200,
        n_slices: int = 2,
    ):
        super().__init__()
        if config is None:
            config = ExprFieldConfig()

        self.encoder = NerfiesPositionalEncoding(2, config.n_freqs, config.max_freq_log2)
        self.n_freqs = config.n_freqs
        self.n_genes = n_genes
        self.batch_emb_dim = config.batch_emb_dim
        self.latent_dim = config.latent_dim

        # Per-slice batch embedding
        self.batch_emb = nn.Embedding(n_slices, config.batch_emb_dim)
        nn.init.zeros_(self.batch_emb.weight)  # start at zero → canonical by default

        # Backbone MLP: PE + batch_emb → hidden
        backbone_layers = []
        in_dim = self.encoder.d_out + config.batch_emb_dim
        for i in range(config.layers - 1):
            backbone_layers.append(nn.Linear(in_dim if i == 0 else config.hidden, config.hidden))
            backbone_layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone_layers)

        # Bottleneck: hidden → latent_dim
        self.bottleneck = nn.Linear(config.hidden, config.latent_dim)

        # Head: latent_dim → n_genes
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.latent_dim, n_genes),
        )

    def forward(
        self,
        coords: torch.Tensor,
        slice_id: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict gene expression.

        Args:
            coords: ``(N, 2)`` spatial coordinates.
            slice_id: ``(N,)`` integer slice indices, or ``None`` for canonical
                (batch_emb = zero vector).
            alpha: Coarse-to-fine window scalar (optional).

        Returns:
            ``(N, n_genes)`` predicted expression values.
        """
        pe = self.encoder(coords, alpha)
        if slice_id is not None:
            b = self.batch_emb(slice_id)
        else:
            b = torch.zeros(coords.shape[0], self.batch_emb_dim, device=coords.device)
        h = self.backbone(torch.cat([pe, b], dim=-1))
        z = self.bottleneck(h)
        return self.head(z)

    def canonical(self, coords: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batch-free canonical prediction (batch_emb = zero).

        Args:
            coords: ``(N, 2)`` spatial coordinates.
            alpha: Coarse-to-fine window scalar (optional).

        Returns:
            ``(N, n_genes)`` canonical (batch-corrected) expression.
        """
        return self.forward(coords, slice_id=None, alpha=alpha)

    def get_embedding(self, coords: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract bottleneck embedding for clustering.

        Uses canonical mode (batch_emb = zero) to produce batch-free
        representations.

        Args:
            coords: ``(N, 2)`` spatial coordinates.
            alpha: Coarse-to-fine window scalar (optional).

        Returns:
            ``(N, latent_dim)`` canonical embedding.
        """
        pe = self.encoder(coords, alpha)
        b = torch.zeros(coords.shape[0], self.batch_emb_dim, device=coords.device)
        h = self.backbone(torch.cat([pe, b], dim=-1))
        return self.bottleneck(h)


# ---- Expression normalization ------------------------------------------------


def normalize_expression(
    expr: torch.Tensor,
    method: str = "per_gene",
    stats: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Normalize expression matrix for stable training.

    Args:
        expr: ``(N, G)`` expression matrix.
        method: ``"per_gene"`` z-score normalization (recommended).
        stats: Pre-computed ``{"mean": ..., "std": ...}`` from the target
            slice.  If ``None``, statistics are computed from *expr*.

    Returns:
        ``(normalized_expr, stats_dict)``.  Pass ``stats_dict`` when
        normalizing the source slice to ensure consistent scaling.
    """
    if method == "per_gene":
        if stats is None:
            mean = expr.mean(dim=0)
            std = expr.std(dim=0) + 1e-8
            stats = {"mean": mean, "std": std}
        return (expr - stats["mean"]) / stats["std"], stats
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ---- Expression INR pre-training ---------------------------------------------


def pretrain_expression_inr(
    expr_inr: ExpressionINR,
    coords: torch.Tensor,
    expression: torch.Tensor,
    config: Optional[ExpressionINRConfig] = None,
) -> ExpressionINR:
    """Pre-train the ExpressionINR on target-slice expression.

    Learns the mapping ``coords → expression`` using MSE loss with
    Nerfies-style coarse-to-fine alpha scheduling.

    Args:
        expr_inr: ``ExpressionINR`` model (on device).
        coords: ``(N, 2)`` target coordinates (normalized, on device).
        expression: ``(N, G)`` target HVG expression (normalized, on device).
        config: Expression INR hyper-parameters.

    Returns:
        Trained ``ExpressionINR`` with best checkpoint restored.
    """
    if config is None:
        config = ExpressionINRConfig()

    optimizer = torch.optim.Adam(expr_inr.parameters(), lr=config.pretrain_lr)
    n_freqs = expr_inr.n_freqs
    best_loss = float("inf")
    best_state = None
    start = time.time()

    for ep in range(config.pretrain_epochs):
        warmup = int(config.pretrain_epochs * config.pretrain_warmup)
        alpha = n_freqs * (ep / warmup) if ep < warmup else float(n_freqs)
        alpha_t = torch.tensor(alpha, device=coords.device)

        optimizer.zero_grad(set_to_none=True)
        pred = expr_inr(coords, alpha_t)
        loss = F.mse_loss(pred, expression)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in expr_inr.state_dict().items()}

        if ep % config.pretrain_print_every == 0 or ep == config.pretrain_epochs - 1:
            # R-squared
            with torch.no_grad():
                ss_res = (expression - pred).pow(2).sum()
                ss_tot = (expression - expression.mean(dim=0)).pow(2).sum()
                r2 = 1 - ss_res / (ss_tot + 1e-8)
            print(
                f"    ExprINR pretrain ep={ep:03d} | "
                f"MSE={loss.item():.6f} | R²={r2.item():.4f}"
            )

    # Restore best
    if best_state:
        expr_inr.load_state_dict({k: v.to(coords.device) for k, v in best_state.items()})

    elapsed = time.time() - start
    print(f"    ExprINR pretrain done. Best MSE={best_loss:.6f} ({elapsed:.1f}s)")
    return expr_inr


# ---- ExprField joint pre-training --------------------------------------------


def pretrain_expr_field(
    expr_field: ExprField,
    coords_list: list,
    expr_list: list,
    config: Optional[ExprFieldConfig] = None,
) -> ExprField:
    """Pre-train ExprField jointly on all slices.

    Each slice gets its own ``batch_emb`` to absorb technical variation,
    while the shared backbone learns the canonical expression field.

    Args:
        expr_field: ``ExprField`` model (on device).
        coords_list: List of ``(N_s, 2)`` coordinate tensors per slice (on device).
        expr_list: List of ``(N_s, G)`` expression tensors per slice (on device).
        config: ExprField hyper-parameters.

    Returns:
        Trained ``ExprField`` with best checkpoint restored.
    """
    if config is None:
        config = ExprFieldConfig()

    device = coords_list[0].device
    n_slices = len(coords_list)
    optimizer = torch.optim.Adam(expr_field.parameters(), lr=config.pretrain_lr)
    n_freqs = expr_field.n_freqs
    best_loss = float("inf")
    best_state = None
    start = time.time()

    # Build per-slice slice_id tensors
    slice_ids = [
        torch.full((c.shape[0],), s, dtype=torch.long, device=device)
        for s, c in enumerate(coords_list)
    ]

    for ep in range(config.pretrain_epochs):
        warmup = int(config.pretrain_epochs * config.pretrain_warmup)
        alpha = n_freqs * (ep / warmup) if ep < warmup else float(n_freqs)
        alpha_t = torch.tensor(alpha, device=device)

        optimizer.zero_grad(set_to_none=True)

        total_recon = 0.0
        for s in range(n_slices):
            pred = expr_field(coords_list[s], slice_ids[s], alpha_t)
            total_recon = total_recon + F.mse_loss(pred, expr_list[s])

        # L2 regularization on batch embeddings
        batch_reg = config.pretrain_batch_reg * expr_field.batch_emb.weight.pow(2).mean()
        loss = total_recon / n_slices + batch_reg

        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.cpu().clone() for k, v in expr_field.state_dict().items()}

        if ep % config.pretrain_print_every == 0 or ep == config.pretrain_epochs - 1:
            with torch.no_grad():
                batch_norm = expr_field.batch_emb.weight.norm(dim=1).mean().item()
            print(
                f"    ExprField pretrain ep={ep:03d} | "
                f"MSE={total_recon.item() / n_slices:.6f} | "
                f"batch_reg={batch_reg.item():.6f} | "
                f"batch_norm={batch_norm:.4f}"
            )

    # Restore best
    if best_state:
        expr_field.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    elapsed = time.time() - start
    print(f"    ExprField pretrain done. Best loss={best_loss:.6f} ({elapsed:.1f}s)")
    return expr_field


# ============================================================================
# Rigid Alignment — PCA helpers
# ============================================================================


def pca_axes(X: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """PCA decomposition returning ``(mean, eigenvectors, eigenvalues)``."""
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (len(Xc) - 1)
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    U = evecs[:, idx]
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
    return mu, U, evals


def nn_rmse(A: NDArray, B: NDArray) -> float:
    """Nearest-neighbour RMSE from *B* to *A*."""
    tree = cKDTree(A)
    d, _ = tree.query(B, k=1)
    return float(np.sqrt(np.mean(d ** 2)))


# ============================================================================
# Rigid Alignment — ICP
# ============================================================================


def pca_coarse_align(A: NDArray, B: NDArray) -> Tuple[NDArray, NDArray, float]:
    """PCA axis alignment with 4-candidate disambiguation.

    Returns:
        ``(R, t, rmse)`` of the best candidate.
    """
    muA, UA, _ = pca_axes(A)
    muB, UB, _ = pca_axes(B)

    Bc = B - muB
    candidates = []

    for swap in [False, True]:
        UBs = UB[:, [1, 0]].copy() if swap else UB.copy()
        for flip in [+1, -1]:
            Utmp = UBs.copy()
            Utmp[:, 1] *= flip
            if np.linalg.det(Utmp) < 0:
                Utmp[:, 0] *= -1
            R = UA @ Utmp.T
            B_try = (Bc @ R.T) + muA
            err = nn_rmse(A, B_try)
            candidates.append((err, R, muA - muB @ R.T))

    candidates.sort(key=lambda x: x[0])
    best_err, R_best, t_best = candidates[0]
    return R_best, t_best, best_err


def icp_refine(
    A: NDArray,
    B_init: NDArray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[NDArray, NDArray]:
    """ICP refinement of *B_init* onto *A*.

    Returns:
        ``(R_total, t_total)`` accumulated rotation and translation.
    """
    B = B_init.copy()
    tree = cKDTree(A)
    R_total = np.eye(2)
    t_total = np.zeros(2)

    for _ in range(max_iter):
        _, idx = tree.query(B, k=1)
        A_match = A[idx]
        muA, muB = A_match.mean(0), B.mean(0)
        H = (B - muB).T @ (A_match - muA)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = muA - muB @ R.T
        B_new = B @ R.T + t
        if np.linalg.norm(B_new - B) / (np.linalg.norm(B) + 1e-8) < tol:
            break
        B = B_new
        R_total = R @ R_total
        t_total = R @ t_total + t

    return R_total, t_total


# ============================================================================
# Expression-based rotation scoring
# ============================================================================


def _expression_score(
    A: NDArray,
    B_aligned: NDArray,
    emb_A: NDArray,
    emb_B: NDArray,
    k: int = 10,
) -> float:
    """Score a rotation candidate by expression similarity.

    For each cell in *B_aligned*, find its *k* nearest neighbours in *A*,
    then compute the mean cosine similarity between their PCA embeddings.

    Returns:
        Mean cosine similarity in ``[0, 1]``.  Higher is better.
    """
    tree = cKDTree(A)
    _, nn_idx = tree.query(B_aligned, k=k)

    # L2-normalize embeddings
    emb_A_norm = emb_A / (np.linalg.norm(emb_A, axis=1, keepdims=True) + 1e-8)
    emb_B_norm = emb_B / (np.linalg.norm(emb_B, axis=1, keepdims=True) + 1e-8)

    # For each cell in B, mean cosine sim to its k nearest A cells
    n = len(B_aligned)
    sims = np.zeros(n)
    for i in range(n):
        nbr_embs = emb_A_norm[nn_idx[i]]  # (k, D)
        sims[i] = np.mean(emb_B_norm[i] @ nbr_embs.T)

    return float(np.mean(sims))


def _select_best_with_expression(
    candidates: list,
    B: NDArray,
    emb_A: NDArray,
    emb_B: NDArray,
    A: NDArray,
    verbose: bool = False,
    tag: str = "",
) -> tuple:
    """Re-rank ICP candidates using expression similarity.

    Scores each candidate by ``RMSE / (expr_sim + eps)`` — lower is better.
    This prefers candidates that are both spatially accurate and biologically
    consistent.

    Args:
        candidates: List of ``(rmse, R, t, angle)`` tuples, sorted by RMSE.
        B: ``(N2, 2)`` source coordinates.
        emb_A: ``(N1, D)`` target PCA embeddings.
        emb_B: ``(N2, D)`` source PCA embeddings.
        A: ``(N1, 2)`` target coordinates.
        verbose: Print expression scores.
        tag: Label for verbose output.

    Returns:
        Best ``(rmse, R, t, angle)`` tuple.
    """
    scored = []
    for err, R, t, angle in candidates:
        B_aligned = (B @ R.T) + t
        expr_sim = _expression_score(A, B_aligned, emb_A, emb_B, k=10)
        # Combined score: rank by expression similarity (higher = better).
        # Among candidates with similar expr_sim, prefer lower RMSE.
        # Use negative expr_sim as primary key so higher is better when sorted ascending.
        combined = (-expr_sim, err)
        scored.append((combined, expr_sim, err, R, t, angle))

    scored.sort(key=lambda x: x[0])

    if verbose:
        print(f"  Expression re-ranking ({tag}, top {len(scored)}):")
        for rank, (comb, esim, err, _, _, ang) in enumerate(scored):
            marker = " <-- best" if rank == 0 else ""
            print(f"    #{rank + 1}: angle={ang:.1f}\u00b0, RMSE={err:.4f}, expr_sim={esim:.4f}{marker}")

    _, _, best_err, best_R, best_t, best_angle = scored[0]
    return (best_err, best_R, best_t, best_angle)


# ============================================================================
# Adaptive ICP — PCA-guided + full-search fallback
# ============================================================================


def adaptive_icp(
    A: NDArray,
    B: NDArray,
    config: Optional[ICPConfig] = None,
    verbose: bool = False,
    emb_A: Optional[NDArray] = None,
    emb_B: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray, float, float]:
    """Adaptive ICP: PCA-guided when confident, full angular search otherwise.

    Strategy:
      1. Generate 4 PCA candidates and refine each with ICP.
      2. If the top-2 RMSE ratio is below ``pca_rmse_ratio``, PCA is
         uncertain -> fall back to brute-force angular search.
      3. Otherwise trust the PCA result.
      4. When PCA embeddings are provided (``emb_A``, ``emb_B``),
         re-rank top candidates by expression similarity to avoid
         biologically incorrect rotations (e.g. 180-degree flips).

    Args:
        A: ``(N1, 2)`` target coordinates (normalized).
        B: ``(N2, 2)`` source coordinates (normalized).
        config: ICP hyper-parameters.
        verbose: Print diagnostics.
        emb_A: ``(N1, D)`` PCA embeddings for *A* (optional).
        emb_B: ``(N2, D)`` PCA embeddings for *B* (optional).

    Returns:
        ``(R, t, angle_deg, rmse)`` — rotation matrix, translation,
        rotation angle in degrees, and final RMSE.
    """
    if config is None:
        config = ICPConfig()

    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Bc = B - muB

    # ------------------------------------------------------------------ #
    # icp_only mode: no rotation search, just ICP from identity
    # ------------------------------------------------------------------ #
    if config.mode == "icp_only":
        # Translate centroids, then ICP refine
        B_centered = B - muB + muA
        R_icp, t_icp = icp_refine(A, B_centered, max_iter=config.icp_max_iter)
        # Compose: B_final = B_centered @ R_icp.T + t_icp
        #        = (B - muB + muA) @ R_icp.T + t_icp
        #        = B @ R_icp.T + (-muB + muA) @ R_icp.T + t_icp
        R_total = R_icp
        t_total = (-muB + muA) @ R_icp.T + t_icp

        B_final = (B @ R_total.T) + t_total
        rmse = nn_rmse(A, B_final)
        angle_deg = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))
        if verbose:
            print(f"ICP mode: icp_only (no rotation search)")
            print(f"  angle={angle_deg:.1f}\u00b0, RMSE={rmse:.4f}")
        return R_total, t_total, angle_deg, rmse

    # ------------------------------------------------------------------ #
    # Phase 1: PCA-guided candidates
    # ------------------------------------------------------------------ #
    muA_pca, UA, _ = pca_axes(A)
    muB_pca, UB, _ = pca_axes(B)

    pca_candidates = []
    for swap in [False, True]:
        UBs = UB[:, [1, 0]].copy() if swap else UB.copy()
        for flip in [+1, -1]:
            Utmp = UBs.copy()
            Utmp[:, 1] *= flip
            if np.linalg.det(Utmp) < 0:
                Utmp[:, 0] *= -1
            R_pca = UA @ Utmp.T
            t_pca = muA_pca - muB_pca @ R_pca.T
            B_pca = (B @ R_pca.T) + t_pca

            R_icp, t_icp = icp_refine(A, B_pca, max_iter=config.icp_max_iter)
            R_total = R_icp @ R_pca
            t_total = t_pca @ R_icp.T + t_icp

            B_final = (B @ R_total.T) + t_total
            err = nn_rmse(A, B_final)
            angle = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))
            pca_candidates.append((err, R_total, t_total, angle))

    pca_candidates.sort(key=lambda x: x[0])

    pca_confident = (
        len(pca_candidates) >= 2
        and pca_candidates[1][0] / (pca_candidates[0][0] + 1e-8) > config.pca_rmse_ratio
    )

    # In "pca" mode, always treat PCA as confident (never fall back to full search)
    if config.mode == "pca":
        pca_confident = True

    use_expr = emb_A is not None and emb_B is not None

    if verbose:
        if config.mode == "pca":
            mode_str = "PCA-only (forced)"
        elif pca_confident:
            mode_str = "PCA-guided"
        else:
            mode_str = "Full-search (PCA uncertain)"
        print(f"ICP mode: {mode_str}")
        print(f"  PCA top-1: angle={pca_candidates[0][3]:.1f}\u00b0, RMSE={pca_candidates[0][0]:.4f}")
        print(f"  PCA top-2: angle={pca_candidates[1][3]:.1f}\u00b0, RMSE={pca_candidates[1][0]:.4f}")
        ratio = pca_candidates[1][0] / (pca_candidates[0][0] + 1e-8)
        print(f"  RMSE ratio: {ratio:.3f} (threshold={config.pca_rmse_ratio})")

    if pca_confident:
        if use_expr:
            # Re-rank PCA candidates with expression similarity
            best = _select_best_with_expression(
                pca_candidates, B, emb_A, emb_B, A, verbose=verbose, tag="PCA",
            )
        else:
            best = pca_candidates[0]
        best_err, R_best, t_best, best_angle = best
        if verbose:
            print(f"  -> Using PCA result: angle={best_angle:.1f}\u00b0, RMSE={best_err:.4f}")
        return R_best, t_best, best_angle, best_err

    # ------------------------------------------------------------------ #
    # Phase 2: Full angular search
    # ------------------------------------------------------------------ #
    angles = np.arange(0, 360, config.angle_step)
    all_candidates = []

    for ang in angles:
        rad = np.radians(ang)
        R_init = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        B_rot = (Bc @ R_init.T) + muA

        R_icp, t_icp = icp_refine(A, B_rot, max_iter=config.icp_max_iter)

        R_total = R_icp @ R_init
        t_total = (muA - muB @ R_init.T) @ R_icp.T + t_icp

        B_final = (B @ R_total.T) + t_total
        err = nn_rmse(A, B_final)
        final_angle = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))
        all_candidates.append((err, R_total, t_total, final_angle))

    all_candidates.sort(key=lambda x: x[0])

    if use_expr:
        # Re-rank top candidates with expression similarity
        best = _select_best_with_expression(
            all_candidates[:6], B, emb_A, emb_B, A, verbose=verbose, tag="Full-search",
        )
    else:
        best = all_candidates[0]
    best_err, R_best, t_best, best_angle = best

    if verbose:
        print(f"  -> Full-search result: angle={best_angle:.1f}\u00b0, RMSE={best_err:.4f}")
        for rank, (err, _, _, ang) in enumerate(all_candidates[:3]):
            print(f"    #{rank + 1}: angle={ang:.1f}\u00b0, RMSE={err:.4f}")

    return R_best, t_best, best_angle, best_err

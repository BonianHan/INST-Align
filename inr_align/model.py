"""Neural network components and rigid alignment (PCA + adaptive ICP).

Architecture::

    Phase 1 (INR Pretrain):
        coords1 -> ExprINR -> emb -> Decoder(emb) -> expr1_hat

    Phase 2 (Joint Alignment):
        coords2 -> DeformNet -> coords2_def
        ExprINR(coords1)     -> emb -> P-matrix + Decoder(emb) -> expr1_hat
        ExprINR(coords2_def) -> emb -> P-matrix + Decoder(emb) -> expr2_hat

Components:
- ``WindowedPositionalEncoding``: windowed Fourier positional encoding.
- ``DeformationNet``: spatial-only deformation network.
- ``ExprINR``: coords -> PE -> MLP -> embedding (single INR for both slices).
- ``ExprDecoder``: embedding -> reconstructed expression (no batch info).
- ``UnifiedCostMatcher``: adaptive-temperature soft-assignment matcher.
- ``adaptive_icp``: PCA-guided + full-search ICP for arbitrary rotations.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from inr_align.config import ICPConfig, JointConfig, MatcherConfig, ModelConfig


# ============================================================================
# Positional Encoding
# ============================================================================


class WindowedPositionalEncoding(nn.Module):
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
# Deformation Network (split heads)
# ============================================================================


class DeformationNet(nn.Module):
    """Spatial-only deformation network.

    Outputs per-point displacements ``(dx, dy)``.  The coordinate output is
    ``x + delta(x)`` so that identity mapping is the default at init.

    Embedding is handled by the independent :class:`ExprEncoder`, not by
    this network.  Jacobian regularization calls ``forward()`` directly.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.pe = WindowedPositionalEncoding(config.d, config.n_freqs, config.max_freq_log2)
        self.n_freqs = config.n_freqs

        # --- Trunk: PE -> hidden layers ---
        trunk = []
        in_dim = self.pe.d_out
        for i in range(config.layers):
            trunk.append(nn.Linear(in_dim if i == 0 else config.hidden, config.hidden))
            trunk.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk)

        # --- Spatial head: trunk_hidden -> (dx, dy) ---
        sp_layers = []
        in_d = config.hidden
        for i in range(config.spatial_head_layers):
            sp_layers.append(nn.Linear(in_d if i == 0 else config.spatial_head_hidden, config.spatial_head_hidden))
            sp_layers.append(nn.ReLU())
            in_d = config.spatial_head_hidden
        sp_layers.append(nn.Linear(in_d, config.d))
        self.spatial_head = nn.Sequential(*sp_layers)

        # Near-zero init for displacement output (start at identity)
        with torch.no_grad():
            self.spatial_head[-1].weight.uniform_(-1e-4, 1e-4)
            self.spatial_head[-1].bias.zero_()

    def forward(
        self,
        x: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return deformed coordinates ``x + delta(x)``."""
        h = self.trunk(self.pe(x, alpha))
        return x + self.spatial_head(h)


# ============================================================================
# Expression Encoder (expression -> embedding, independent of coordinates)
# ============================================================================


class ExprEncoder(nn.Module):
    """MLP: per-point expression -> embedding.

    Input is gene expression (PCA features), NOT coordinates.
    This means embeddings are biologically meaningful across slices
    from the very first epoch, without requiring spatial alignment.

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
        """``(N, n_input) -> (N, emb_dim)``  (raw, NOT L2-normalised)."""
        return self.net(x)


# ============================================================================
# Expression INR (coords -> embedding via PE + MLP bottleneck)
# ============================================================================


class ExprINR(nn.Module):
    """Implicit Neural Representation: spatial coords -> embedding.

    Architecture::

        (x, y) -> WindowedPE -> MLP(LayerNorm+ReLU) -> bottleneck(emb_dim)

    A single ExprINR is used for both slices.  Pretrained on slice 1 only
    (Phase 1), then used for both slices in Phase 2 after rigid alignment
    brings coordinates into the same space.

    The bottleneck embedding is fed to an :class:`ExprDecoder` for gene
    expression reconstruction, supervised by SUICA-style loss.
    """

    def __init__(
        self,
        d_in: int = 2,
        emb_dim: int = 64,
        hidden: int = 256,
        n_layers: int = 4,
        n_freqs: int = 6,
        max_freq_log2: int = 5,
    ):
        super().__init__()
        self.pe = WindowedPositionalEncoding(d_in, n_freqs, max_freq_log2)
        self.n_freqs = n_freqs

        layers = []
        in_d = self.pe.d_out
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            in_d = hidden
        layers.append(nn.Linear(in_d, emb_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(
        self,
        coords: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``(N, 2) -> (N, emb_dim)``  (raw, NOT L2-normalised)."""
        return self.backbone(self.pe(coords, alpha))


# ============================================================================
# Expression Decoder (embedding + slice_id -> expression)
# ============================================================================


class ExprDecoder(nn.Module):
    """MLP: embedding -> reconstructed expression.

    Simple decoder that maps INR bottleneck embeddings to gene expression.
    No batch/slice information — the INR embedding is the sole input.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        n_output: int = 50,
        hidden: int = 256,
        n_layers: int = 2,
        **kwargs,  # absorb legacy n_slices
    ):
        super().__init__()
        layers = []
        in_d = emb_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden), nn.ReLU()]
            in_d = hidden
        layers.append(nn.Linear(in_d, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """``(N, emb_dim) -> (N, n_output)``."""
        return self.net(emb)


# ============================================================================
# k-NN graph for spatial smoothing
# ============================================================================


def build_knn_graph(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Pre-compute k-NN indices.  ``(N, 2) -> (N, k)``."""
    tree = cKDTree(np.asarray(coords, dtype=np.float64))
    _, idx = tree.query(coords, k=k + 1)
    return idx[:, 1:]


# ============================================================================
# Factory — build decoder + discriminator from JointConfig
# ============================================================================


def build_joint_models(
    config: JointConfig,
    device: str = "cuda",
) -> Dict[str, nn.Module]:
    """Build a single ExprINR + decoder from config.

    One ExprINR maps spatial coordinates to embeddings for both slices.
    An ExprDecoder reconstructs gene expression from embeddings without
    any batch/slice information (STINR-style).

    Args:
        config: Joint config with INR/decoder architecture params.
        device: Target device.

    Returns:
        Dict with keys ``"expr_inr"``, ``"decoder"``.
    """
    expr_inr = ExprINR(
        d_in=2, emb_dim=config.emb_dim,
        hidden=config.inr_hidden, n_layers=config.inr_layers,
        n_freqs=config.inr_n_freqs, max_freq_log2=config.inr_max_freq_log2,
    ).to(device)

    dec = ExprDecoder(
        emb_dim=config.emb_dim, n_output=config.n_output,
        hidden=config.decoder_hidden, n_layers=config.decoder_layers,
    ).to(device)

    return {"expr_inr": expr_inr, "decoder": dec}


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
        self.spatial_scale: float = 1.0
        self.feat_scale: float = 1.0

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
        self.spatial_scale = 1.0
        self.feat_scale = 1.0


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


def icp_refine(
    A: NDArray,
    B_init: NDArray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Tuple[NDArray, NDArray]:
    """ICP refinement of *B_init* onto *A*."""
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
    A: NDArray, B_aligned: NDArray,
    emb_A: NDArray, emb_B: NDArray,
    k: int = 10,
) -> float:
    """Score rotation candidate by expression similarity."""
    tree = cKDTree(A)
    _, nn_idx = tree.query(B_aligned, k=k)

    emb_A_norm = emb_A / (np.linalg.norm(emb_A, axis=1, keepdims=True) + 1e-8)
    emb_B_norm = emb_B / (np.linalg.norm(emb_B, axis=1, keepdims=True) + 1e-8)

    n = len(B_aligned)
    sims = np.zeros(n)
    for i in range(n):
        nbr_embs = emb_A_norm[nn_idx[i]]
        sims[i] = np.mean(emb_B_norm[i] @ nbr_embs.T)

    return float(np.mean(sims))


def _select_best_with_expression(
    candidates: list, B: NDArray,
    emb_A: NDArray, emb_B: NDArray, A: NDArray,
    verbose: bool = False, tag: str = "",
) -> tuple:
    """Re-rank ICP candidates using expression similarity."""
    scored = []
    for err, R, t, angle in candidates:
        B_aligned = (B @ R.T) + t
        expr_sim = _expression_score(A, B_aligned, emb_A, emb_B, k=10)
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
    A: NDArray, B: NDArray,
    config: Optional[ICPConfig] = None,
    verbose: bool = False,
    emb_A: Optional[NDArray] = None,
    emb_B: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray, float, float]:
    """Adaptive ICP: PCA-guided when confident, full angular search otherwise."""
    if config is None:
        config = ICPConfig()

    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    Bc = B - muB

    # icp_only mode
    if config.mode == "icp_only":
        B_centered = B - muB + muA
        R_icp, t_icp = icp_refine(A, B_centered, max_iter=config.icp_max_iter)
        R_total = R_icp
        t_total = (-muB + muA) @ R_icp.T + t_icp

        B_final = (B @ R_total.T) + t_total
        rmse = nn_rmse(A, B_final)
        angle_deg = float(np.degrees(np.arctan2(R_total[1, 0], R_total[0, 0])))
        if verbose:
            print(f"ICP mode: icp_only (no rotation search)")
            print(f"  angle={angle_deg:.1f}\u00b0, RMSE={rmse:.4f}")
        return R_total, t_total, angle_deg, rmse

    # Phase 1: PCA-guided candidates
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
            best = _select_best_with_expression(
                pca_candidates, B, emb_A, emb_B, A, verbose=verbose, tag="PCA",
            )
        else:
            best = pca_candidates[0]
        best_err, R_best, t_best, best_angle = best
        if verbose:
            print(f"  -> Using PCA result: angle={best_angle:.1f}\u00b0, RMSE={best_err:.4f}")
        return R_best, t_best, best_angle, best_err

    # Phase 2: Full angular search
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

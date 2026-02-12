"""Loss functions: soft matching, Jacobian regularization, expression reconstruction.

Key functions:

- ``compute_P_matrix`` — builds a sparse soft-assignment (transport) plan
  between deformed source and target, using both spatial distance and
  gene-expression cosine similarity.
- ``expression_reconstruction_loss`` — MSE between ExpressionINR predictions
  at deformed coordinates and ground-truth HVG expression. Gradients flow
  back through the deformation network.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_align.model import UnifiedCostMatcher


# ============================================================================
# Sinkhorn normalization (doubly-stochastic transport plan)
# ============================================================================


def _sinkhorn_weights(
    cost: torch.Tensor,
    tau: float,
    n_iters: int,
) -> torch.Tensor:
    """Sinkhorn normalization on a top-k cost matrix.

    Given a cost matrix ``(N, K)`` where K is the top-k neighbourhood size,
    this computes an approximate doubly-stochastic transport plan via
    alternating row/column normalization on the Gibbs kernel.

    Unlike plain softmax which only normalizes rows, Sinkhorn enforces
    that columns (target points) receive roughly equal total mass,
    preventing popular target points from dominating the assignment.

    Note: Because the cost matrix is sparse (N×K, not N×N), true
    doubly-stochastic normalization is approximate.  Column normalization
    is applied across the K-neighbourhood dimension.  This still provides
    a useful marginal-balancing effect compared to pure row-softmax.

    Args:
        cost: ``(N, K)`` unified cost (lower = better match).
        tau: Temperature for the Gibbs kernel.
        n_iters: Number of Sinkhorn iterations.

    Returns:
        ``(N, K)`` normalized weights that sum to 1 per row.
    """
    # Gibbs kernel: K_ij = exp(-C_ij / tau)
    log_K = -cost / tau
    # Stabilize with log-domain Sinkhorn
    # u, v are dual variables in log-space
    log_u = torch.zeros(cost.shape[0], 1, device=cost.device)
    log_v = torch.zeros(1, cost.shape[1], device=cost.device)

    for _ in range(n_iters):
        # Row normalization (so each row sums to 1)
        log_u = -torch.logsumexp(log_K + log_v, dim=1, keepdim=True)
        # Column normalization (so each column sums to ~1)
        log_v = -torch.logsumexp(log_K + log_u, dim=0, keepdim=True)

    # Final weights with row normalization to guarantee row-sum = 1
    log_P = log_K + log_u + log_v
    weights = F.softmax(log_P, dim=1)
    return weights


# ============================================================================
# Soft-assignment P matrix
# ============================================================================


def compute_P_matrix(
    x2_def: torch.Tensor,
    x1: torch.Tensor,
    emb2: torch.Tensor,
    emb1: torch.Tensor,
    matcher: UnifiedCostMatcher,
    topk: int = 64,
    update_tau: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k soft assignment from *x2_def* to *x1*.

    When ``matcher.outlier_weight > 0``, a CPD-style uniform outlier
    component is added (Myronenko & Song, 2010; Qiu et al., 2024).
    Each source point gets an additional "no-match" option that competes
    with the K real neighbours in the softmax.  Points far from all
    targets (e.g. at tissue edges with no corresponding region in the
    other slice) have most of their weight absorbed by the outlier
    component, so their matching loss contribution is automatically
    down-weighted.

    The outlier logit is derived from CPD's posterior::

        c = (w / (1-w)) · (K / N1)

    where ``w`` is the outlier weight fraction.  Converted to a logit
    ``log(c)`` and appended to the cost-based logits before softmax.

    Args:
        x2_def: ``(N2, 2)`` deformed source coordinates.
        x1: ``(N1, 2)`` target coordinates.
        emb2: ``(N2, D)`` source embeddings (should be L2-normalized).
        emb1: ``(N1, D)`` target embeddings (should be L2-normalized).
        matcher: Stateful matcher for temperature and scale.
        topk: Number of spatial nearest-neighbours to consider.
        update_tau: Whether to update the matcher state.

    Returns:
        ``(topk_idx, weights)`` both of shape ``(N2, K)``.
        When outlier is active, ``weights.sum(dim=1) < 1`` for outlier points.
    """
    N2, N1 = x2_def.shape[0], x1.shape[0]
    K = min(topk, N1)

    # Spatial cost
    spatial_dist_sq = torch.cdist(x2_def, x1).pow(2)
    topk_dist_sq, topk_idx = torch.topk(spatial_dist_sq, k=K, dim=1, largest=False)

    # Feature cost (cosine distance)
    e1n = F.normalize(emb1, dim=1)
    e2n = F.normalize(emb2, dim=1)
    e1_neighbors = e1n[topk_idx.reshape(-1)].reshape(N2, K, -1)
    feat_sim = torch.einsum("bd,bkd->bk", e2n, e1_neighbors)
    feat_dist = 1 - feat_sim

    # Update adaptive scales
    if update_tau:
        matcher.update_scales(topk_dist_sq, feat_dist)

    # Unified cost
    spatial_norm = topk_dist_sq / (matcher.spatial_scale + 1e-8)
    feat_norm = feat_dist / (matcher.feat_scale + 1e-8)
    unified_cost = spatial_norm + matcher.lambda_feat * feat_norm

    # Logits for real neighbours
    logits = -unified_cost / matcher.tau  # (N2, K)

    w = matcher.outlier_weight
    if w > 0:
        # CPD-style outlier component (Myronenko & Song, 2010)
        # c = (w/(1-w)) * (K/N1) is the relative strength of the uniform
        # outlier distribution vs. the Gaussian mixture.
        c = (w / (1.0 - w + 1e-8)) * (K / N1)
        outlier_logit = np.log(c + 1e-8)
        # Append outlier column → softmax over K+1 options
        outlier_col = torch.full((N2, 1), outlier_logit, device=logits.device)
        logits_full = torch.cat([logits, outlier_col], dim=1)  # (N2, K+1)
        weights_full = F.softmax(logits_full, dim=1)
        weights = weights_full[:, :K]  # (N2, K) — row sum < 1 for outliers
    elif matcher.sinkhorn_iters > 0:
        weights = _sinkhorn_weights(unified_cost, matcher.tau, matcher.sinkhorn_iters)
    else:
        weights = F.softmax(logits, dim=1)

    if update_tau:
        matcher.update_tau_em(weights, unified_cost)

    return topk_idx, weights


# ============================================================================
# Jacobian regularization
# ============================================================================


def jacobian_reg(
    model: nn.Module,
    x: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """SVD-based Jacobian regularization (isometry penalty).

    Computes the full Jacobian ``J_F = dF/dx`` and penalizes singular values
    deviating from 1: ``mean(sum(log(sigma_i)^2))``.  This prevents local
    shape distortion (shear, anisotropic stretch).

    Args:
        model: Deformation network ``F(x) = x + delta(x)``.
        x: ``(B, 2)`` input coordinates.
        alpha: Coarse-to-fine window scalar.
        eps: Clamp floor for singular values.

    Returns:
        Scalar loss.
    """
    x = x.requires_grad_(True)
    y = model(x, alpha)
    B, D = x.shape
    J = torch.zeros(B, D, D, device=x.device)
    for d in range(D):
        grad_out = torch.zeros_like(y)
        grad_out[:, d] = 1.0
        J[:, d, :] = torch.autograd.grad(y, x, grad_out, create_graph=True, retain_graph=True)[0]

    svals = torch.clamp(torch.linalg.svdvals(J), min=eps)
    return (torch.log(svals) ** 2).sum(dim=1).mean()


# ============================================================================
# Matching losses (convenience wrappers)
# ============================================================================


def compute_matching_loss(
    x2_def: torch.Tensor,
    x1: torch.Tensor,
    emb2: torch.Tensor,
    emb1: torch.Tensor,
    matcher: UnifiedCostMatcher,
    topk: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward matching loss.

    When ``matcher.outlier_weight > 0``, uses CPD-style inlier weighting:
    renormalize weights for target computation, scale per-point loss by
    inlier probability (Myronenko & Song, 2010).

    Returns:
        ``(loss, topk_idx, weights)``
    """
    N2 = x2_def.shape[0]
    K = min(topk, x1.shape[0])
    topk_idx, weights = compute_P_matrix(x2_def, x1, emb2, emb1, matcher, topk, update_tau=True)
    x1_neighbors = x1[topk_idx.reshape(-1)].reshape(N2, K, 2)

    if matcher.outlier_weight > 0:
        p_inlier = weights.sum(dim=1)
        w_norm = weights / (p_inlier.unsqueeze(1) + 1e-8)
        target = torch.einsum("bk,bkd->bd", w_norm, x1_neighbors)
        loss = (p_inlier * (x2_def - target).pow(2).sum(dim=1)).mean()
    else:
        target = torch.einsum("bk,bkd->bd", weights, x1_neighbors)
        loss = (x2_def - target).pow(2).sum(dim=1).mean()
    return loss, topk_idx, weights


def compute_bidirectional_loss(
    x2_def: torch.Tensor,
    x1: torch.Tensor,
    emb2: torch.Tensor,
    emb1: torch.Tensor,
    matcher: UnifiedCostMatcher,
    topk: int = 64,
    weight_rev: float = 1.0,
) -> torch.Tensor:
    """Bidirectional (forward + reverse) matching loss.

    The forward loss aligns deformed source → target.
    The reverse loss aligns target → deformed source.
    When ``matcher.outlier_weight > 0``, CPD-style inlier weighting is applied.
    """
    N2, N1 = x2_def.shape[0], x1.shape[0]
    K = min(topk, N1)

    # Forward: x2_def → x1
    idx_fwd, w_fwd = compute_P_matrix(x2_def, x1, emb2, emb1, matcher, topk, update_tau=True)
    x1_nbrs = x1[idx_fwd.reshape(-1)].reshape(N2, K, 2)

    if matcher.outlier_weight > 0:
        p_fwd = w_fwd.sum(dim=1)
        w_fwd_n = w_fwd / (p_fwd.unsqueeze(1) + 1e-8)
        target_fwd = torch.einsum("bk,bkd->bd", w_fwd_n, x1_nbrs)
        loss_fwd = (p_fwd * (x2_def - target_fwd).pow(2).sum(dim=1)).mean()
    else:
        target_fwd = torch.einsum("bk,bkd->bd", w_fwd, x1_nbrs)
        loss_fwd = (x2_def - target_fwd).pow(2).sum(dim=1).mean()

    # Reverse: x1 → x2_def
    K_rev = min(topk, N2)  # K may differ when x2_def is smaller than x1
    idx_rev, w_rev = compute_P_matrix(x1, x2_def, emb1, emb2, matcher, topk, update_tau=False)
    x2_def_nbrs = x2_def[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)

    if matcher.outlier_weight > 0:
        p_rev = w_rev.sum(dim=1)
        w_rev_n = w_rev / (p_rev.unsqueeze(1) + 1e-8)
        target_rev = torch.einsum("bk,bkd->bd", w_rev_n, x2_def_nbrs)
        loss_rev = (p_rev * (x1 - target_rev).pow(2).sum(dim=1)).mean()
    else:
        target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_def_nbrs)
        loss_rev = (x1 - target_rev).pow(2).sum(dim=1).mean()

    return loss_fwd + weight_rev * loss_rev


# ============================================================================
# Canonical consistency loss (ExprField-based)
# ============================================================================


def canonical_consistency_loss(
    expr_field: nn.Module,
    x2_def: torch.Tensor,
    expr2_gt: torch.Tensor,
) -> torch.Tensor:
    """Canonical consistency loss: deformed coords should predict source expression.

    The ExprField learns a canonical expression field from all slices jointly.
    This loss says: if a source cell with expression ``expr2_gt`` is deformed
    to position ``x2_def``, the canonical field at that position should match
    the cell's actual expression.  This provides an alignment signal that is
    independent of the soft-matching — it uses the cell's own expression as
    ground truth.

    Gradients flow through both ``x2_def`` (to the DeformationNet) and
    through the ExprField (unfrozen backbone adapts to deformed geometry).

    Args:
        expr_field: ``ExprField`` (backbone trainable during deformation).
        x2_def: ``(B, 2)`` deformed source coordinates (with grad).
        expr2_gt: ``(B, G)`` ground-truth source HVG expression (normalized).

    Returns:
        Scalar MSE loss.
    """
    expr_pred = expr_field.canonical(x2_def)
    return F.mse_loss(expr_pred, expr2_gt.detach())


# ============================================================================
# Embedding cosine consistency loss
# ============================================================================


def embedding_cosine_loss(
    expr_field: nn.Module,
    x2_def: torch.Tensor,
    target_fwd: torch.Tensor,
) -> torch.Tensor:
    """Embedding cosine consistency: deformed source and matched target should
    have similar canonical embeddings.

    Uses the frozen ExprField's ``get_embedding`` to extract bottleneck
    embeddings at the deformed source positions and the matched target
    positions, then penalizes cosine dissimilarity.

    Gradients flow through ``x2_def`` to the DeformationNet.
    The ExprField should be frozen (no grad).

    Args:
        expr_field: Pre-trained ``ExprField`` (frozen).
        x2_def: ``(B, 2)`` deformed source coordinates (with grad).
        target_fwd: ``(B, 2)`` matched target positions (detached).

    Returns:
        Scalar loss in ``[0, 2]`` (mean of ``1 - cosine_sim``).
    """
    emb_src = expr_field.get_embedding(x2_def)            # (B, latent_dim)
    emb_tgt = expr_field.get_embedding(target_fwd.detach())  # (B, latent_dim)
    cos_sim = F.cosine_similarity(emb_src, emb_tgt, dim=1)  # (B,)
    return (1 - cos_sim).mean()


# ============================================================================
# Expression reconstruction loss (legacy ExpressionINR)
# ============================================================================


def expression_reconstruction_loss(
    expr_inr: nn.Module,
    x2_def: torch.Tensor,
    expr2_hvg: torch.Tensor,
    idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expression reconstruction loss via the ExpressionINR.

    The ExpressionINR learns a spatial expression field from the target
    (reference) slice.  Given deformed source coordinates ``x2_def``,
    this function predicts the expected HVG expression at those locations
    and compares against the actual source expression ``expr2_hvg``.

    Crucially, gradients flow through ``x2_def`` back to the
    ``DeformationNet``, so minimizing this loss encourages the deformation
    to move source points to locations where the predicted expression
    matches the observed expression — i.e., biologically consistent
    positions.

    Args:
        expr_inr: Pre-trained (optionally fine-tuned) ``ExpressionINR``
            mapping ``(N, 2) coords → (N, n_genes)``.
        x2_def: ``(B, 2)`` deformed source coordinates (with grad).
        expr2_hvg: ``(N2, G)`` or ``(B, G)`` ground-truth HVG expression
            for the source slice (normalized).
        idx: ``(B,)`` batch indices into ``expr2_hvg``.  If ``None``,
            assumes ``expr2_hvg`` is already sliced to batch size ``B``.

    Returns:
        Scalar MSE loss.
    """
    expr_pred = expr_inr(x2_def)  # (B, n_genes) — gradients flow to DeformationNet
    if idx is not None:
        expr_target = expr2_hvg[idx]
    else:
        expr_target = expr2_hvg
    return F.mse_loss(expr_pred, expr_target)

"""Loss functions: soft matching, Jacobian regularization, reconstruction.

Key functions:

- ``compute_P_matrix`` — sparse soft-assignment between deformed source and target.
- ``jacobian_reg`` — SVD-based Jacobian regularization.
- ``sparse_recon_loss`` — sparse-aware reconstruction loss (MSE_nz + L1 + Dice).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_align.model import UnifiedCostMatcher


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

    # Logits → softmax weights
    logits = -unified_cost / matcher.tau  # (N2, K)
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
    compression_weight: float = 5.0,
) -> torch.Tensor:
    """Asymmetric SVD-based Jacobian regularization.

    Penalizes singular values of the Jacobian deviating from 1,
    with compression (σ < 1) penalized ``compression_weight``× more.

    Args:
        model: Deformation network ``F(x) = x + delta(x)``.
        x: ``(B, 2)`` input coordinates.
        alpha: Coarse-to-fine window scalar.
        eps: Clamp floor for singular values.
        compression_weight: Extra penalty multiplier for compression (σ < 1).

    Returns:
        Scalar SVD loss.
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
    log_svals = torch.log(svals)
    weights = torch.where(log_svals < 0, compression_weight, 1.0)
    return (weights * log_svals ** 2).sum(dim=1).mean()


# ============================================================================
# Assignment uniqueness loss — directly penalizes many-to-one mapping
# ============================================================================


def assignment_uniqueness_loss(
    x2_def: torch.Tensor,
    x1: torch.Tensor,
    topk_idx: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize multiple source points mapping to the same target.

    For each target point, accumulates the total soft-assignment weight it
    receives from all source points.  Any target receiving total weight > 1
    is "over-assigned" (many-to-one).  The loss is the variance of the
    per-target load, encouraging a uniform 1-to-1 mapping.

    This directly targets the Ratio metric degradation.

    Args:
        x2_def: ``(N2, 2)`` deformed source coordinates (unused, kept for API).
        x1: ``(N1, 2)`` target coordinates (unused, kept for API).
        topk_idx: ``(N2, K)`` indices into x1 for each source point.
        weights: ``(N2, K)`` soft-assignment weights (sum to 1 per row).

    Returns:
        Scalar loss — variance of per-target load.
    """
    N1 = x1.shape[0]
    N2, K = topk_idx.shape

    # Accumulate total weight each target receives
    # load[j] = sum_i weights[i, k] where topk_idx[i, k] == j
    load = torch.zeros(N1, device=x2_def.device)
    flat_idx = topk_idx.reshape(-1)     # (N2*K,)
    flat_w = weights.reshape(-1)         # (N2*K,)
    load.scatter_add_(0, flat_idx, flat_w)

    # Ideal: load[j] ≈ N2/N1 for all j (uniform distribution)
    # Penalize variance of load — high variance = some targets get lots, others get none
    ideal_load = float(N2) / float(N1)
    uniqueness_loss = (load - ideal_load).pow(2).mean()

    return uniqueness_loss


# ============================================================================
# Embedding KL loss (INR embedding → Splane embedding)
# ============================================================================


def embedding_kl_loss(
    emb_pred: torch.Tensor,
    emb_target: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between INR-predicted embeddings and Splane embeddings.

    Both inputs are treated as unnormalized logits.  Applies log-softmax
    to ``emb_pred`` and softmax to ``emb_target``.

    Args:
        emb_pred: ``(N, D)`` embeddings from DeformationNet.
        emb_target: ``(N, D)`` pre-computed Splane embeddings.

    Returns:
        Scalar KL divergence (batchmean).
    """
    p = F.log_softmax(emb_pred, dim=1)
    q = F.softmax(emb_target, dim=1)
    return F.kl_div(p, q, reduction="batchmean")


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
    """
    N2, N1 = x2_def.shape[0], x1.shape[0]
    K = min(topk, N1)

    # Forward: x2_def → x1
    idx_fwd, w_fwd = compute_P_matrix(x2_def, x1, emb2, emb1, matcher, topk, update_tau=True)
    x1_nbrs = x1[idx_fwd.reshape(-1)].reshape(N2, K, 2)
    target_fwd = torch.einsum("bk,bkd->bd", w_fwd, x1_nbrs)
    loss_fwd = (x2_def - target_fwd).pow(2).sum(dim=1).mean()

    # Reverse: x1 → x2_def
    K_rev = min(topk, N2)
    idx_rev, w_rev = compute_P_matrix(x1, x2_def, emb1, emb2, matcher, topk, update_tau=False)
    x2_def_nbrs = x2_def[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)
    target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_def_nbrs)
    loss_rev = (x1 - target_rev).pow(2).sum(dim=1).mean()

    return loss_fwd + weight_rev * loss_rev


# ============================================================================
# Sparse-aware reconstruction loss
# ============================================================================


def _dice_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss for zero/non-zero pattern preservation.

    Args:
        pred: ``(N, G)`` predicted expression (raw values).
        gt: ``(N, G)`` ground-truth expression.

    Returns:
        Scalar Dice loss in ``[0, 1]``.
    """
    pred_binary = 2.0 * torch.sigmoid(pred) - 1.0
    gt_binary = (gt > 0).float()
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    dice_coeff = (2.0 * intersection + 1.0) / (union + 1.0)
    return 1.0 - dice_coeff


def sparse_recon_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    nz_mask: Optional[torch.Tensor] = None,
    dice_weight: float = 0.01,
) -> torch.Tensor:
    """Sparse-aware reconstruction loss: MSE_nonzero + L1 + Dice.

    Combines three terms:
      1. MSE on non-zero entries (biological signal)
      2. L1 over all entries (sparsity-encouraging, robust to outliers)
      3. Soft Dice loss on zero/non-zero pattern (weighted by ``dice_weight``)

    Args:
        pred: ``(N, G)`` predicted expression.
        gt: ``(N, G)`` ground-truth expression (z-score normalized).
        nz_mask: ``(N, G)`` boolean mask — ``True`` where the *pre-normalized*
            expression was non-zero.  ``None`` uses ``gt != 0`` as fallback.
        dice_weight: Weight for the Dice loss term (default 0.01).

    Returns:
        Scalar loss.
    """
    if nz_mask is None:
        nz_mask = gt != 0

    nz_count = nz_mask.sum()
    if nz_count > 0:
        mse_nz = (pred[nz_mask] - gt[nz_mask]).pow(2).mean()
    else:
        mse_nz = F.mse_loss(pred, gt)

    l1 = F.l1_loss(pred, gt)
    dice = _dice_loss(pred, gt)

    return mse_nz + l1 + dice_weight * dice

"""Loss functions for joint deformation + embedding training.

Core losses:

- ``compute_P_matrix`` — sparse soft-assignment between deformed source and target.
- ``matching_loss_joint`` — bidirectional matching (forward + reverse).
- ``jacobian_reg`` — SVD-based Jacobian regularization (spatial head only).
- ``dice_loss`` — Dice loss for sparse zero/nonzero pattern matching.
- ``recon_loss`` — Dice + masked MSE + L1 reconstruction on HVG expression.
- ``recon_loss_from_emb`` — same as recon_loss but takes precomputed INR embeddings.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from insta.model import UnifiedCostMatcher


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

    Returns:
        ``(topk_idx, weights)`` both of shape ``(N2, K)``.
    """
    N2, N1 = x2_def.shape[0], x1.shape[0]
    K = min(topk, N1)

    spatial_dist_sq = torch.cdist(x2_def, x1).pow(2)
    topk_dist_sq, topk_idx = torch.topk(spatial_dist_sq, k=K, dim=1, largest=False)

    e1n = F.normalize(emb1, dim=1)
    e2n = F.normalize(emb2, dim=1)
    e1_neighbors = e1n[topk_idx.reshape(-1)].reshape(N2, K, -1)
    feat_sim = torch.einsum("bd,bkd->bk", e2n, e1_neighbors)
    feat_dist = 1 - feat_sim

    if update_tau:
        matcher.update_scales(topk_dist_sq, feat_dist)

    spatial_norm = topk_dist_sq / (matcher.spatial_scale + 1e-8)
    feat_norm = feat_dist / (matcher.feat_scale + 1e-8)
    unified_cost = spatial_norm + matcher.lambda_feat * feat_norm

    logits = -unified_cost / matcher.tau
    weights = F.softmax(logits, dim=1)

    if update_tau:
        matcher.update_tau_em(weights, unified_cost)

    return topk_idx, weights


# ============================================================================
# Jacobian regularization (spatial head only)
# ============================================================================


def jacobian_reg(
    model: nn.Module,
    x: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1e-6,
    compression_weight: float = 5.0,
    n_samples: int = 256,
) -> torch.Tensor:
    """Asymmetric SVD-based Jacobian regularization (sampling-based).

    Randomly samples ``n_samples`` points from *x* to compute the Jacobian,
    avoiding the cost of full-batch autograd + SVD.

    Penalizes singular values of the Jacobian deviating from 1,
    with compression (sigma < 1) penalized more heavily.
    """
    B = x.shape[0]
    if B > n_samples:
        idx = torch.randperm(B, device=x.device)[:n_samples]
        x_sub = x[idx]
    else:
        x_sub = x

    x_sub = x_sub.requires_grad_(True)
    y = model(x_sub, alpha)
    Bs, D = x_sub.shape
    J = torch.zeros(Bs, D, D, device=x_sub.device)
    for d in range(D):
        grad_out = torch.zeros_like(y)
        grad_out[:, d] = 1.0
        J[:, d, :] = torch.autograd.grad(y, x_sub, grad_out, create_graph=True, retain_graph=True)[0]

    svals = torch.clamp(torch.linalg.svdvals(J), min=eps)
    log_svals = torch.log(svals)
    weights = torch.where(log_svals < 0, compression_weight, 1.0)
    return (weights * log_svals ** 2).sum(dim=1).mean()

# ============================================================================
# Bidirectional matching loss
# ============================================================================


def matching_loss_joint(
    x2_def: torch.Tensor,
    x1: torch.Tensor,
    emb2: torch.Tensor,
    emb1: torch.Tensor,
    matcher: UnifiedCostMatcher,
    topk: int = 64,
    weight_rev: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bidirectional matching loss.

    Gradients flow to DeformationNet (via x2_def) and optionally to
    embed_head (via emb2, if not detached).

    Returns:
        ``(loss, topk_idx_fwd, weights_fwd)``
    """
    N2, N1 = x2_def.shape[0], x1.shape[0]
    K = min(topk, N1)

    emb1n = F.normalize(emb1, dim=1)
    emb2n = F.normalize(emb2, dim=1)

    # Forward: x2_def -> x1
    idx_fwd, w_fwd = compute_P_matrix(
        x2_def, x1, emb2n, emb1n, matcher, topk, update_tau=True,
    )
    x1_nbrs = x1[idx_fwd.reshape(-1)].reshape(N2, K, 2)
    target_fwd = torch.einsum("bk,bkd->bd", w_fwd, x1_nbrs)
    loss_fwd = (x2_def - target_fwd).pow(2).sum(dim=1).mean()

    # Reverse: x1 -> x2_def
    K_rev = min(topk, N2)
    idx_rev, w_rev = compute_P_matrix(
        x1, x2_def, emb1n, emb2n, matcher, topk, update_tau=False,
    )
    x2_nbrs = x2_def[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)
    target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_nbrs)
    loss_rev = (x1 - target_rev).pow(2).sum(dim=1).mean()

    loss = loss_fwd + weight_rev * loss_rev
    return loss, idx_fwd, w_fwd


# ============================================================================
# Reconstruction loss (embed_head -> Decoder -> expression)
# ============================================================================


def dice_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Dice loss for sparse gene expression (SUICA-style).

    Treats reconstruction as a quasi-classification problem:
    did the model correctly predict which genes are expressed vs. silent?
    """
    pred_binary = 2.0 * torch.sigmoid(pred) - 1
    gt_binary = (gt > 0).float()
    intersection = (pred_binary * gt_binary).sum()
    union = pred_binary.sum() + gt_binary.sum()
    dice_coeff = (2.0 * intersection + 1.0) / (union + 1.0)
    return 1.0 - dice_coeff


def recon_loss(
    encoder: nn.Module,
    decoder: nn.Module,
    expr_input: torch.Tensor,
    expr_target: torch.Tensor,
) -> torch.Tensor:
    """Self-reconstruction: expr -> encoder -> decoder(emb) -> expr_hat.

    Uses SUICA-style three-component loss for sparse gene expression:
    1. Masked MSE — only on nonzero entries (get magnitudes right)
    2. L1 / MAE — on all entries (encourage sparsity)
    3. Dice loss — penalizes incorrect zero/nonzero pattern

    Args:
        encoder: Expression encoder (input: PCA features or HVG).
        decoder: Expression decoder (output: HVG expression).
        expr_input: ``(N, D_in)`` encoder input (PCA features).
        expr_target: ``(N, D_out)`` reconstruction target (HVG expression,
            log-normalized).
    """
    emb = encoder(expr_input)
    recon = decoder(emb)

    # 1) MSE only on nonzero entries
    nonzero_mask = (expr_target != 0)
    if nonzero_mask.any():
        mse = F.mse_loss(recon[nonzero_mask], expr_target[nonzero_mask])
    else:
        mse = F.mse_loss(recon, expr_target)

    # 2) L1 on everything (sparsity)
    l1 = F.l1_loss(recon, expr_target)

    # 3) Dice loss (zero/nonzero pattern)
    dice = dice_loss(recon, expr_target)

    return mse + l1 + 0.01 * dice


def recon_loss_from_emb(
    emb: torch.Tensor,
    decoder: nn.Module,
    expr_target: torch.Tensor,
) -> torch.Tensor:
    """Reconstruction loss from precomputed embeddings.

    Same SUICA-style three-component loss as :func:`recon_loss`, but takes
    embeddings directly instead of running through an encoder.  This is used
    with :class:`ExprINR` where the caller manages the INR forward pass.

    Args:
        emb: ``(N, emb_dim)`` precomputed embeddings from an INR.
        decoder: ``ExprDecoder`` (embedding -> expression, no batch info).
        expr_target: ``(N, G)`` reconstruction target (HVG expression,
            log-normalized).
    """
    recon = decoder(emb)

    # 1) MSE only on nonzero entries
    nonzero_mask = (expr_target != 0)
    if nonzero_mask.any():
        mse = F.mse_loss(recon[nonzero_mask], expr_target[nonzero_mask])
    else:
        mse = F.mse_loss(recon, expr_target)

    # 2) L1 on everything (sparsity)
    l1 = F.l1_loss(recon, expr_target)

    # 3) Dice loss (zero/nonzero pattern)
    dice = dice_loss(recon, expr_target)

    return mse + l1 + 0.01 * dice

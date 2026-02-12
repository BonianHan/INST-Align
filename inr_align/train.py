"""Training loop and inference utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_align.config import TrainConfig
from inr_align.loss import (
    canonical_consistency_loss, compute_P_matrix,
    embedding_cosine_loss, expression_reconstruction_loss,
    jacobian_reg,
)
from inr_align.model import DeformationNet, UnifiedCostMatcher


# ============================================================================
# Training result container
# ============================================================================


@dataclass
class TrainResult:
    """Returned by :func:`train`."""

    model: DeformationNet
    matcher: UnifiedCostMatcher
    best_match_loss: float
    training_time: float
    history: List[Dict[str, float]] = field(default_factory=list)
    expr_inr: Optional[Any] = None  # Trained ExpressionINR if used


# ============================================================================
# Training loop
# ============================================================================


def train(
    model: DeformationNet,
    matcher: UnifiedCostMatcher,
    x1: torch.Tensor,
    emb1: torch.Tensor,
    x2: torch.Tensor,
    emb2: torch.Tensor,
    config: Optional[TrainConfig] = None,
    # --- ExprField ---
    expr_field: Optional[nn.Module] = None,
    lam_canonical: float = 0.0,
    expr2_canon: Optional[torch.Tensor] = None,
    lam_embed_cos: float = 0.0,
    # --- Expression INR (legacy, optional) ---
    expr_inr: Optional[nn.Module] = None,
    expr2_hvg: Optional[torch.Tensor] = None,
    lam_recon: float = 0.0,
    finetune_lr_factor: float = 0.0,
) -> TrainResult:
    """Train the deformation network.

    Args:
        model: ``DeformationNet`` (on device).
        matcher: ``UnifiedCostMatcher`` instance.
        x1: ``(N1, 2)`` reference coordinates (normalized, on device).
        emb1: ``(N1, D)`` reference PCA embeddings (on device).
        x2: ``(N2, 2)`` source coordinates after rigid alignment (on device).
        emb2: ``(N2, D)`` source PCA embeddings (on device).
        config: Training hyper-parameters.
        expr_field: Pre-trained ``ExprField`` (optional).  When provided
            with ``lam_canonical > 0``, canonical consistency loss is added.
            Backbone is unfrozen and fine-tuned with lower LR.
        lam_canonical: Weight for canonical consistency loss.
        expr2_canon: ``(N2, G)`` source HVG expression (normalized, on device).
            Required when ``expr_field`` is provided.
        expr_inr: Pre-trained ``ExpressionINR`` (legacy, optional).
        expr2_hvg: ``(N2, G)`` source HVG expression (normalized, on device).
        lam_recon: Weight for the expression reconstruction loss.
        finetune_lr_factor: If ``0`` the ExpressionINR is frozen.

    Returns:
        :class:`TrainResult` with the best-checkpoint model restored.
    """
    if config is None:
        config = TrainConfig()

    device = x1.device
    N1, N2 = x1.shape[0], x2.shape[0]
    n_freqs = model.n_freqs
    K = min(config.topk, N1)

    emb1_norm = F.normalize(emb1, dim=1)
    emb2_norm = F.normalize(emb2, dim=1)

    # --- ExprField setup ---
    use_canonical = (expr_field is not None and lam_canonical > 0 and expr2_canon is not None)
    use_embed_cos = (expr_field is not None and lam_embed_cos > 0)
    if use_embed_cos and not use_canonical:
        # Freeze ExprField entirely for embedding cosine loss
        expr_field.requires_grad_(False)

    # --- Expression INR setup (legacy) ---
    use_recon = (expr_inr is not None and expr2_hvg is not None and lam_recon > 0)
    if use_recon and finetune_lr_factor == 0:
        expr_inr.requires_grad_(False)

    # --- Optimizer ---
    param_groups = [{"params": model.parameters(), "lr": config.lr}]
    if use_recon and finetune_lr_factor > 0:
        param_groups.append({
            "params": expr_inr.parameters(),
            "lr": config.lr * finetune_lr_factor,
        })
    if use_canonical:
        # Fine-tune entire ExprField with lower LR
        param_groups.append({
            "params": expr_field.parameters(),
            "lr": config.lr * 0.1,
        })
    optimizer = torch.optim.Adam(param_groups)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    batch_size = min(config.batch_size, N2)

    best_match = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []
    start = time.time()
    prev_lr = config.lr

    for ep in range(config.epochs):
        # Nerfies coarse-to-fine schedule
        warmup = int(config.epochs * config.warmup_fraction)
        alpha = n_freqs * (ep / warmup) if ep < warmup else float(n_freqs)
        alpha_t = torch.tensor(alpha, device=device)

        perm = torch.randperm(N2, device=device)
        epoch_loss = 0.0
        epoch_match = 0.0
        epoch_recon = 0.0
        epoch_canon = 0.0
        epoch_emb_cos = 0.0
        epoch_inlier = 0.0  # mean inlier probability (CPD outlier)
        n_batch = 0

        for bi in range(0, N2, batch_size):
            idx = perm[bi : min(bi + batch_size, N2)]
            bs = idx.shape[0]
            x2_batch = x2[idx]
            emb2_batch = emb2_norm[idx]

            optimizer.zero_grad(set_to_none=True)
            x2_input = x2_batch.requires_grad_(True)
            x2_def = model(x2_input, alpha_t)

            # Forward loss: deformed source → target
            idx_fwd, w_fwd = compute_P_matrix(
                x2_def, x1, emb2_batch, emb1_norm, matcher, config.topk, update_tau=True
            )
            x1_nbrs = x1[idx_fwd.reshape(-1)].reshape(bs, K, 2)

            # CPD-style outlier handling (Myronenko & Song, 2010):
            # When outlier_weight > 0, w_fwd.sum(dim=1) < 1 for outlier points.
            # Renormalize weights for target computation; weight loss by p_inlier.
            if matcher.outlier_weight > 0:
                p_inlier_fwd = w_fwd.sum(dim=1)  # (bs,) inlier probability
                w_fwd_norm = w_fwd / (p_inlier_fwd.unsqueeze(1) + 1e-8)
                target_fwd = torch.einsum("bk,bkd->bd", w_fwd_norm, x1_nbrs)
                loss_fwd = (p_inlier_fwd * (x2_def - target_fwd).pow(2).sum(dim=1)).mean()
            else:
                target_fwd = torch.einsum("bk,bkd->bd", w_fwd, x1_nbrs)
                loss_fwd = (x2_def - target_fwd).pow(2).sum(dim=1).mean()

            # Reverse loss: target → deformed source
            K_rev = min(config.topk, bs)  # K may differ when last batch is small
            idx_rev, w_rev = compute_P_matrix(
                x1, x2_def, emb1_norm, emb2_batch, matcher, config.topk, update_tau=False
            )
            x2_def_nbrs = x2_def[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)

            if matcher.outlier_weight > 0:
                p_inlier_rev = w_rev.sum(dim=1)  # (N1,) inlier probability
                w_rev_norm = w_rev / (p_inlier_rev.unsqueeze(1) + 1e-8)
                target_rev = torch.einsum("bk,bkd->bd", w_rev_norm, x2_def_nbrs)
                loss_rev = (p_inlier_rev * (x1 - target_rev).pow(2).sum(dim=1)).mean()
            else:
                target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_def_nbrs)
                loss_rev = (x1 - target_rev).pow(2).sum(dim=1).mean()

            L_match = loss_fwd + config.weight_rev * loss_rev

            # Jacobian regularization (SVD: isometry)
            loss = L_match
            if config.lam_jacobian > 0:
                L_jac = jacobian_reg(model, x2_input, alpha_t)
                loss = loss + config.lam_jacobian * L_jac

            # Expression reconstruction loss (legacy, only after warmup)
            L_recon_val = 0.0
            if use_recon and ep >= warmup:
                L_recon = expression_reconstruction_loss(expr_inr, x2_def, expr2_hvg, idx)
                loss = loss + lam_recon * L_recon
                L_recon_val = L_recon.item()

            # Canonical consistency loss (ExprField, ramp from 0 → lam during warmup, full after)
            L_canon_val = 0.0
            if use_canonical:
                ramp = min(ep / max(warmup, 1), 1.0)  # 0→1 during warmup, 1.0 after
                canon_weight = lam_canonical * ramp
                L_canon = canonical_consistency_loss(expr_field, x2_def, expr2_canon[idx])
                loss = loss + canon_weight * L_canon
                L_canon_val = L_canon.item()

            # Embedding cosine consistency loss (frozen ExprField, after warmup)
            L_emb_cos_val = 0.0
            if use_embed_cos and ep >= warmup:
                L_emb_cos = embedding_cosine_loss(expr_field, x2_def, target_fwd)
                loss = loss + lam_embed_cos * L_emb_cos
                L_emb_cos_val = L_emb_cos.item()

            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_match += L_match.item()
            epoch_recon += L_recon_val
            epoch_canon += L_canon_val
            epoch_emb_cos += L_emb_cos_val
            if matcher.outlier_weight > 0:
                epoch_inlier += p_inlier_fwd.mean().item()
            n_batch += 1

        avg_match = epoch_match / n_batch
        if avg_match < best_match:
            best_match = avg_match
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # LR scheduler step
        scheduler.step(avg_match)
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": ep, "loss": epoch_loss / n_batch, "match": avg_match,
            "tau": matcher.tau, "lr": cur_lr,
        }
        if use_recon:
            record["recon"] = epoch_recon / n_batch
        if use_canonical:
            record["canonical"] = epoch_canon / n_batch
        if use_embed_cos:
            record["emb_cos"] = epoch_emb_cos / n_batch
        history.append(record)

        if ep % config.print_every == 0 or ep == config.epochs - 1:
            recon_str = f" recon={epoch_recon / n_batch:.5f}" if use_recon else ""
            canon_str = f" canon={epoch_canon / n_batch:.5f}" if use_canonical else ""
            emb_cos_str = f" emb_cos={epoch_emb_cos / n_batch:.5f}" if use_embed_cos else ""
            inlier_str = f" inlier={epoch_inlier / n_batch:.3f}" if matcher.outlier_weight > 0 else ""
            print(
                f"  ep={ep:03d} | lr={cur_lr:.1e} | \u03c4={matcher.tau:.5f} | "
                f"loss={epoch_loss / n_batch:.5f} match={avg_match:.5f}{recon_str}{canon_str}{emb_cos_str}{inlier_str}"
            )

        # Log LR changes
        if cur_lr < prev_lr:
            print(f"  >> LR reduced: {prev_lr:.1e} -> {cur_lr:.1e} (patience={config.scheduler_patience})")
            prev_lr = cur_lr

    # Restore best checkpoint
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    elapsed = time.time() - start
    return TrainResult(
        model=model,
        matcher=matcher,
        best_match_loss=best_match,
        training_time=elapsed,
        history=history,
        expr_inr=expr_inr,
    )


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def apply_model(
    model: DeformationNet,
    x: torch.Tensor,
    snap_to_grid: bool = False,
) -> torch.Tensor:
    """Apply the trained deformation network.

    Reads ``n_freqs`` from the model so it does not need to be passed
    explicitly.
    """
    alpha = torch.tensor(float(model.n_freqs), device=x.device)
    return model(x, alpha, snap_to_grid=snap_to_grid)

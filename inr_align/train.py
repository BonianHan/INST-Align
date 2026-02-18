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
    assignment_uniqueness_loss,
    compute_P_matrix,
    embedding_kl_loss,
    jacobian_reg,
    sparse_recon_loss,
)
from inr_align.model import DeformationNet, ExprField, GeneDecoder, UnifiedCostMatcher


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
    expr_field: Optional[ExprField] = None
    gene_decoder: Optional[GeneDecoder] = None


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
    # --- ExprField (joint training, legacy) ---
    expr_field: Optional[ExprField] = None,
    expr2_gt: Optional[torch.Tensor] = None,
    lam_expr: float = 0.0,
    # --- Splane embedding KL ---
    splane_emb2: Optional[torch.Tensor] = None,
    # --- Gene reconstruction (GeneDecoder) ---
    gene_decoder: Optional[GeneDecoder] = None,
    gene_expr2_gt: Optional[torch.Tensor] = None,
    slice_ids2: Optional[torch.Tensor] = None,
) -> TrainResult:
    """Train the deformation network.

    Losses:
      1. Bidirectional soft matching (forward + reverse)
      2. Jacobian regularization (SVD isometry)
      3. Splane embedding KL (DeformationNet emb_head → Splane target)
      4. Gene reconstruction (GeneDecoder: emb + batch → gene expression)
      5. Expression reconstruction (ExprField, legacy joint training)

    Args:
        model: ``DeformationNet`` (on device).
        matcher: ``UnifiedCostMatcher`` instance.
        x1: ``(N1, 2)`` reference coordinates (normalized, on device).
        emb1: ``(N1, D)`` reference PCA embeddings (on device).
        x2: ``(N2, 2)`` source coordinates after rigid alignment (on device).
        emb2: ``(N2, D)`` source PCA embeddings (on device).
        config: Training hyper-parameters.
        expr_field: ``ExprField`` to train jointly (optional, legacy).
        expr2_gt: ``(N2, G)`` source expression for ExprField (normalized).
        lam_expr: Weight for ExprField expression reconstruction loss.
        splane_emb2: ``(N2, E)`` pre-computed Splane embeddings for source
            (on device). Required for KL loss when ``config.lam_kl > 0``.
        gene_decoder: ``GeneDecoder`` for gene reconstruction.
        gene_expr2_gt: ``(N2, G)`` source gene expression for gene recon.
        slice_ids2: ``(N2,)`` integer slice IDs for GeneDecoder batch_emb.

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

    use_expr = (expr_field is not None and expr2_gt is not None and lam_expr > 0)
    use_kl = (splane_emb2 is not None and config.lam_kl > 0 and model.emb_head is not None)
    use_recon = (gene_decoder is not None and gene_expr2_gt is not None
                 and slice_ids2 is not None and config.lam_recon > 0)

    # --- Optimizer ---
    param_groups = [{"params": model.parameters(), "lr": config.lr}]
    if use_expr:
        param_groups.append({"params": expr_field.parameters(), "lr": config.lr})
    if use_recon:
        param_groups.append({"params": gene_decoder.parameters(), "lr": config.lr})
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
    best_expr_state = None
    best_decoder_state = None
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
        epoch_expr = 0.0
        epoch_uniq = 0.0
        epoch_kl = 0.0
        epoch_recon = 0.0
        n_batch = 0

        for bi in range(0, N2, batch_size):
            idx = perm[bi : min(bi + batch_size, N2)]
            bs = idx.shape[0]
            x2_batch = x2[idx]
            emb2_batch = emb2_norm[idx]

            optimizer.zero_grad(set_to_none=True)
            x2_input = x2_batch.requires_grad_(True)

            # Forward pass — get both coords and embedding if available
            if use_kl or use_recon:
                x2_def, emb_pred = model.forward_with_emb(x2_input, alpha_t)
            else:
                x2_def = model(x2_input, alpha_t)
                emb_pred = None

            # Forward loss: deformed source → target
            idx_fwd, w_fwd = compute_P_matrix(
                x2_def, x1, emb2_batch, emb1_norm, matcher, config.topk, update_tau=True
            )
            x1_nbrs = x1[idx_fwd.reshape(-1)].reshape(bs, K, 2)
            target_fwd = torch.einsum("bk,bkd->bd", w_fwd, x1_nbrs)
            loss_fwd = (x2_def - target_fwd).pow(2).sum(dim=1).mean()

            # Reverse loss: target → deformed source
            K_rev = min(config.topk, bs)
            idx_rev, w_rev = compute_P_matrix(
                x1, x2_def, emb1_norm, emb2_batch, matcher, config.topk, update_tau=False
            )
            x2_def_nbrs = x2_def[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)
            target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_def_nbrs)
            loss_rev = (x1 - target_rev).pow(2).sum(dim=1).mean()

            L_match = loss_fwd + config.weight_rev * loss_rev

            # Assignment uniqueness loss — penalizes many-to-one mapping
            L_uniq_val = 0.0
            if config.lam_uniqueness > 0:
                L_uniq = assignment_uniqueness_loss(x2_def, x1, idx_fwd, w_fwd)
                L_uniq_val = L_uniq.item()

            # Jacobian regularization (SVD)
            loss = L_match
            if config.lam_uniqueness > 0:
                loss = loss + config.lam_uniqueness * L_uniq
            if config.lam_jacobian > 0:
                L_svd = jacobian_reg(model, x2_input, alpha_t)
                loss = loss + config.lam_jacobian * L_svd

            # Splane embedding KL loss (after warmup)
            L_kl_val = 0.0
            if use_kl and emb_pred is not None and ep >= warmup:
                splane_batch = splane_emb2[idx]
                L_kl = embedding_kl_loss(emb_pred, splane_batch)
                loss = loss + config.lam_kl * L_kl
                L_kl_val = L_kl.item()

            # Gene reconstruction loss (GeneDecoder, after warmup)
            L_recon_val = 0.0
            if use_recon and emb_pred is not None and ep >= warmup:
                sid_batch = slice_ids2[idx]
                gene_pred = gene_decoder(emb_pred, sid_batch)
                gene_target = gene_expr2_gt[idx]
                L_recon = sparse_recon_loss(gene_pred, gene_target)
                loss = loss + config.lam_recon * L_recon
                L_recon_val = L_recon.item()

            # Expression reconstruction loss (ExprField, legacy, after warmup)
            L_expr_val = 0.0
            if use_expr and ep >= warmup:
                expr_pred = expr_field(x2_def)
                expr_target = expr2_gt[idx]
                L_expr = sparse_recon_loss(expr_pred, expr_target)
                loss = loss + lam_expr * L_expr
                L_expr_val = L_expr.item()

            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if use_expr:
                    torch.nn.utils.clip_grad_norm_(expr_field.parameters(), config.grad_clip)
                if use_recon:
                    torch.nn.utils.clip_grad_norm_(gene_decoder.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_match += L_match.item()
            epoch_expr += L_expr_val
            epoch_uniq += L_uniq_val
            epoch_kl += L_kl_val
            epoch_recon += L_recon_val
            n_batch += 1

        # --- Full-coverage reverse loss (separate optimizer step) ---
        # Ensures every x1 point can find its nearest match across ALL
        # deformed source points, not just the current batch.
        if config.full_reverse_interval > 0 and ep % config.full_reverse_interval == 0:
            optimizer.zero_grad(set_to_none=True)
            x2_all_input = x2.requires_grad_(True)
            if use_kl or use_recon:
                x2_def_all, _ = model.forward_with_emb(x2_all_input, alpha_t)
            else:
                x2_def_all = model(x2_all_input, alpha_t)
            K_rev_full = min(config.topk, N2)
            idx_rev_full, w_rev_full = compute_P_matrix(
                x1, x2_def_all, emb1_norm, emb2_norm, matcher, config.topk, update_tau=False
            )
            x2_nbrs_full = x2_def_all[idx_rev_full.reshape(-1)].reshape(N1, K_rev_full, 2)
            target_rev_full = torch.einsum("bk,bkd->bd", w_rev_full, x2_nbrs_full)
            loss_rev_full = config.weight_rev * (x1 - target_rev_full).pow(2).sum(dim=1).mean()
            loss_rev_full.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        avg_match = epoch_match / n_batch
        if avg_match < best_match:
            best_match = avg_match
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if use_expr:
                best_expr_state = {k: v.cpu().clone() for k, v in expr_field.state_dict().items()}
            if use_recon:
                best_decoder_state = {k: v.cpu().clone() for k, v in gene_decoder.state_dict().items()}

        # LR scheduler step
        scheduler.step(avg_match)
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": ep, "loss": epoch_loss / n_batch, "match": avg_match,
            "tau": matcher.tau, "lr": cur_lr,
        }
        if config.lam_uniqueness > 0:
            record["uniq"] = epoch_uniq / n_batch
        if use_kl:
            record["kl"] = epoch_kl / n_batch
        if use_recon:
            record["recon"] = epoch_recon / n_batch
        if use_expr:
            record["expr"] = epoch_expr / n_batch
        history.append(record)

        if ep % config.print_every == 0 or ep == config.epochs - 1:
            parts = [f"ep={ep:03d} | lr={cur_lr:.1e} | \u03c4={matcher.tau:.5f}"]
            parts.append(f"loss={epoch_loss / n_batch:.5f} match={avg_match:.5f}")
            if config.lam_uniqueness > 0:
                parts.append(f"uniq={epoch_uniq / n_batch:.6f}")
            if use_kl:
                parts.append(f"kl={epoch_kl / n_batch:.5f}")
            if use_recon:
                parts.append(f"recon={epoch_recon / n_batch:.5f}")
            if use_expr:
                parts.append(f"expr={epoch_expr / n_batch:.5f}")
            print(f"  {' '.join(parts)}")

        # Log LR changes
        if cur_lr < prev_lr:
            print(f"  >> LR reduced: {prev_lr:.1e} -> {cur_lr:.1e} (patience={config.scheduler_patience})")
            prev_lr = cur_lr

    # Restore best checkpoint
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if best_expr_state and expr_field is not None:
        expr_field.load_state_dict({k: v.to(device) for k, v in best_expr_state.items()})
    if best_decoder_state and gene_decoder is not None:
        gene_decoder.load_state_dict({k: v.to(device) for k, v in best_decoder_state.items()})

    elapsed = time.time() - start
    return TrainResult(
        model=model,
        matcher=matcher,
        best_match_loss=best_match,
        training_time=elapsed,
        history=history,
        expr_field=expr_field,
        gene_decoder=gene_decoder,
    )


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def apply_model(
    model: DeformationNet,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply the trained deformation network.

    Reads ``n_freqs`` from the model so it does not need to be passed
    explicitly.
    """
    alpha = torch.tensor(float(model.n_freqs), device=x.device)
    return model(x, alpha)

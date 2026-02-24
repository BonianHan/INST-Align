"""Two-phase training loop and inference utilities.

Phase 1 — INR Pretrain (``jcfg.inr_pretrain_epochs`` iterations):
    Two independent ExprINRs each learn their own slice's expression field,
    sharing a single Decoder to force embedding-space alignment:
        coords1 → ExprINR_s1 → emb → SharedDecoder(emb) → expr1_hat
        coords2 → ExprINR_s2 → emb → SharedDecoder(emb) → expr2_hat

Phase 2 — Joint Alignment (``config.epochs`` iterations):
    DeformNet learns deformation via matching loss.
    Only ExprINR_s1 is used — x2_def is in s1's canonical space after deformation,
    so a single INR can encode both sides of the P-matrix.
    ExprINR_s1 + decoder continue learning via recon loss (reduced weight).
    As DeformNet improves → x2_def more accurate → ExprINR_s1(x2_def) more meaningful
    → P-matrix feature_dist more accurate → better matching → virtuous cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_align.config import JointConfig, TrainConfig
from inr_align.loss import (
    assignment_uniqueness_loss,
    compute_P_matrix,
    jacobian_reg,
    matching_loss_joint,
    recon_loss_from_emb,
)
from inr_align.model import (
    DeformationNet,
    ExprDecoder,
    ExprINR,
    UnifiedCostMatcher,
)


# ============================================================================
# Training result container
# ============================================================================


@dataclass
class TrainResult:
    """Returned by :func:`train`."""

    deform: DeformationNet
    matcher: UnifiedCostMatcher
    best_match_loss: float
    training_time: float
    history: List[Dict[str, float]] = field(default_factory=list)
    expr_inr: Optional[Any] = None       # alias → expr_inr_s1 (backward compat)
    expr_inr_s1: Optional[Any] = None    # reference INR
    expr_inr_s2: Optional[Any] = None    # source INR
    decoder: Optional[Any] = None


# ============================================================================
# Two-phase training loop
# ============================================================================


def train(
    deform: DeformationNet,
    matcher: UnifiedCostMatcher,
    x1: torch.Tensor,
    emb1_pca: torch.Tensor,
    x2: torch.Tensor,
    emb2_pca: torch.Tensor,
    config: Optional[TrainConfig] = None,
    jcfg: Optional[JointConfig] = None,
    # --- INR components (None = PCA-only alignment, no embedding) ---
    expr_inr_s1: Optional[ExprINR] = None,
    expr_inr_s2: Optional[ExprINR] = None,
    decoder: Optional[ExprDecoder] = None,
    hvg1: Optional[torch.Tensor] = None,
    hvg2: Optional[torch.Tensor] = None,
    # --- Backward compat: single INR alias ---
    expr_inr: Optional[ExprINR] = None,
) -> TrainResult:
    """Train the deformation network with two-phase dual-INR strategy.

    Phase 1 — INR Pretrain (``jcfg.inr_pretrain_epochs``):
        - Two independent ExprINRs learn each slice's expression field.
        - Shared decoder forces embedding-space alignment.
        - coords1 → ExprINR_s1 → emb → SharedDecoder → expr1_hat
        - coords2 → ExprINR_s2 → emb → SharedDecoder → expr2_hat
        - Loss: (recon_s1 + recon_s2) / 2.
        - DeformNet pretrained independently with PCA matching.

    Phase 2 — Joint Alignment (``config.epochs``):
        - Only ExprINR_s1 used (canonical space): x2_def is in s1's space.
        - P-matrix: spatial_dist + feat_dist(ExprINR_s1(x2_def), ExprINR_s1(x1)).
        - DeformNet + ExprINR_s1 + decoder trained jointly.
        - ExprINR_s2 is frozen/unused in Phase 2.
        - Recon loss at reduced weight (``lam_recon_phase2``).

    Args:
        deform: ``DeformationNet`` (spatial-only, on device).
        matcher: ``UnifiedCostMatcher`` instance.
        x1: ``(N1, 2)`` reference coords (normalized, on device).
        emb1_pca: ``(N1, D)`` reference PCA embeddings (on device).
        x2: ``(N2, 2)`` source coords after rigid alignment (on device).
        emb2_pca: ``(N2, D)`` source PCA embeddings (on device).
        config: Training hyper-parameters.
        jcfg: Joint config (INR/decoder params + loss weights).
        expr_inr_s1: ``ExprINR`` for reference slice (optional).
        expr_inr_s2: ``ExprINR`` for source slice (optional).
        decoder: ``ExprDecoder`` shared across both INRs (optional).
        hvg1: ``(N1, G)`` reference HVG expression (decoder recon target).
        hvg2: ``(N2, G)`` source HVG expression (decoder recon target).
        expr_inr: Backward compat alias — if provided and expr_inr_s1 is None,
            uses this as expr_inr_s1 (single-INR fallback).

    Returns:
        :class:`TrainResult` with best-checkpoint models restored.
    """
    if config is None:
        config = TrainConfig()
    if jcfg is None:
        jcfg = JointConfig()

    # Backward compat: single expr_inr → use as s1, s2 shares same object
    if expr_inr_s1 is None and expr_inr is not None:
        expr_inr_s1 = expr_inr
    if expr_inr_s2 is None and expr_inr_s1 is not None:
        # If only s1 provided, s2 falls back to s1 (legacy single-INR mode)
        expr_inr_s2 = expr_inr_s1

    device = x1.device
    N1, N2 = x1.shape[0], x2.shape[0]
    n_freqs_deform = deform.n_freqs
    batch_size = min(config.batch_size, max(N1, N2))

    # Detect INR mode: need both INRs + decoder + HVG data
    use_inr = (expr_inr_s1 is not None
               and expr_inr_s2 is not None
               and decoder is not None
               and hvg1 is not None and hvg2 is not None)
    dual_inr = use_inr and (expr_inr_s1 is not expr_inr_s2)

    # L2-normalize PCA features for P-matrix (fallback when no INR)
    emb1_pca_norm = F.normalize(emb1_pca, dim=1)
    emb2_pca_norm = F.normalize(emb2_pca, dim=1)

    history: List[Dict[str, float]] = []
    start = time.time()

    # ==================================================================
    # Phase 1: Pretrain (dual INR recon + DeformNet matching, independent)
    # ==================================================================
    if use_inr and jcfg.inr_pretrain_epochs > 0:
        mode_str = "dual INR (s1+s2)" if dual_inr else "single INR (s1 only)"
        print(f"\n  Phase 1: Pretrain ({jcfg.inr_pretrain_epochs} epochs)"
              f" — {mode_str} + DeformNet PCA matching (independent)")
        n_freqs_inr = expr_inr_s1.n_freqs

        # Single optimizer for both INRs + shared decoder
        inr_params = list(expr_inr_s1.parameters()) + list(decoder.parameters())
        if dual_inr:
            inr_params += list(expr_inr_s2.parameters())
        opt_inr = torch.optim.Adam(inr_params, lr=jcfg.inr_pretrain_lr)

        opt_deform = torch.optim.Adam(
            deform.parameters(),
            lr=config.lr,
        )

        # Save matcher tau so Phase 2 starts fresh
        tau_before_p1 = matcher.tau

        # Best checkpoint tracking (recon loss for INR)
        best_recon = float("inf")
        best_pretrain_states: Dict[str, dict] = {}
        patience_counter = 0
        patience_limit = 300  # stop if no improvement for 300 epochs

        for ep in range(jcfg.inr_pretrain_epochs):
            warmup_inr = max(jcfg.inr_pretrain_epochs // 3, 1)

            # Coarse-to-fine alpha for INR PE
            alpha_inr = n_freqs_inr * (ep / warmup_inr) if ep < warmup_inr else float(n_freqs_inr)
            alpha_inr_t = torch.tensor(alpha_inr, device=device)

            # Coarse-to-fine alpha for DeformNet PE
            alpha_def = n_freqs_deform * (ep / warmup_inr) if ep < warmup_inr else float(n_freqs_deform)
            alpha_def_t = torch.tensor(alpha_def, device=device)

            # Sample batches from both slices
            idx1 = torch.randint(0, N1, (batch_size,), device=device)
            idx2_inr = torch.randint(0, N2, (batch_size,), device=device)

            # --- (A) INR recon: dual INR + shared decoder on both slices ---
            opt_inr.zero_grad(set_to_none=True)

            # Slice 1 recon
            emb1_batch = expr_inr_s1(x1[idx1], alpha_inr_t)
            L_recon1 = recon_loss_from_emb(emb1_batch, decoder, hvg1[idx1])

            if dual_inr:
                # Slice 2 recon (using original x2 coords — before deformation)
                emb2_batch = expr_inr_s2(x2[idx2_inr], alpha_inr_t)
                L_recon2 = recon_loss_from_emb(emb2_batch, decoder, hvg2[idx2_inr])
                L_recon = (L_recon1 + L_recon2) / 2.0
            else:
                L_recon = L_recon1

            L_recon.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(expr_inr_s1.parameters(), config.grad_clip)
                if dual_inr:
                    nn.utils.clip_grad_norm_(expr_inr_s2.parameters(), config.grad_clip)
                nn.utils.clip_grad_norm_(decoder.parameters(), config.grad_clip)
            opt_inr.step()

            # --- (B) DeformNet matching (independent, PCA features) ---
            opt_deform.zero_grad(set_to_none=True)
            idx2 = torch.randint(0, N2, (batch_size,), device=device)

            x2_def = deform(x2[idx2], alpha_def_t)
            L_match, _, _ = matching_loss_joint(
                x2_def, x1, emb2_pca_norm[idx2], emb1_pca_norm,
                matcher, config.topk, config.weight_rev,
            )
            L_match.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(deform.parameters(), config.grad_clip)
            opt_deform.step()

            # Best checkpoint for INR (skip warmup phase)
            recon_val = L_recon.item()
            match_val = L_match.item()
            if ep >= warmup_inr:
                if recon_val < best_recon:
                    best_recon = recon_val
                    best_pretrain_states = {
                        "expr_inr_s1": {k: v.cpu().clone() for k, v in expr_inr_s1.state_dict().items()},
                        "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                    }
                    if dual_inr:
                        best_pretrain_states["expr_inr_s2"] = {
                            k: v.cpu().clone() for k, v in expr_inr_s2.state_dict().items()
                        }
                    patience_counter = 0
                else:
                    patience_counter += 1

            if ep % config.print_every == 0 or ep == jcfg.inr_pretrain_epochs - 1:
                ckpt_marker = " *" if recon_val <= best_recon else ""
                extra = ""
                if dual_inr:
                    extra = f" | r1={L_recon1.item():.5f} r2={L_recon2.item():.5f}"
                print(f"    ep={ep:03d} | recon={recon_val:.5f}{extra} | match={match_val:.5f} | τ={matcher.tau:.4f}{ckpt_marker}")

            history.append({
                "epoch": -(jcfg.inr_pretrain_epochs - ep),
                "phase": 1, "recon": recon_val, "match": match_val,
                "loss": recon_val + match_val, "tau": matcher.tau,
                "lr": jcfg.inr_pretrain_lr,
            })

            # Early stopping (based on INR recon)
            if ep >= warmup_inr and patience_counter >= patience_limit:
                print(f"    Early stopping at ep={ep} (no improvement for {patience_limit} epochs)")
                break

        # Restore best INR checkpoint
        if best_pretrain_states:
            expr_inr_s1.load_state_dict({k: v.to(device) for k, v in best_pretrain_states["expr_inr_s1"].items()})
            decoder.load_state_dict({k: v.to(device) for k, v in best_pretrain_states["decoder"].items()})
            if dual_inr and "expr_inr_s2" in best_pretrain_states:
                expr_inr_s2.load_state_dict({k: v.to(device) for k, v in best_pretrain_states["expr_inr_s2"].items()})
            print(f"  Phase 1 done. Best recon: {best_recon:.5f} (restored)")
        else:
            print(f"  Phase 1 done. Final recon: {L_recon.item():.5f}")

        # Reset matcher tau so Phase 2 starts fresh
        matcher.tau = tau_before_p1

    # ==================================================================
    # Phase 2: Joint Alignment (uses ExprINR_s1 only — canonical space)
    # ==================================================================
    # After deformation, x2_def lives in s1's canonical space, so we use
    # ExprINR_s1 for BOTH sides of the P-matrix and recon loss.
    # ExprINR_s2 is NOT used in Phase 2 (it was only needed for Phase 1
    # pretraining on original s2 coordinates).
    print(f"\n  Phase 2: Joint Alignment ({config.epochs} epochs)"
          f" — ExprINR_s1 only (canonical space)")

    # Optionally freeze ExprINR_s1 + decoder in Phase 2
    freeze_inr = use_inr and jcfg.freeze_inr_phase2
    if freeze_inr:
        for p in expr_inr_s1.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False
        expr_inr_s1.eval()
        decoder.eval()
        print("  [ExprINR_s1 + decoder frozen in Phase 2]")

    # Freeze ExprINR_s2 entirely — not used in Phase 2
    if use_inr and dual_inr:
        for p in expr_inr_s2.parameters():
            p.requires_grad = False
        expr_inr_s2.eval()

    # Optimizer: DeformNet + (optionally ExprINR_s1 + decoder if not frozen)
    all_params = [{"params": deform.parameters(), "lr": config.lr}]
    if use_inr and not freeze_inr:
        all_params.append({"params": expr_inr_s1.parameters(), "lr": config.lr})
        all_params.append({"params": decoder.parameters(), "lr": config.lr})

    optimizer = torch.optim.Adam(all_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    best_match = float("inf")
    best_states: Dict[str, dict] = {}
    prev_lr = config.lr

    for ep in range(config.epochs):
        # Coarse-to-fine alpha for DeformNet PE
        warmup = int(config.epochs * config.warmup_fraction)
        alpha = n_freqs_deform * (ep / warmup) if ep < warmup else float(n_freqs_deform)
        alpha_t = torch.tensor(alpha, device=device)

        # INR alpha (fully open in Phase 2 — INR already pretrained)
        if use_inr:
            alpha_inr_t = torch.tensor(float(expr_inr_s1.n_freqs), device=device)

        # Reference embeddings for P-matrix (recomputed each epoch)
        # ExprINR_s1 encodes the reference slice
        if use_inr:
            with torch.no_grad():
                emb1_epoch = F.normalize(expr_inr_s1(x1, alpha_inr_t), dim=1)
        else:
            emb1_epoch = emb1_pca_norm

        perm = torch.randperm(N2, device=device)
        ep_match = 0.0
        ep_recon = 0.0
        ep_jac = 0.0
        ep_uniq = 0.0
        ep_total = 0.0
        n_batch = 0

        for bi in range(0, N2, batch_size):
            idx = perm[bi: min(bi + batch_size, N2)]
            bs = idx.shape[0]

            optimizer.zero_grad(set_to_none=True)

            # --- DeformNet forward (spatial only) ---
            x2_batch = x2[idx].requires_grad_(True)
            x2_def = deform(x2_batch, alpha_t)

            # --- P-matrix features ---
            if use_inr:
                # x2_def is in s1's canonical space → use ExprINR_s1
                # Detach: embedding is a fixed soft mask for P-matrix, not a gradient path
                emb2_for_match = F.normalize(expr_inr_s1(x2_def.detach(), alpha_inr_t), dim=1)
            else:
                emb2_for_match = emb2_pca_norm[idx]

            # --- 1. Matching loss ---
            L_match, fwd_idx, fwd_w = matching_loss_joint(
                x2_def, x1, emb2_for_match, emb1_epoch,
                matcher, config.topk, config.weight_rev,
            )

            # --- 2. Assignment uniqueness loss ---
            L_uniq = torch.tensor(0.0, device=device)
            if jcfg.lam_uniqueness > 0:
                L_uniq = assignment_uniqueness_loss(x2_def, x1, fwd_idx, fwd_w)

            # --- 3. Jacobian reg (spatial head only) ---
            L_jac = torch.tensor(0.0, device=device)
            if jcfg.lam_jacobian > 0:
                L_jac = jacobian_reg(deform, x2_batch, alpha_t)

            # --- 4. Deformation magnitude penalty ---
            L_deform_mag = torch.tensor(0.0, device=device)
            if jcfg.lam_deform_mag > 0:
                displacement = x2_def - x2[idx]
                L_deform_mag = displacement.pow(2).sum(dim=1).mean()

            # --- 5. Reconstruction loss (ExprINR_s1 + decoder, reduced weight) ---
            #     Both sides use ExprINR_s1 (canonical space)
            #     Skip if INR is frozen — no trainable params benefit from it
            L_recon = torch.tensor(0.0, device=device)
            if use_inr and jcfg.lam_recon_phase2 > 0 and not freeze_inr:
                # Reference recon: ExprINR_s1(x1) → decoder(emb) → expr1_hat
                ref_idx = torch.randint(0, N1, (bs,), device=device)
                emb1_recon = expr_inr_s1(x1[ref_idx], alpha_inr_t)
                L_recon1 = recon_loss_from_emb(emb1_recon, decoder, hvg1[ref_idx])

                # Source recon: ExprINR_s1(x2_def) → decoder(emb) → expr2_hat
                # x2_def is in canonical space → ExprINR_s1 applies
                # As deformation improves, x2_def → true s1 coords → recon improves
                emb2_recon = expr_inr_s1(x2_def.detach(), alpha_inr_t)
                L_recon2 = recon_loss_from_emb(emb2_recon, decoder, hvg2[idx])

                L_recon = (L_recon1 + L_recon2) / 2.0

            # --- Total loss ---
            loss = (jcfg.lam_match * L_match
                    + jcfg.lam_uniqueness * L_uniq
                    + jcfg.lam_jacobian * L_jac
                    + jcfg.lam_deform_mag * L_deform_mag
                    + jcfg.lam_recon_phase2 * L_recon)

            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(deform.parameters(), config.grad_clip)
                if use_inr and not freeze_inr:
                    nn.utils.clip_grad_norm_(expr_inr_s1.parameters(), config.grad_clip)
                    nn.utils.clip_grad_norm_(decoder.parameters(), config.grad_clip)
            optimizer.step()

            ep_match += L_match.item()
            ep_recon += L_recon.item()
            ep_jac += L_jac.item()
            ep_uniq += L_uniq.item()
            ep_total += loss.item()
            n_batch += 1

        # --- Full-coverage reverse loss (DeformNet only) ---
        if config.full_reverse_interval > 0 and ep % config.full_reverse_interval == 0:
            optimizer.zero_grad(set_to_none=True)
            x2_all_in = x2.requires_grad_(True)
            x2_def_all = deform(x2_all_in, alpha_t)

            if use_inr:
                # x2_def_all in canonical space → ExprINR_s1
                emb2_fc = F.normalize(expr_inr_s1(x2_def_all.detach(), alpha_inr_t), dim=1)
            else:
                emb2_fc = emb2_pca_norm

            K_rev = min(config.topk, N2)
            idx_rev, w_rev = compute_P_matrix(
                x1, x2_def_all, emb1_epoch, emb2_fc, matcher, config.topk, update_tau=False,
            )
            x2_nbrs = x2_def_all[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)
            target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_nbrs)
            loss_rev_full = config.weight_rev * (x1 - target_rev).pow(2).sum(dim=1).mean()
            loss_rev_full.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(deform.parameters(), config.grad_clip)
            optimizer.step()

        # --- Best checkpoint (match + uniqueness penalty) ---
        avg_match = ep_match / max(n_batch, 1)
        avg_uniq = ep_uniq / max(n_batch, 1)
        avg_score = avg_match + jcfg.lam_uniqueness * avg_uniq
        if avg_score < best_match:
            best_match = avg_score
            best_states = {
                "deform": {k: v.cpu().clone() for k, v in deform.state_dict().items()},
            }
            if use_inr and not freeze_inr:
                best_states["expr_inr_s1"] = {k: v.cpu().clone() for k, v in expr_inr_s1.state_dict().items()}
                best_states["decoder"] = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

        # --- Scheduler ---
        scheduler.step(avg_score)
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": ep, "loss": ep_total / max(n_batch, 1),
            "match": avg_match, "tau": matcher.tau, "lr": cur_lr,
            "recon": ep_recon / max(n_batch, 1),
            "jac": ep_jac / max(n_batch, 1),
            "phase": 2,
        }
        history.append(record)

        if ep % config.print_every == 0 or ep == config.epochs - 1:
            parts = [f"ep={ep:03d} [P2]"]
            parts.append(f"lr={cur_lr:.1e}")
            parts.append(f"\u03c4={matcher.tau:.4f}")
            parts.append(f"match={avg_match:.5f}")
            if jcfg.lam_jacobian > 0:
                parts.append(f"jac={ep_jac / max(n_batch, 1):.5f}")
            if use_inr:
                parts.append(f"recon={ep_recon / max(n_batch, 1):.5f}")
            print(f"  {' | '.join(parts)}")

        # Log LR changes
        if cur_lr < prev_lr:
            print(f"  >> LR {prev_lr:.1e} -> {cur_lr:.1e}")
            prev_lr = cur_lr

    # --- Restore best checkpoint ---
    if "deform" in best_states:
        deform.load_state_dict({k: v.to(device) for k, v in best_states["deform"].items()})
    if use_inr:
        if "expr_inr_s1" in best_states:
            expr_inr_s1.load_state_dict({k: v.to(device) for k, v in best_states["expr_inr_s1"].items()})
        if "decoder" in best_states:
            decoder.load_state_dict({k: v.to(device) for k, v in best_states["decoder"].items()})

    elapsed = time.time() - start
    return TrainResult(
        deform=deform,
        matcher=matcher,
        best_match_loss=best_match,
        training_time=elapsed,
        history=history,
        expr_inr=expr_inr_s1 if use_inr else None,
        expr_inr_s1=expr_inr_s1 if use_inr else None,
        expr_inr_s2=expr_inr_s2 if use_inr else None,
        decoder=decoder if use_inr else None,
    )


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def apply_model(
    deform: DeformationNet,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply the trained deformation network (coordinates only)."""
    alpha = torch.tensor(float(deform.n_freqs), device=x.device)
    return deform(x, alpha)


@torch.no_grad()
def apply_model_with_inr(
    deform: DeformationNet,
    inr: ExprINR,
    x: torch.Tensor,
) -> tuple:
    """Apply deformation and get INR embedding, returning ``(x_def, embedding)``.

    Args:
        deform: Trained DeformationNet.
        inr: Trained ExprINR (typically INR1 — target INR).
        x: ``(N, 2)`` coordinates to deform and embed.

    Returns:
        ``(x_def, emb)`` — deformed coordinates and INR embeddings.
    """
    alpha_deform = torch.tensor(float(deform.n_freqs), device=x.device)
    alpha_inr = torch.tensor(float(inr.n_freqs), device=x.device)
    x_def = deform(x, alpha_deform)
    emb = inr(x_def, alpha_inr)
    return x_def, emb

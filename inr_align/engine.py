"""Two-phase training loop and inference utilities.

Phase 1 — INR Pretrain (``jcfg.inr_pretrain_epochs`` iterations):
    Two independent ExprINRs learn spatial expression fields:
    coords1 → INR1 → emb → SharedDecoder(emb, batch=0) → expr1_hat
    coords2 → INR2 → emb → SharedDecoder(emb, batch=1) → expr2_hat

Phase 2 — Joint Alignment (``config.epochs`` iterations):
    INR2 frozen.  INR1 + decoder continue training (recon at reduced weight).
    P-matrix uses INR1 for both sides (target coordinate space).
    DeformNet trained: matching + Jacobian + uniqueness.
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

    model: DeformationNet
    matcher: UnifiedCostMatcher
    best_match_loss: float
    training_time: float
    history: List[Dict[str, float]] = field(default_factory=list)
    inr1: Optional[Any] = None
    inr2: Optional[Any] = None
    decoder: Optional[Any] = None


# ============================================================================
# Two-phase training loop
# ============================================================================


def train(
    model: DeformationNet,
    matcher: UnifiedCostMatcher,
    x1: torch.Tensor,
    emb1_pca: torch.Tensor,
    x2: torch.Tensor,
    emb2_pca: torch.Tensor,
    config: Optional[TrainConfig] = None,
    jcfg: Optional[JointConfig] = None,
    # --- INR components (None = PCA-only alignment, no embedding) ---
    inr1: Optional[ExprINR] = None,
    inr2: Optional[ExprINR] = None,
    decoder: Optional[ExprDecoder] = None,
    hvg1: Optional[torch.Tensor] = None,
    hvg2: Optional[torch.Tensor] = None,
    nbr_idx2: Optional[torch.Tensor] = None,
) -> TrainResult:
    """Train the deformation network with two-phase INR strategy.

    Phase 1 — INR Pretrain (``jcfg.inr_pretrain_epochs``):
        - Two independent INRs learn spatial expression fields.
        - coords1 → INR1 → emb → Decoder(emb, batch=0) → expr1_hat
        - coords2 → INR2 → emb → Decoder(emb, batch=1) → expr2_hat
        - Loss: SUICA-style (masked MSE + L1 + Dice).
        - Alpha windowing: coarse-to-fine PE.

    Phase 2 — Joint Alignment (``config.epochs``):
        - INR2 frozen (no longer needed).
        - INR1 used for both sides (coords in same space after rigid + deform).
        - P-matrix: spatial_dist + feat_dist(INR1(x2_def), INR1(x1)).
        - DeformNet + INR1 + decoder trained jointly.
        - Recon loss at reduced weight (``lam_recon_phase2``) to preserve
          biological information in embeddings.

    Args:
        model: ``DeformationNet`` (spatial-only, on device).
        matcher: ``UnifiedCostMatcher`` instance.
        x1: ``(N1, 2)`` reference coords (normalized, on device).
        emb1_pca: ``(N1, D)`` reference PCA embeddings (on device).
        x2: ``(N2, 2)`` source coords after rigid alignment (on device).
        emb2_pca: ``(N2, D)`` source PCA embeddings (on device).
        config: Training hyper-parameters.
        jcfg: Joint config (INR/decoder params + loss weights).
        inr1: ``ExprINR`` for reference slice (optional, None = PCA-only).
        inr2: ``ExprINR`` for source slice (optional, None = PCA-only).
        decoder: ``ExprDecoder`` (optional).
        hvg1: ``(N1, G)`` reference HVG expression (decoder recon target).
        hvg2: ``(N2, G)`` source HVG expression (decoder recon target).
        nbr_idx2: ``(N2, k)`` pre-computed kNN for spatial smooth (unused).

    Returns:
        :class:`TrainResult` with best-checkpoint models restored.
    """
    if config is None:
        config = TrainConfig()
    if jcfg is None:
        jcfg = JointConfig()

    device = x1.device
    N1, N2 = x1.shape[0], x2.shape[0]
    n_freqs_deform = model.n_freqs
    batch_size = min(config.batch_size, max(N1, N2))

    # Detect INR mode: need both INRs + decoder + HVG data
    use_inr = (inr1 is not None and inr2 is not None
               and decoder is not None
               and hvg1 is not None and hvg2 is not None)

    # L2-normalize PCA features for P-matrix (fallback when no INR)
    emb1_pca_norm = F.normalize(emb1_pca, dim=1)
    emb2_pca_norm = F.normalize(emb2_pca, dim=1)

    # Slice-ID tensors for decoder
    if use_inr:
        sid0 = torch.zeros(batch_size, dtype=torch.long, device=device)
        sid1 = torch.ones(batch_size, dtype=torch.long, device=device)

    history: List[Dict[str, float]] = []
    start = time.time()

    # ==================================================================
    # Phase 1: INR Pretrain
    # ==================================================================
    if use_inr and jcfg.inr_pretrain_epochs > 0:
        print(f"\n  Phase 1: INR Pretrain ({jcfg.inr_pretrain_epochs} epochs)")
        n_freqs_inr = inr1.n_freqs

        opt_pretrain = torch.optim.Adam(
            list(inr1.parameters()) + list(inr2.parameters()) + list(decoder.parameters()),
            lr=jcfg.inr_pretrain_lr,
        )

        for ep in range(jcfg.inr_pretrain_epochs):
            # Coarse-to-fine alpha for INR PE
            warmup_inr = max(jcfg.inr_pretrain_epochs // 3, 1)
            alpha_inr = n_freqs_inr * (ep / warmup_inr) if ep < warmup_inr else float(n_freqs_inr)
            alpha_t = torch.tensor(alpha_inr, device=device)

            opt_pretrain.zero_grad(set_to_none=True)

            # Sample batch from each slice
            idx1 = torch.randint(0, N1, (batch_size,), device=device)
            idx2 = torch.randint(0, N2, (batch_size,), device=device)

            # Slice 1: INR1 → decoder(batch=0)
            emb1_batch = inr1(x1[idx1], alpha_t)
            sid0_b = torch.zeros(idx1.shape[0], dtype=torch.long, device=device)
            L_recon1 = recon_loss_from_emb(emb1_batch, decoder, sid0_b, hvg1[idx1])

            # Slice 2: INR2 → decoder(batch=1)
            emb2_batch = inr2(x2[idx2], alpha_t)
            sid1_b = torch.ones(idx2.shape[0], dtype=torch.long, device=device)
            L_recon2 = recon_loss_from_emb(emb2_batch, decoder, sid1_b, hvg2[idx2])

            loss = L_recon1 + L_recon2
            loss.backward()

            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(inr1.parameters(), config.grad_clip)
                nn.utils.clip_grad_norm_(inr2.parameters(), config.grad_clip)
                nn.utils.clip_grad_norm_(decoder.parameters(), config.grad_clip)

            opt_pretrain.step()

            if ep % config.print_every == 0 or ep == jcfg.inr_pretrain_epochs - 1:
                print(f"    ep={ep:03d} | recon1={L_recon1.item():.5f} | recon2={L_recon2.item():.5f} | total={loss.item():.5f}")

            history.append({
                "epoch": -(jcfg.inr_pretrain_epochs - ep),  # negative = pretrain
                "phase": 1, "recon": loss.item(),
                "loss": loss.item(), "match": 0.0,
                "tau": matcher.tau, "lr": jcfg.inr_pretrain_lr,
            })

        print(f"  Phase 1 done. Final recon: {loss.item():.5f}")

    # ==================================================================
    # Phase 2: Joint Alignment
    # ==================================================================
    print(f"\n  Phase 2: Joint Alignment ({config.epochs} epochs)")

    # Freeze INR2 (no longer needed)
    if use_inr:
        for p in inr2.parameters():
            p.requires_grad = False
        inr2.eval()

    # Optionally freeze INR1 + decoder in Phase 2 (preserve embedding quality)
    freeze_inr = use_inr and jcfg.freeze_inr_phase2
    if freeze_inr:
        for p in inr1.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False
        inr1.eval()
        decoder.eval()
        print("  [INR1 + decoder frozen in Phase 2]")

    # Optimizer: DeformNet + (optionally INR1 + decoder if not frozen)
    all_params = [{"params": model.parameters(), "lr": config.lr}]
    if use_inr and not freeze_inr:
        all_params.append({"params": inr1.parameters(), "lr": config.lr})
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
            alpha_inr_t = torch.tensor(float(inr1.n_freqs), device=device)

        # Reference embeddings for P-matrix (recomputed each epoch)
        if use_inr:
            with torch.no_grad():
                emb1_epoch = F.normalize(inr1(x1, alpha_inr_t), dim=1)
        else:
            emb1_epoch = emb1_pca_norm

        perm = torch.randperm(N2, device=device)
        ep_match = 0.0
        ep_recon = 0.0
        ep_uniq = 0.0
        ep_jac = 0.0
        ep_dmag = 0.0
        ep_total = 0.0
        n_batch = 0

        for bi in range(0, N2, batch_size):
            idx = perm[bi: min(bi + batch_size, N2)]
            bs = idx.shape[0]

            optimizer.zero_grad(set_to_none=True)

            # --- DeformNet forward (spatial only) ---
            x2_batch = x2[idx].requires_grad_(True)
            x2_def = model(x2_batch, alpha_t)

            # --- P-matrix features ---
            if use_inr:
                # Use INR1 for both sides (coords in same space)
                # Detach from deformnet for INR embedding computation
                emb2_for_match = F.normalize(inr1(x2_def.detach(), alpha_inr_t), dim=1)
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
                L_jac = jacobian_reg(model, x2_batch, alpha_t)

            # --- 3b. Deformation magnitude penalty ---
            L_deform_mag = torch.tensor(0.0, device=device)
            if jcfg.lam_deform_mag > 0:
                displacement = x2_def - x2[idx]
                L_deform_mag = displacement.pow(2).sum(dim=1).mean()

            # --- 4. Reconstruction loss (INR1 + decoder, reduced weight) ---
            #     Skip if INR1 is frozen — no trainable params benefit from it
            L_recon = torch.tensor(0.0, device=device)
            if use_inr and jcfg.lam_recon_phase2 > 0 and not freeze_inr:
                # Target slice recon: INR1(x1) → decoder(emb, batch=0)
                ref_idx = torch.randint(0, N1, (bs,), device=device)
                emb1_recon = inr1(x1[ref_idx], alpha_inr_t)
                sid0_b = torch.zeros(ref_idx.shape[0], dtype=torch.long, device=device)
                L_recon1 = recon_loss_from_emb(emb1_recon, decoder, sid0_b, hvg1[ref_idx])

                # Source slice recon: INR1(x2_def) → decoder(emb, batch=1)
                emb2_recon = inr1(x2_def.detach(), alpha_inr_t)
                sid1_b = torch.ones(bs, dtype=torch.long, device=device)
                L_recon2 = recon_loss_from_emb(emb2_recon, decoder, sid1_b, hvg2[idx])

                L_recon = (L_recon1 + L_recon2) / 2.0

            # --- Total loss ---
            loss = (jcfg.lam_match * L_match
                    + jcfg.lam_uniqueness * L_uniq
                    + jcfg.lam_jacobian * L_jac
                    + jcfg.lam_deform_mag * L_deform_mag
                    + jcfg.lam_recon_phase2 * L_recon)

            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if use_inr and not freeze_inr:
                    nn.utils.clip_grad_norm_(inr1.parameters(), config.grad_clip)
                    nn.utils.clip_grad_norm_(decoder.parameters(), config.grad_clip)
            optimizer.step()

            ep_match += L_match.item()
            ep_recon += L_recon.item()
            ep_uniq += L_uniq.item()
            ep_jac += L_jac.item()
            ep_dmag += L_deform_mag.item()
            ep_total += loss.item()
            n_batch += 1

        # --- Full-coverage reverse loss (DeformNet only) ---
        if config.full_reverse_interval > 0 and ep % config.full_reverse_interval == 0:
            optimizer.zero_grad(set_to_none=True)
            x2_all_in = x2.requires_grad_(True)
            x2_def_all = model(x2_all_in, alpha_t)

            if use_inr:
                emb2_fc = F.normalize(inr1(x2_def_all.detach(), alpha_inr_t), dim=1)
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
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # --- Best checkpoint ---
        avg_match = ep_match / max(n_batch, 1)
        if avg_match < best_match:
            best_match = avg_match
            best_states = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            }
            if use_inr and not freeze_inr:
                best_states["inr1"] = {k: v.cpu().clone() for k, v in inr1.state_dict().items()}
                best_states["decoder"] = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

        # --- Scheduler ---
        scheduler.step(avg_match)
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": ep, "loss": ep_total / max(n_batch, 1),
            "match": avg_match, "tau": matcher.tau, "lr": cur_lr,
            "recon": ep_recon / max(n_batch, 1),
            "uniq": ep_uniq / max(n_batch, 1),
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
            if jcfg.lam_deform_mag > 0:
                parts.append(f"dmag={ep_dmag / max(n_batch, 1):.5f}")
            if use_inr:
                parts.append(f"recon={ep_recon / max(n_batch, 1):.5f}")
            if jcfg.lam_uniqueness > 0:
                parts.append(f"uniq={ep_uniq / max(n_batch, 1):.6f}")
            print(f"  {' | '.join(parts)}")

        # Log LR changes
        if cur_lr < prev_lr:
            print(f"  >> LR {prev_lr:.1e} -> {cur_lr:.1e}")
            prev_lr = cur_lr

    # --- Restore best checkpoint ---
    if "model" in best_states:
        model.load_state_dict({k: v.to(device) for k, v in best_states["model"].items()})
    if use_inr:
        if "inr1" in best_states:
            inr1.load_state_dict({k: v.to(device) for k, v in best_states["inr1"].items()})
        if "decoder" in best_states:
            decoder.load_state_dict({k: v.to(device) for k, v in best_states["decoder"].items()})

    elapsed = time.time() - start
    return TrainResult(
        model=model,
        matcher=matcher,
        best_match_loss=best_match,
        training_time=elapsed,
        history=history,
        inr1=inr1 if use_inr else None,
        inr2=inr2 if use_inr else None,
        decoder=decoder if use_inr else None,
    )


# ============================================================================
# Inference
# ============================================================================


@torch.no_grad()
def apply_model(
    model: DeformationNet,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply the trained deformation network (coordinates only)."""
    alpha = torch.tensor(float(model.n_freqs), device=x.device)
    return model(x, alpha)


@torch.no_grad()
def apply_model_with_inr(
    model: DeformationNet,
    inr: ExprINR,
    x: torch.Tensor,
) -> tuple:
    """Apply deformation and get INR embedding, returning ``(x_def, embedding)``.

    Args:
        model: Trained DeformationNet.
        inr: Trained ExprINR (typically INR1 — target INR).
        x: ``(N, 2)`` coordinates to deform and embed.

    Returns:
        ``(x_def, emb)`` — deformed coordinates and INR embeddings.
    """
    alpha_deform = torch.tensor(float(model.n_freqs), device=x.device)
    alpha_inr = torch.tensor(float(inr.n_freqs), device=x.device)
    x_def = model(x, alpha_deform)
    emb = inr(x_def, alpha_inr)
    return x_def, emb

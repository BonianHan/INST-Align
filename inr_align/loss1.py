"""Decoupled joint optimization — losses and training loop.

Loss components (all differentiable through encoder):

1. **Matching loss** — P matrix couples WarpNet + Encoder.
   ``P = softmax(-(spatial_dist(x2_def, x1) + λ·feat_dist(enc2, enc1)) / τ)``
   Gradients flow to WarpNet (via x2_def) AND Encoder (via enc2).

2. **Reconstruction loss** — Encoder + Decoder.
   ``MSE(decoder(encoder(expr), slice_id), expr)``
   Decoder receives slice_id → batch effects stay in decoder, not embedding.

3. **Adversarial loss** — Encoder + Discriminator (via GRL).
   Discriminator predicts slice from embedding; encoder fools it.

4. **Spatial smooth loss** — mild kNN smoothing on embeddings (denoise).

5. **Jacobian reg** — re-used from existing ``loss.py``.

Backward-compatible: uses existing ``compute_P_matrix``,
``jacobian_reg``, ``UnifiedCostMatcher``.

Usage::

    from loss1 import train_joint
    result = train_joint(
        warpnet, matcher, encoder, decoder, disc,
        x1, x2, expr1, expr2,
        train_config, joint_config,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from inr_align.config import TrainConfig
from inr_align.loss import assignment_uniqueness_loss, compute_P_matrix, jacobian_reg
from inr_align.model import DeformationNet, UnifiedCostMatcher
from inr_align.train import TrainResult, apply_model

from inr_align.model1 import (
    ExprDecoder,
    ExprEncoder,
    JointConfig,
    SliceDiscriminator,
    build_joint_models,
    build_knn_graph,
)


# ============================================================================
# Individual loss functions
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
    """Bidirectional matching loss using encoder embeddings.

    Identical to existing ``compute_bidirectional_loss`` but returns
    (loss, topk_idx, weights) for downstream use.

    Encoder gradients flow through emb2 (and emb1 if not detached).
    WarpNet gradients flow through x2_def.
    """
    N2, N1 = x2_def.shape[0], x1.shape[0]
    K = min(topk, N1)

    # L2-normalise for cosine distance in P matrix
    emb1n = F.normalize(emb1, dim=1)
    emb2n = F.normalize(emb2, dim=1)

    # Forward: x2_def → x1
    idx_fwd, w_fwd = compute_P_matrix(
        x2_def, x1, emb2n, emb1n, matcher, topk, update_tau=True,
    )
    x1_nbrs = x1[idx_fwd.reshape(-1)].reshape(N2, K, 2)
    target_fwd = torch.einsum("bk,bkd->bd", w_fwd, x1_nbrs)
    loss_fwd = (x2_def - target_fwd).pow(2).sum(dim=1).mean()

    # Reverse: x1 → x2_def
    K_rev = min(topk, N2)
    idx_rev, w_rev = compute_P_matrix(
        x1, x2_def, emb1n, emb2n, matcher, topk, update_tau=False,
    )
    x2_nbrs = x2_def[idx_rev.reshape(-1)].reshape(N1, K_rev, 2)
    target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_nbrs)
    loss_rev = (x1 - target_rev).pow(2).sum(dim=1).mean()

    loss = loss_fwd + weight_rev * loss_rev
    return loss, idx_fwd, w_fwd


def recon_loss(
    encoder: ExprEncoder,
    decoder: ExprDecoder,
    expr_batch: torch.Tensor,
    slice_ids: torch.Tensor,
) -> torch.Tensor:
    """Self-reconstruction: expr → encoder → decoder(·, slice_id) → expr̂.

    Args:
        encoder: Expression encoder.
        decoder: Expression decoder.
        expr_batch: ``(N, D)`` input expression (PCA).
        slice_ids: ``(N,)`` integer slice indices.

    Returns:
        Scalar MSE reconstruction loss.
    """
    emb = encoder(expr_batch)
    recon = decoder(emb, slice_ids)
    return F.mse_loss(recon, expr_batch)


def adversarial_loss(
    discriminator: SliceDiscriminator,
    emb: torch.Tensor,
    slice_ids: torch.Tensor,
) -> torch.Tensor:
    """Adversarial batch correction loss.

    With GRL built into the discriminator, a single backward pass
    sends *reversed* gradients to the encoder (fool the discriminator)
    and *normal* gradients to the discriminator (classify better).

    Args:
        discriminator: Discriminator with GRL.
        emb: ``(N, emb_dim)`` embeddings from encoder.
        slice_ids: ``(N,)`` ground-truth slice indices.

    Returns:
        Scalar cross-entropy loss.
    """
    logits = discriminator(emb)  # GRL applied internally
    return F.cross_entropy(logits, slice_ids)


def spatial_smooth_loss(
    emb: torch.Tensor,
    nbr_idx: torch.Tensor,
    weight: float = 0.01,
) -> torch.Tensor:
    """Mild spatial smoothing: neighbours should have similar embeddings.

    Weight should be small (0.001–0.05) — just denoise, not Splane-style
    aggressive averaging.

    Args:
        emb: ``(N, D)`` embeddings.
        nbr_idx: ``(N, k)`` integer neighbour indices.

    Returns:
        Scalar MSE-based smoothness loss.
    """
    emb_nbr = emb[nbr_idx]                       # (N, k, D)
    emb_center = emb.unsqueeze(1)                  # (N, 1, D)
    return weight * (emb_center - emb_nbr).pow(2).mean()


# ============================================================================
# GRL lambda schedule (ramp 0 → max during warmup)
# ============================================================================


def grl_lambda_schedule(
    epoch: int,
    total_epochs: int,
    max_lambda: float = 1.0,
    warmup_frac: float = 0.3,
) -> float:
    """Ramp GRL strength from 0 at start of warmup to max_lambda.

    During warmup, GRL is off (encoder learns freely).
    After warmup, ramps linearly to max_lambda.
    """
    warmup = int(total_epochs * warmup_frac)
    if epoch < warmup:
        return 0.0
    progress = (epoch - warmup) / max(total_epochs - warmup, 1)
    return max_lambda * min(progress * 2, 1.0)  # reach max at 50% post-warmup


# ============================================================================
# Training loop — joint optimization
# ============================================================================


def train_joint(
    warpnet: DeformationNet,
    matcher: UnifiedCostMatcher,
    encoder: ExprEncoder,
    decoder: ExprDecoder,
    discriminator: SliceDiscriminator,
    x1: torch.Tensor,
    x2: torch.Tensor,
    expr1: torch.Tensor,
    expr2: torch.Tensor,
    config: TrainConfig,
    jcfg: JointConfig,
    nbr_idx2: Optional[torch.Tensor] = None,
) -> TrainResult:
    """Joint training: WarpNet + Encoder + Decoder + Discriminator.

    Drop-in replacement for ``inr_align.train.train()`` with the
    decoupled joint architecture.

    Args:
        warpnet: ``DeformationNet`` for coordinate warping.
        matcher: ``UnifiedCostMatcher`` for adaptive temperature.
        encoder: ``ExprEncoder``  (expression → embedding).
        decoder: ``ExprDecoder``  (embedding + slice_id → expression).
        discriminator: ``SliceDiscriminator`` (embedding → slice logit).
        x1: ``(N1, 2)`` reference coordinates (normalised, on device).
        x2: ``(N2, 2)`` source coordinates after rigid align (on device).
        expr1: ``(N1, D)`` reference PCA features (on device).
        expr2: ``(N2, D)`` source PCA features (on device).
        config: Training hyper-parameters (epochs, lr, etc.).
        jcfg: Joint-optimisation hyper-parameters.
        nbr_idx2: ``(N2, k)`` pre-computed kNN for source slice
            (from ``build_knn_graph``).  ``None`` → skip smooth loss.

    Returns:
        ``TrainResult`` (same type as ``inr_align.train.train``).
    """
    device = x1.device
    N1, N2 = x1.shape[0], x2.shape[0]
    n_freqs = warpnet.n_freqs
    K = min(config.topk, N1)
    batch_size = min(config.batch_size, N2)

    # ---- Optimiser (single, with GRL) ----
    optimizer = torch.optim.Adam([
        {"params": warpnet.parameters(), "lr": config.lr},
        {"params": encoder.parameters(), "lr": config.lr},
        {"params": decoder.parameters(), "lr": config.lr},
        {"params": discriminator.parameters(), "lr": config.lr},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    # ---- Pre-compute kNN for spatial smoothing ----
    if nbr_idx2 is not None:
        nbr_idx2 = torch.as_tensor(nbr_idx2, dtype=torch.long, device=device)

    # ---- Slice-ID tensors ----
    sid1 = torch.zeros(N1, dtype=torch.long, device=device)
    sid2 = torch.ones(N2, dtype=torch.long, device=device)

    # ---- Training state ----
    best_match = float("inf")
    best_states = {}
    history: List[Dict[str, float]] = []
    start = time.time()
    prev_lr = config.lr

    for ep in range(config.epochs):
        # Coarse-to-fine alpha (Nerfies schedule)
        warmup = int(config.epochs * config.warmup_fraction)
        alpha = n_freqs * (ep / warmup) if ep < warmup else float(n_freqs)
        alpha_t = torch.tensor(alpha, device=device)

        # GRL ramp
        grl_lam = grl_lambda_schedule(
            ep, config.epochs, jcfg.grl_max_lambda, config.warmup_fraction,
        )
        discriminator.set_grl_lambda(grl_lam)

        # Reference embeddings (re-compute per epoch, detach for efficiency)
        with torch.no_grad():
            emb1_epoch = encoder(expr1)
        emb1_epoch = emb1_epoch.detach()

        perm = torch.randperm(N2, device=device)
        ep_match = 0.0
        ep_recon = 0.0
        ep_adv = 0.0
        ep_uniq = 0.0
        ep_total = 0.0
        n_batch = 0

        for bi in range(0, N2, batch_size):
            idx = perm[bi : min(bi + batch_size, N2)]
            bs = idx.shape[0]

            optimizer.zero_grad(set_to_none=True)

            # ---- WarpNet forward ----
            x2_batch = x2[idx].requires_grad_(True)
            x2_def = warpnet(x2_batch, alpha_t)

            # ---- Encoder forward (source batch) ----
            emb2_batch = encoder(expr2[idx])

            # ---- 1. Matching loss (couples WarpNet + Encoder) ----
            L_match, fwd_idx, fwd_w = matching_loss_joint(
                x2_def, x1, emb2_batch, emb1_epoch,
                matcher, config.topk, config.weight_rev,
            )

            # ---- 2. Reconstruction loss (both slices) ----
            # Source batch
            L_recon_src = recon_loss(encoder, decoder, expr2[idx], sid2[idx])
            # Sample from reference (same batch size)
            ref_idx = torch.randint(0, N1, (bs,), device=device)
            L_recon_ref = recon_loss(encoder, decoder, expr1[ref_idx], sid1[ref_idx])
            L_recon = (L_recon_src + L_recon_ref) / 2.0

            # ---- 3. Adversarial loss (after warmup) ----
            L_adv_val = 0.0
            if grl_lam > 0:
                # Reuse source embeddings; sample reference
                emb_ref = encoder(expr1[ref_idx])
                emb_all = torch.cat([emb2_batch, emb_ref], dim=0)
                sid_all = torch.cat([sid2[idx], sid1[ref_idx]], dim=0)
                L_adv = adversarial_loss(discriminator, emb_all, sid_all)
                L_adv_val = L_adv.item()
            else:
                L_adv = torch.tensor(0.0, device=device)

            # ---- 4. Assignment uniqueness loss (anti many-to-one) ----
            L_uniq = torch.tensor(0.0, device=device)
            if jcfg.lam_uniqueness > 0:
                L_uniq = assignment_uniqueness_loss(x2_def, x1, fwd_idx, fwd_w)

            # ---- 5. Spatial smooth loss ----
            # (computed once per epoch, outside batch loop — see below)
            L_smooth = torch.tensor(0.0, device=device)

            # ---- 6. Jacobian regularisation ----
            L_jac = torch.tensor(0.0, device=device)
            if jcfg.lam_jacobian > 0:
                L_jac = jacobian_reg(warpnet, x2_batch, alpha_t)

            # ---- Total loss (per batch) ----
            loss = (
                jcfg.lam_match * L_match
                + jcfg.lam_recon * L_recon
                + jcfg.lam_adv * L_adv
                + jcfg.lam_uniqueness * L_uniq
                + jcfg.lam_jacobian * L_jac
            )

            loss.backward()
            if config.grad_clip > 0:
                for m in [warpnet, encoder, decoder, discriminator]:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), config.grad_clip)
            optimizer.step()

            ep_match += L_match.item()
            ep_recon += L_recon.item()
            ep_adv += L_adv_val
            ep_uniq += L_uniq.item()
            ep_total += loss.item()
            n_batch += 1

        # ---- Spatial smooth loss (once per epoch, not per batch) ----
        ep_smooth = 0.0
        if nbr_idx2 is not None and jcfg.lam_smooth > 0:
            optimizer.zero_grad(set_to_none=True)
            emb2_all = encoder(expr2)
            L_smooth = spatial_smooth_loss(emb2_all, nbr_idx2, weight=jcfg.lam_smooth)
            L_smooth.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.grad_clip)
            optimizer.step()
            ep_smooth = L_smooth.item()

        # ---- Full-coverage reverse (same as existing train.py) ----
        if config.full_reverse_interval > 0 and ep % config.full_reverse_interval == 0:
            optimizer.zero_grad(set_to_none=True)
            x2_all_in = x2.requires_grad_(True)
            x2_def_all = warpnet(x2_all_in, alpha_t)
            emb2_all_fc = encoder(expr2).detach()  # detach encoder here — this is WarpNet-only step
            emb2n = F.normalize(emb2_all_fc, dim=1)
            emb1n = F.normalize(emb1_epoch, dim=1)
            K_rev_full = min(config.topk, N2)
            idx_rev, w_rev = compute_P_matrix(
                x1, x2_def_all, emb1n, emb2n, matcher, config.topk, update_tau=False,
            )
            x2_nbrs = x2_def_all[idx_rev.reshape(-1)].reshape(N1, K_rev_full, 2)
            target_rev = torch.einsum("bk,bkd->bd", w_rev, x2_nbrs)
            loss_rev_full = config.weight_rev * (x1 - target_rev).pow(2).sum(dim=1).mean()
            loss_rev_full.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(warpnet.parameters(), config.grad_clip)
            optimizer.step()

        # ---- Bookkeeping ----
        avg_match = ep_match / n_batch
        if avg_match < best_match:
            best_match = avg_match
            best_states = {
                "warpnet": {k: v.cpu().clone() for k, v in warpnet.state_dict().items()},
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                "discriminator": {k: v.cpu().clone() for k, v in discriminator.state_dict().items()},
            }

        scheduler.step(avg_match)
        cur_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": ep, "loss": ep_total / n_batch, "match": avg_match,
            "recon": ep_recon / n_batch, "adv": ep_adv / n_batch,
            "uniq": ep_uniq / n_batch, "smooth": ep_smooth,
            "tau": matcher.tau, "lr": cur_lr, "grl_lam": grl_lam,
        }
        history.append(record)

        if ep % config.print_every == 0 or ep == config.epochs - 1:
            print(
                f"  ep={ep:03d} | lr={cur_lr:.1e} | τ={matcher.tau:.4f} | "
                f"grl={grl_lam:.2f} | loss={ep_total / n_batch:.5f} | "
                f"match={avg_match:.5f} | recon={ep_recon / n_batch:.5f} | "
                f"adv={ep_adv / n_batch:.5f} | uniq={ep_uniq / n_batch:.6f}"
            )

        if cur_lr < prev_lr:
            print(f"  >> LR {prev_lr:.1e} → {cur_lr:.1e}")
            prev_lr = cur_lr

    # ---- Restore best ----
    if best_states:
        warpnet.load_state_dict({k: v.to(device) for k, v in best_states["warpnet"].items()})
        encoder.load_state_dict({k: v.to(device) for k, v in best_states["encoder"].items()})
        decoder.load_state_dict({k: v.to(device) for k, v in best_states["decoder"].items()})
        discriminator.load_state_dict({k: v.to(device) for k, v in best_states["discriminator"].items()})

    elapsed = time.time() - start
    return TrainResult(
        model=warpnet,
        matcher=matcher,
        best_match_loss=best_match,
        training_time=elapsed,
        history=history,
        expr_field=None,
        gene_decoder=None,
    )


# ============================================================================
# Convenience: run full joint pipeline on a pair (mirrors run_ours)
# ============================================================================


def align_pair_joint(
    slice1,
    slice2,
    config,           # PipelineConfig
    jcfg: JointConfig,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Align a pair with the decoupled joint architecture.

    Drop-in for ``benchmark.run_ours``.

    Returns:
        ``(coords2_rigid_denorm, coords2_final, elapsed)``.
    """
    import scanpy as sc
    import spateo as st

    from inr_align.model import DeformationNet, adaptive_icp, UnifiedCostMatcher
    from inr_align.utils import normalize_coordinates

    # ---- Preprocess ----
    for ad_ in [slice1, slice2]:
        if "counts" not in ad_.layers:
            ad_.layers["counts"] = ad_.X.copy()
        sc.pp.normalize_total(ad_)
        sc.pp.log1p(ad_)
        if "highly_variable" not in ad_.var.columns:
            sc.pp.highly_variable_genes(ad_, n_top_genes=config.n_top_genes)
    st.align.group_pca([slice1, slice2], pca_key=config.pca_key)

    start = time.time()

    coords1 = slice1.obsm[config.spatial_key]
    coords2 = slice2.obsm[config.spatial_key]
    [c1n, c2n], mean, std = normalize_coordinates([coords1, coords2])

    # ---- ICP ----
    R, t, angle, rmse = adaptive_icp(
        c1n, c2n, config.icp, verbose=True,
        emb_A=slice1.obsm[config.pca_key].astype(np.float32),
        emb_B=slice2.obsm[config.pca_key].astype(np.float32),
    )
    c2_rigid = ((R @ c2n.T).T + t).astype(np.float32)

    # ---- GPU tensors ----
    x1 = torch.tensor(c1n.astype(np.float32), device=device)
    x2 = torch.tensor(c2_rigid, device=device)
    expr1 = torch.tensor(slice1.obsm[config.pca_key].astype(np.float32), device=device)
    expr2 = torch.tensor(slice2.obsm[config.pca_key].astype(np.float32), device=device)

    # ---- Set n_input/n_output from data ----
    jcfg.n_input = expr1.shape[1]
    jcfg.n_output = expr1.shape[1]

    # ---- Build models ----
    warpnet = DeformationNet(config.model).to(device)
    matcher = UnifiedCostMatcher(config.matcher)
    models = build_joint_models(jcfg, device)

    # ---- kNN for smooth loss ----
    nbr_idx = None
    if jcfg.lam_smooth > 0:
        nbr_idx = build_knn_graph(c2_rigid, k=jcfg.smooth_k)

    # ---- Train ----
    result = train_joint(
        warpnet, matcher,
        models["encoder"], models["decoder"], models["discriminator"],
        x1, x2, expr1, expr2,
        config.train, jcfg,
        nbr_idx2=nbr_idx,
    )

    # ---- Inference ----
    warpnet.eval()
    x2_def = apply_model(warpnet, x2)
    coords2_final = x2_def.cpu().numpy() * std + mean
    coords2_rigid_denorm = c2_rigid * std + mean

    elapsed = time.time() - start
    return coords2_rigid_denorm, coords2_final, elapsed

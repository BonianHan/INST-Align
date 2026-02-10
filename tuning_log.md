# Hyperparameter Tuning Log

## Best Configuration (Current)

### Global Defaults (config.py)
| Parameter | Value | Notes |
|-----------|-------|-------|
| epochs | 200 | Converges well for all datasets |
| batch_size | 2500 | |
| topk | 64 | 128 tested on MERFISH, worse |
| lr | 1e-3 | |
| lam_jacobian | 0.015 | Default; overridden per dataset |
| jacobian_samples | 512 | Increased from 256 |
| grad_clip | 1.0 | |
| warmup_fraction | 0.3 | Overridden per dataset |
| scheduler_patience | 50 | Increased from 30 |
| tau_init | 0.1 | |
| tau_min | 0.05 | Increased from 0.01 |
| tau_max | 1.0 | Increased from 0.5; EM settles ~0.71-0.79 |
| sinkhorn_iters | 0 | Disabled; incompatible with sparse top-K |
| lam_repulsion | 0.0 | Tested, mixed results, disabled |

### Per-Dataset Overrides (run_test_acc.py)
| Dataset | lam_jacobian | warmup_fraction |
|---------|-------------|-----------------|
| DLPFC | 0.1 | 0.3 |
| STARMap | 0.01 | 0.4 |
| BaristaSeq | 0.01 | 0.4 |
| MERFISH | 0.001 | 0.4 |

### ExprField (Canonical Expression Field)
| Parameter | Value |
|-----------|-------|
| use_expr_field | False (opt-in) |
| lam_canonical | 0.005 |
| pretrain_epochs | 300 |
| n_hvg | 200 |
| latent_dim | 32 |
| batch_emb_dim | 16 |

---

## Tuning History

### Round 1: Initial baseline
- Config: epochs=200, tau_max=0.5, tau_min=0.01, sinkhorn=0, jac_samples=256, patience=30
- OVERALL: OT=0.7753, Ratio=0.3906

### Round 2: tau/patience/sinkhorn changes
- Changed: tau_min→0.05, tau_max→0.5, patience→50, jac_samples→512, sinkhorn→5, DLPFC jac→0.04
- Result: Sinkhorn BROKE training — tau saturated at 0.5, loss unstable
- Root cause: Sinkhorn creates positive feedback with EM tau in sparse top-K framework

### Round 3: Sinkhorn removed, tau_max=0.5
- Changed: sinkhorn→0
- OVERALL: OT=0.7686, Ratio=0.3504

### Round 4: tau_max exploration
- tau_max=0.2: WORSE (Ratio=0.3270 on first pair, should be ~0.05)
- tau_max=1.0: EM naturally settles at ~0.71-0.79, OVERALL OT=0.7716
- Conclusion: Higher tau_max lets EM find natural equilibrium

### Round 5: DLPFC Jacobian tuning
- DLPFC jac=1.0: Some pairs excellent (Ratio=0.009) but others much worse
- DLPFC jac=0.1: Best balance across all pairs
- MERFISH topk=128, bs=5000: Worse than default 64/2500
- MERFISH 400 epochs: No improvement (converges by ep 30)
- Final: DLPFC jac=0.1, OVERALL OT=0.7715, NN=0.7867, Ratio=0.3429

### Round 6: Loss function exploration
- Repulsion loss (anti-collapse): Mixed results, some pairs improved, others much worse
- Decision: Disabled (lam_repulsion=0.0), pivoted to ExprField canonical consistency

### Round 7: ExprField test (DLPFC_sample1)
- ExprField pretrain: 300ep, MSE 1.01→0.91, 23.6s
- Canonical loss active during deformation (lam=0.005, ramp during warmup)
- Results similar to baseline — canonical loss is subtle at 0.005 weight
- Next: tune lam_canonical, potentially increase weight

---

## Key Findings

1. **Sinkhorn is incompatible with sparse top-K matching** — creates positive feedback loop
2. **tau_max=1.0 is optimal** — lets EM freely settle at natural equilibrium (~0.7-0.8)
3. **DLPFC needs stronger Jacobian (0.1)** — grid data needs more regularity
4. **MERFISH bottleneck is matching strategy**, not epochs or topk
5. **Repulsion loss gave mixed results** — abandoned in favor of ExprField approach
6. **ExprField pipeline works end-to-end** — pretrain + canonical consistency loss active

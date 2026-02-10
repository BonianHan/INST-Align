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

## Tuning History & Reasoning

### Round 1: Initial baseline
- Config: epochs=200, tau_max=0.5, tau_min=0.01, sinkhorn=0, jac_samples=256, patience=30
- OVERALL: OT=0.7753, Ratio=0.3906
- **问题**: Ratio 太高（collapse 严重），accuracy 落后 Spateo

### Round 2: Sinkhorn 尝试
- **思路**: Sinkhorn normalization 让 transport plan 更均匀，防止多对一 collapse
- Changed: sinkhorn_iters=5, 同时 tau_min→0.05, patience→50, jac_samples→512
- **结果**: 灾难性失败 — tau 立刻飙到 0.5（max），loss 不稳定，Jacobian 爆炸
- **分析**: Sinkhorn 和 EM tau 更新在 sparse top-K 框架下形成正反馈循环：
  - Sinkhorn 把 weights 推向均匀 → weighted cost 增大 → tau 增大 → 更均匀 → tau 饱和
  - 根本原因：Sinkhorn 设计给 N×N full OT，不适用于 N×K sparse
- **决定**: 完全去掉 Sinkhorn

### Round 3-4: tau_max 探索
- **思路**: tau 控制 matching 的 softness，太低会过于 sharp（collapse），太高会太均匀
- tau_max=0.2: Ratio 反而变差（0.3270 vs 0.2332）— 太低了
- tau_max=1.0: EM 自然稳定在 ~0.71-0.79，不被 cap 住，OVERALL OT=0.7716
- **结论**: tau_max=1.0 最优，让 EM 自由找到 natural equilibrium

### Round 5: Per-dataset Jacobian 调优
- **思路**: 不同数据集形变程度不同，需要不同强度的 Jacobian 正则化
- DLPFC (grid data): jac=1.0 太强（有些 pair 0.009 很好但其他 0.38 很差），jac=0.1 最平衡
- MERFISH: topk=128 bs=5000 更差（OT 从 0.7149 降到 0.7068），400 epochs 无用（epoch 30 就收敛了）
- **结论**: MERFISH 的瓶颈不是 epochs 或 topk，而是 matching 策略本身和 ICP 初始化质量
- Final best: OVERALL OT=0.7715, NN=0.7867, Ratio=0.3429

### Round 6: Repulsion loss（反 collapse 损失）
- **思路**: Jacobian 是局部正则化，能不能加一个全局的反 collapse 信号？
- 实现: `L_rep = mean(ReLU(d_before² - d_after²) / (d_before² + eps))` — 惩罚距离收缩
- **结果**: 混合 — DLPFC P1 改善但 P0 大幅恶化（Ratio 从 0.05→0.43）
- **决定**: 放弃，转向 ExprField canonical consistency

### Round 7: ExprField canonical consistency（v1: frozen backbone + target_fwd）
- **思路**: 预训练 ExprField 消除 batch effect，然后用 canonical(x2_def) vs canonical(target_fwd) 约束形变
- 即：形变后的位置和 matching 到的位置，在 canonical 表达空间应该一致
- lam=0.005: 效果微弱，canon loss 只有 0.003-0.006，远小于 match loss
- lam=0.11: **Ratio 大幅改善** P0: 0.20→0.11, P2: 0.19→0.12，但 OT 略降
- **问题分析**: 为什么效果有限？
  1. target_fwd 是 matching 的结果 — 如果 matching 本身在 collapse，canonical loss 追着 collapse 跑
  2. Backbone 冻住，梯度只通过 PE encoder 传，信号弱
  3. ExprField 表达场空间上 smooth，相邻位置预测差异小
- **但是**: frozen 版本的 Ratio 改善反而是最好的（起到正则化效果）

### Round 8: ExprField canonical consistency（v2: unfrozen backbone + source GT expression）
- **新思路**: 不用 target_fwd（依赖 matching），直接用 source cell 的真实表达做 target
  - Loss = MSE(canonical(x2_def), expr2_source)
  - 含义：形变后的位置的 canonical 表达应该跟 source cell 自己的表达一致
  - 这是独立于 matching 的信号
- **同时**: 去掉 freeze，让 ExprField backbone 一起训练（LR=主LR×0.1）
- lam=0.11: canon loss 值 ~0.2-0.3（比旧版 0.003 大 100 倍），match loss 被淹没
- lam=0.003: 更平衡，但效果不如 frozen 版本
- **DLPFC_sample1 对比表**:

| 版本 | P0 OT | P0 Ratio | P2 OT | P2 Ratio | Mean Ratio |
|------|-------|----------|-------|----------|------------|
| 无 ExprField | 0.8151 | 0.2026 | 0.8594 | 0.1857 | 0.2356 |
| v1 frozen lam=0.11 | 0.8116 | **0.1053** | 0.8470 | **0.1187** | **0.1988** |
| v2 unfrozen lam=0.11 | 0.8082 | 0.1638 | 0.8561 | 0.1954 | 0.2250 |
| v2 unfrozen lam=0.003 | 0.8119 | 0.1971 | 0.8580 | 0.1772 | 0.2283 |

- **分析**:
  - v1 frozen+target_fwd Ratio 最好，因为 frozen ExprField + 高权重 canon loss 起到了**隐式正则化**效果
  - v2 unfrozen 的问题：canon loss 太大（ExprField pretrain MSE=0.91 就是上限），loss landscape 被 canon 主导
  - ExprField 拟合能力有限（200 genes, 4 层 MLP），无法完美预测表达，所以 canon loss 有一个无法消除的 residual

---

## Key Findings

1. **Sinkhorn 与 sparse top-K 不兼容** — 正反馈循环导致 tau 饱和
2. **tau_max=1.0 最优** — EM 自由收敛到 ~0.71-0.79
3. **DLPFC 需要强 Jacobian (0.1)** — grid data 需要更多形变规律性
4. **MERFISH 瓶颈是 matching 策略** — 不是 epochs 或 topk
5. **Repulsion loss 效果不稳定** — 有些 pair 改善但其他恶化
6. **ExprField frozen+target_fwd 最有效** — 虽然理论上有闭环问题，但实际起到了正则化效果
7. **ExprField unfrozen+source_GT** — canon loss 数值太大（受限于 ExprField 拟合能力），需要更精细的权重平衡

## Open Questions / Next Steps
- 能不能只用 ExprField 的 bottleneck embedding 做 cosine similarity loss，而不是 MSE on full expression？
- 增加 ExprField 拟合能力（更多 HVG、更深网络）能否让 unfrozen 版本更有效？
- frozen v1 在全量 benchmark 上的表现？

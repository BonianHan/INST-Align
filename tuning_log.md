# Hyperparameter Tuning Log

## Best Configuration (Current — 2026-02-18)

### Best Results (OVERALL 13 pairs: DLPFC 9 + STARMap 2 + BaristaSeq 2)
| Metric | Value |
|--------|-------|
| OT Accuracy↑ | 0.792 |
| NN Accuracy↑ | 0.797 |
| Ratio↓ | 0.257 |
| CLC↑ | 0.592 |

### Global Defaults (config.py TrainConfig)
| Parameter | Value | Notes |
|-----------|-------|-------|
| epochs | 200 | Converges well for all datasets |
| batch_size | 2500 | |
| topk | 64 | 128 tested on MERFISH, worse |
| lr | 1e-3 | |
| weight_rev | 1.0 | Reverse matching weight (3.0 tested, worse) |
| lam_jacobian | 0.015 | Default; overridden per dataset |
| lam_divergence | 10.0 | Penalizes negative divergence (global compression) |
| lam_sigma_floor | 1.0 | Hard σ floor penalty (rarely activates, safety net) |
| sigma_min | 0.8 | Min allowed singular value |
| full_reverse_interval | 1 | Full-coverage reverse loss every epoch |
| lam_uniqueness | 0.1 | Assignment uniqueness loss (0 for grid data) |
| grad_clip | 1.0 | |
| warmup_fraction | 0.3 | Overridden per dataset |
| scheduler_patience | 50 | Increased from 30 |
| tau_init | 0.1 | |
| tau_min | 0.05 | Increased from 0.01 |
| tau_max | 1.0 | EM settles ~0.71-0.79 naturally |
| sinkhorn_iters | 0 | Disabled; incompatible with sparse top-K |

### Training Loss Components
1. **Forward matching**: deformed source → target soft top-K weighted centroid
2. **Reverse matching** (batch): target → deformed source (within batch)
3. **Full-coverage reverse** (per epoch): separate optimizer step with ALL N2 deformed points
4. **Jacobian SVD**: penalizes σ deviating from 1, compression 5× weighted
5. **Divergence**: penalizes negative div(δ), captures global compression
6. **Sigma floor**: ReLU(σ_min - σ)² hard floor (rarely activates)
7. **Assignment uniqueness**: penalizes per-target load variance (anti many-to-one)
8. **Embedding KL + Gene recon**: optional, with Splane embeddings

### Per-Dataset Overrides (run_test_acc.py)
| Dataset | lam_jacobian | warmup | lam_divergence | sigma_min | lam_sigma_floor | lam_uniqueness |
|---------|-------------|--------|----------------|-----------|-----------------|----------------|
| DLPFC | 0.1 | 0.3 | 10.0 | 0.8 | 1.0 | **0.0** |
| STARMap | 0.01 | 0.4 | 10.0 | 0.8 | 1.0 | 0.1 |
| BaristaSeq | 0.01 | 0.4 | 10.0 | 0.8 | 1.0 | 0.1 |
| MERFISH | 0.001 | 0.4 | 10.0 | 0.8 | 1.0 | 0.1 |

Note: DLPFC uses lam_uniqueness=0.0 because uniqueness loss hurts grid data alignment.

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

### Round 9: ExprField 自重建 — frozen encoder vs unfrozen encoder (2026-02-11)

**改动**: Stage 2 不再用 matched target expression，改为 self-reconstruction:
  - Loss = MSE(ExprField(x2_def, slice_id=src), expr2_src)
  - 每个 source cell 重建自己的 expression
  - match_decay: 线性衰减 matching loss weight (1.0 → 0.1)

**实验A — frozen encoder (backbone+bottleneck frozen, 只训练 FiLM+head)**:
  - DLPFC sample0 (3 pairs): Mean OT ≈ 0.597 (baseline 0.604)
  - R² pretrain: 0.089 (200 genes), 0.036 (2000 genes) — very low on grid data
  - recon loss ~0.90-0.93，几乎不下降，梯度噪声太大

**实验B — unfrozen encoder (整个 ExprField 与 DeformNet 一起训练)**:
  - DLPFC 全部 9 pairs (3 samples × 3 pairs):

| Method | OT Acc (9p) | NN Acc (9p) | Ratio (9p) |
|--------|-------------|-------------|------------|
| Baseline (no EF) | **0.7476** | **0.7530** | 0.1933 |
| Unfrozen ExprField | 0.7418 | 0.7557 | **0.2017** |

  - 几乎没有差异 (ΔOT < 0.6%)
  - 修复了 pair_idx bug: 原来所有 pair 都用 slice[1] 的表达 + src_slice_id=1
  - 即使修复后，ExprField 对 DLPFC 的 alignment 无实质帮助

**结论**: ExprField self-reconstruction 在 DLPFC 上无论 frozen/unfrozen 都不能提升 alignment。
核心原因: 在 grid data 上 coords→expression mapping 拟合质量太差 (R² < 0.1)，
recon loss 残差大，梯度信号被噪声主导。

---

### Round 10: Embedding-driven Matching (2026-02-11)

**核心 idea**: 用 ExprField 学到的 embedding 替换 PCA 做 matching，分阶段训练

**设计**:
1. **Stage 0**: 预训练 ExprField (在 reference slice 上 pretrain, 300 epochs)
2. **Stage 1**: 用 PCA 做 matching，训练 DeformNet (warmup, 与 baseline 相同)
3. **Stage 2** (epoch 30+): 联合训练 —
   - 用 ExprField 的 bottleneck embedding (32d, L2 normalized) 替换 PCA embedding
   - `emb2_batch = F.normalize(expr_field.get_embedding(x2_def), dim=1)` — differentiable through x2_def
   - `emb1_ef` 预计算参考 embedding，每 10 epoch 刷新
   - 同时训练 DeformNet (LR×0.5) + ExprField (LR×0.1)
   - Self-recon loss 作为可选辅助

**DLPFC 实验结果 (9 pairs)**:

| Method | OT Acc | NN Acc | Ratio |
|--------|--------|--------|-------|
| Baseline (no EF) | **0.7476** | **0.7530** | **0.1933** |
| Embedding-driven | 0.7447 | 0.7559 | 0.1905 |

**失败原因 — tau collapse**:
- 进入 Stage 2 后，tau 从 ~0.64 快速降到 0.05 (floor)，大部分 pair 在 30-40 epoch 内就降到底
- 只有 S1-P1 例外：tau 稳定在 ~0.34（因为 RMSE=0.072 较大，spatial cost 本身有区分力）
- 根本原因：ExprField pretrain R² 只有 0.10-0.13 (grid data)，embedding 区分力不够
- tau 触底后 → feat_dist 对 matching 贡献为零 → 退化为纯 spatial matching → 和 baseline 一样
- recon loss ~0.78-1.01，无法有效下降（ExprField 对 grid data 拟合能力有限）

**结论**: Embedding-driven matching 在 DLPFC 上无效。
核心瓶颈仍是 ExprField 在 grid data 上拟合能力不足 (R² < 0.15)。
代码已 revert 回 baseline。

---

### Round 11: Spatial compression 根因分析 (2026-02-14)

**问题**: 非刚性变形 (Ours_Spatial) 在 Ratio 和 CLC 上始终差于 Rigid：

| Method | OT↑ | NN↑ | Ratio↓ | CLC↑ |
|--------|------|------|--------|------|
| Ours_Rigid | 0.740 | 0.756 | **0.038** | **0.705** |
| Ours_Spatial | 0.742 | 0.750 | 0.169 | 0.598 |
| SPACEL | **0.776** | **0.875** | 0.120 | **0.768** |

**根因分析** (5 个因素):

1. **Forward matching loss 是"向心力"**: 每个 source 点被拉向 K 个 target 邻居的加权质心 (mean-shift)。同区域多个 source 点被拉向同一质心 → 系统性 compression。

2. **Reverse loss 被 batch 削弱**: Reverse 查询全部 N1 个 target → batch 内的 2500 个 deformed source。batch < N1 时覆盖不完整，anti-collapse 信号弱。去掉 reverse loss 后结果更差（Ratio 0.169→0.205），证明它确实在起作用，只是不够强。

3. **EM tau 正反馈**: 点越近 → distance 越小 → tau 越小 → softmax 越 sharp → 拉力越集中 → 更近。tau 收敛到 ~0.76 后稳定，但 compression 已形成。

4. **Jacobian 正则是对称的**: `log(σ)²` 对 compression 和 expansion 惩罚相同，但 matching loss 只奖励 compression（靠近 target），Jacobian 在跟单方向力对抗，必然输。

5. **snap_to_grid 放大问题**: 多个点挤到同一 grid cell → 贪心分配 → 输家被踢到远处空 cell → neighborhood 结构进一步破坏。

**实验验证**:

| 配置 | OT↑ | NN↑ | Ratio↓ | CLC↑ |
|------|------|------|--------|------|
| 有 reverse loss | 0.742 | 0.750 | 0.169 | 0.598 |
| 去掉 reverse loss | 0.725 | 0.728 | 0.205 | 0.561 |
| + ExprField joint (lam=0.001) | 0.742 | 0.750 | 0.169 | 0.598 |
| + ExprField joint (lam=1.0) | 0.620 | 0.627 | 0.191 | 0.489 |

ExprField joint training 无论权重大小都无法缓解 compression — ExprField 靠自身参数拟合表达，梯度不推动 deformation。

## Open Questions / Next Steps
- 改进 matching loss: 需要某种 injectivity 约束或 repulsion 信号
- 改进 Jacobian: 非对称惩罚（compression 惩罚更重）或 hard floor (σ >= 0.8)
- 或者根本性改变思路: 不用 soft centroid matching, 用 displacement field regression

# 新版改动与日志核查（自动审阅）

## 核查范围
- 最新提交：`0dbb963`（Add CLC metric, CPD-style outlier matching, cleanup unused losses）
- 调参日志：`tuning_log.md`

## 我看到的新版重点
1. 新增了 CLC 指标与 CPD 风格 outlier matching。
2. 清理了未使用的 loss 项，训练路径更聚焦。
3. 参数层面保留了 `tau_max=1.0`、`sinkhorn_iters=0` 的策略。

## 与日志的一致性结论
- 与 `tuning_log.md` 中 Round 2/3/10 的结论一致：
  - sparse top-K 下不启用 Sinkhorn；
  - `tau_max=1.0` 允许 EM 自由收敛；
  - ExprField 在 DLPFC 上收益有限，主线仍应以 baseline matching 稳定性为主。

## 建议（下一轮）
1. 在 benchmark 输出里固定展示 CLC + Ratio + OT 三项，避免"单指标改善"误判。
2. 对 MERFISH/STARMap 单独评估 ExprField embedding-driven（日志里已指出更可能有效）。
3. 给 `run_test_acc.py` 增加一次性 summary 导出（CSV/Markdown）以便复现实验记录。

---

### Round 12: Anti-compression — sigma floor + full-coverage reverse + uniqueness (2026-02-18)

**目标**: 修复 Nonrigid 的 Ratio/CLC 退化问题。

**新增 Loss 组件**:

1. **Sigma floor loss** (`lam_sigma_floor=1.0, sigma_min=0.8`):
   - `ReLU(sigma_min - σ)²` — 硬性防止 singular value 低于 σ_min
   - 实测结果：floor loss 始终为 0.000000（SVD singular values 天然 ≈ 1.0）
   - 结论：compression 不在 Jacobian 层面，而是坐标层面的多对一坍缩
   - 保留为安全网

2. **Full-coverage reverse loss** (`full_reverse_interval=1`):
   - 每 epoch 一次单独 optimizer step：用 ALL N2 个 deformed source 点做 reverse matching
   - 修复了原来 batch-limited reverse 的覆盖不全问题
   - **主要改善来源**: DLPFC Ratio 从 0.169 → 0.070 (Pair 0 几乎完美)

3. **Assignment uniqueness loss** (`lam_uniqueness=0.1`):
   - `var(per_target_load)` — 惩罚某些 target 被过多 source 点映射
   - 对 scattered data (STARMap, BaristaSeq) 有效
   - **DLPFC (grid) 必须关掉** (`lam_uniqueness=0.0`)：uniqueness loss 会扰乱 grid 对齐

**实验历程**:

| 配置 | DLPFC Ratio↓ | DLPFC CLC↑ | 备注 |
|------|-------------|-----------|------|
| Round 11 baseline | 0.169 | 0.598 | 基线 |
| + sigma_floor=0.5 | 0.070 | 0.548 | floor 未激活, 改善来自 full reverse |
| + sigma_floor=0.8 | 0.070 | 0.548 | floor 仍未激活 |
| + weight_rev=3.0 | 0.133 | — | 更差, reverted |
| + uniqueness=0.1 | 0.077 (mixed) | 0.548 | Pair 0 完美(0.014), Pair 1 仍差 |
| + uniqueness=0.5 | 0.120 | — | 更差, reverted to 0.1 |
| DLPFC uniqueness=0, others=0.1 | 0.127 | 0.629 | **最终配置** |

**最终结果 (OVERALL 13 pairs)**:

| Metric | Round 11 | Round 12 | Delta |
|--------|----------|----------|-------|
| OT↑ | 0.772 | **0.794** | +2.8% |
| NN↑ | 0.787 | **0.797** | +1.3% |
| Ratio↓ | 0.343 | **0.257** | −25.1% |
| CLC↑ | — | 0.592 | (new metric) |

**Griddata 后处理测试** (2026-02-18):
- 尝试用 `griddata_resample` 后处理 deformed coords 再算 metric
- 结果：Ratio 0.070→0.154, CLC 0.548→0.451 — **全面恶化**
- 原因：NN label transfer 破坏精确的 cell-to-cell 对应关系
- 决定：**彻底移除 griddata** (use_griddata, griddata_resample, resample_to_grid)

**关键发现**:
- Forward matching 的向心力是 Ratio 退化的根本原因，不在 Jacobian 层面
- Full-coverage reverse loss 是最有效的改善
- Uniqueness loss 对 scattered data 有效但对 grid data 有害
- Griddata 后处理不适合 metric 计算

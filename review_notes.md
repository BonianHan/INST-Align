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
1. 在 benchmark 输出里固定展示 CLC + Ratio + OT 三项，避免“单指标改善”误判。
2. 对 MERFISH/STARMap 单独评估 ExprField embedding-driven（日志里已指出更可能有效）。
3. 给 `run_test_acc.py` 增加一次性 summary 导出（CSV/Markdown）以便复现实验记录。

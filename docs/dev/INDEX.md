# 开发计划与实施结果索引

nnlearn 是提供 PyTorch 风格核心训练接口的独立张量与自动微分框架。此索引用于关联开发计划、实际变更及验证结果。

本文件于 PLAN-062 引入。此前计划和结果仍保存在 `docs/dev/`，尚未逐项补录；不推测历史执行时间。

| 更新时间（Asia/Shanghai） | 计划 | 结果 | 实施摘要 |
| --- | --- | --- | --- |
| 2026-10-24 02:53 | [PLAN-062-training-dtype-consistency.md](PLAN-062-training-dtype-consistency.md) | [PLAN-062-training-dtype-consistency-OUTCOME.md](PLAN-062-training-dtype-consistency-OUTCOME.md) | 统一 Python 常量与浮点梯度精度，修复均值、极值、零维激活和优化器状态升精度；新增 292 项，CPU 全量 702 项通过，NumPy 1.24 兼容回归 25 项通过。 |

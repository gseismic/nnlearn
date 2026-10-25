# 开发计划与实施结果索引

nnlearn 是提供 PyTorch 风格核心训练接口的独立张量与自动微分框架。此索引用于关联开发计划、实际变更及验证结果。

本文件于 PLAN-062 引入。此前计划和结果仍保存在 `docs/dev/`，尚未逐项补录；不推测历史执行时间。

| 更新时间（Asia/Shanghai） | 计划 | 结果 | 实施摘要 |
| --- | --- | --- | --- |
| 2026-10-25 | [PLAN-065-transformer-masks.md](PLAN-065-transformer-masks.md) | [PLAN-065-transformer-masks-OUTCOME.md](PLAN-065-transformer-masks-OUTCOME.md) | 新增注意力 padding、causal 与二维/三维掩码，相关测试 104 项通过；两端变长序列训练收敛，未来与 padding 扰动误差均为零。 |
| 2026-10-25 | [PLAN-066-ci-packaging.md](PLAN-066-ci-packaging.md) | [PLAN-066-ci-packaging-OUTCOME.md](PLAN-066-ci-packaging-OUTCOME.md) | 新增固定版本 CPU 持续集成、无 PyTorch 训练及发行包验收；两套 Python 环境与 8 个干净安装通过，远程 Actions 随提交触发并在最终集成时核验。 |
| 2026-10-25 14:14 | [PLAN-064-higher-order-autograd.md](PLAN-064-higher-order-autograd.md) | [PLAN-064-higher-order-autograd-OUTCOME.md](PLAN-064-higher-order-autograd-OUTCOME.md) | 补齐 LayerNorm、交叉熵、Conv2d 高阶反向；29 项新增回归通过，包含 PyTorch 联合二阶对照及独立中央差分，CPU 联合回归 747 项通过。 |
| 2026-10-25 14:08 | [PLAN-063-half-precision-stability.md](PLAN-063-half-precision-stability.md) | [PLAN-063-half-precision-stability-OUTCOME.md](PLAN-063-half-precision-stability-OUTCOME.md) | 修复 Softmax、交叉熵与归一化半精度中间溢出；新增 38 项通过，NumPy 1.24 无参考框架 20 项通过；CUDA 联合验收见 PLAN-067。 |
| 2026-10-24 02:53 | [PLAN-062-training-dtype-consistency.md](PLAN-062-training-dtype-consistency.md) | [PLAN-062-training-dtype-consistency-OUTCOME.md](PLAN-062-training-dtype-consistency-OUTCOME.md) | 统一 Python 常量与浮点梯度精度，修复均值、极值、零维激活和优化器状态升精度；新增 292 项，CPU 全量 702 项通过，NumPy 1.24 兼容回归 25 项通过。 |

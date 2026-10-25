# PLAN-064：神经网络算子的高阶导数

## 目标

明确并验证高阶导数支持范围，补齐 LayerNorm、CrossEntropy、Conv2d 的二阶依赖和数值回归。

设计：`docs/design/autograd-20261025-higher-order.md`。本任务在 PLAN-063 数值修复之后落地，避免同文件修改冲突。

## 步骤

1. 新增 `tests/test_53_higher_order_nn.py`，记录断图失败与数值梯度证据。
2. 在 create_graph 路径中使用可微 Tensor 表达式，保留现有一阶快速路径与接口。
3. 对照 PyTorch 的一阶、方向二阶及交叉二阶导；使用中央差分独立检查，并复验重复反向与 dtype。
4. 编写 `docs/autograd.md` 支持边界，复查实现及全量测试，生成 `PLAN-064-higher-order-autograd-OUTCOME.md`，更新索引并中文提交后立即推送。

## 验收

- 三类算子在其现有参数子集上保留输入、参数和非线性上游的可微依赖。
- 数值差分与 PyTorch 对照通过，现有一阶训练及数值稳定性无回归。
- 支持范围、常量标签/类别权重及未验证算子边界有中文文档。

## 独立复查后的补充修复

1. 半精度输入配合显式 float32/float64 仿射参数时，高阶 LayerNorm 只能提升低精度上游，不得把 float64 上游强转 float32。新增大幅值且仍有限的参数梯度回归，对照普通反向与建图反向。
2. Conv2d 的合法零批次输入在普通一阶可用，高阶窗口不能用 `(0, -1)` 推断维度。新增零批次前向、一阶和二阶 PyTorch 对照，并把窗口与上游 reshape 改为明确空间大小。
3. 更新结果记录并复验相关及全部 CPU 回归；后续 CUDA 验收仍依 PLAN-067。

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

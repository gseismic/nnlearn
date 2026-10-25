# 神经网络算子的高阶导数

日期：2026-10-25。对应 PLAN-064。

## 目标与接口

沿用 `loss.backward(create_graph=True)` 和后续对梯度调用 backward 的接口，不增加另一套自动微分入口。重点补齐 LayerNorm、CrossEntropy 和 Conv2d 的梯度表达式；权重、输入和非线性上游都应保留依赖。

## 内部设计

- LayerNorm 在需要构建梯度图时，用输入 Tensor 重新计算均值、方差、标准化结果与解析反向，半精度使用可微 float32 转换。普通一阶路径保留数组缓存以控制开销。
- CrossEntropy 在需要梯度图时，从原 logits 重新计算 softmax，并将既有标签分布、掩码、权重与分母视作常量，通过 Tensor 运算应用上游梯度。类别标签和类别权重不作为可微输入。
- Conv2d 在需要梯度图时，用可微切片、零填充、矩阵乘法和求和组成输入与权重梯度；普通一阶路径保留已有数组实现。支持现有 stride/padding/bias 子集，不扩展 groups/dilation。
- 文档按算子明确已验证的高阶范围；不以少数多项式测试推导全部算子支持高阶导数。

## 验证

先用小型 float64 输入在 PyTorch 对照一阶梯度、Hessian-vector product 和输入/参数交叉二阶导，并用中央差分独立核对一阶与二阶方向导数。覆盖非线性上游、重复 backward、不同 reduction/stride/padding；float32 和 float16 额外核查输出与梯度精度。CUDA 单独由 PLAN-067 复验。

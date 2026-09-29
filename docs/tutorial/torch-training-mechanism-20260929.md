# 从训练结果看懂自动微分

日期：2026-09-29

本教程用一个线性模型，逐步比较 `torch_1k` 和 PyTorch 的前向值、损失、梯度与一次 SGD 参数更新。两边使用完全相同的输入、目标、初始权重、初始偏置和学习率。

## 适合谁

如果你已经会写基本的 PyTorch 训练代码，但想弄清 `loss.backward()` 如何得到梯度、优化器如何改参数，可以从这里开始。

## 准备与运行

`torch_1k` 依赖 NumPy 和 Loguru。PyTorch 只用作对照实现，需要单独安装：<https://pytorch.org/get-started/locally/>。

在仓库根目录运行：

```bash
pip install .
python examples/example39_training_mechanism_comparison.py
```

脚本会打印两边各步骤的结果和最大绝对误差。输入、权重和梯度使用 `float32`；误差不超过 `1e-6` 时示例报告通过。

## 模型与损失

本例使用批量线性模型：

\[
\hat{Y} = XW + b
\]

均方误差为：

\[
L = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
\]

反向传播得到：

\[
\frac{\partial L}{\partial W} = \frac{2}{N}X^T(\hat{Y}-Y),\qquad
\frac{\partial L}{\partial b} = \frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)
\]

SGD 再按学习率 `lr` 更新参数：

\[
W \leftarrow W - lr\frac{\partial L}{\partial W},\qquad
b \leftarrow b - lr\frac{\partial L}{\partial b}
\]

## 对照程序的关键步骤

[`example39_training_mechanism_comparison.py`](../../examples/example39_training_mechanism_comparison.py) 先用同一份 NumPy 常量分别创建两套张量：

```python
inputs = torch_api.tensor(INPUTS, dtype=torch_api.float32)
targets = torch_api.tensor(TARGETS, dtype=torch_api.float32)
weight = torch_api.tensor(
    INITIAL_WEIGHT, dtype=torch_api.float32, requires_grad=True
)
bias = torch_api.tensor(
    INITIAL_BIAS, dtype=torch_api.float32, requires_grad=True
)
```

`requires_grad=True` 标记需要求导的参数。前向表达式是普通张量运算：

```python
prediction = inputs @ weight + bias
loss = ((prediction - targets) ** 2).mean()
```

在 `torch_1k` 中，运算会创建 `Function` 节点，并通过输出 Tensor 的 `creator` 指回产生它的节点。`loss.backward()` 从损失开始反向遍历这些节点，逐项应用链式法则，把梯度累加到权重和偏置。可以从 [`tensor.py`](../../torch_1k/tensor.py) 的 `Tensor.backward()` 和 [`function.py`](../../torch_1k/function.py) 的 `Function` 协议继续阅读。

然后，两边都使用 SGD 做一次更新：

```python
optimizer = optimizer_api.SGD([weight, bias], lr=LEARNING_RATE)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

脚本在更新前记录前向值、损失和梯度，在更新后记录参数。这样能区分“模型输出一样”和“梯度及优化过程也一样”。

## 读输出时看什么

- `prediction` 检查矩阵乘法、广播偏置和前向计算。
- `loss` 检查均方误差及规约。
- `weight_grad`、`bias_grad` 检查链式法则和梯度累加。
- `weight_after_step`、`bias_after_step` 检查优化器更新是否使用了对应梯度和学习率。
- 每个项目的最大绝对误差展示两种浮点实现之间的数值差异。

这个例子只核对一层线性模型的一次 SGD 更新。它不能证明所有算子、所有模型、所有 PyTorch 脚本都兼容，也没有比较训练速度。需要理解更完整的数据加载与模型结构时，可以继续看 `examples/example38_beginner_tutorial.py` 和 `examples/example36_pytorch_training_baseline.py`。

## 项目边界

`torch_1k` 用于学习和检查小型训练机制。它只实现一部分 PyTorch 风格 API，不提供完整兼容性或生产性能保证。项目提供基于 CuPy 的可选 CUDA 路径；CUDA 实际可用性取决于本机的 CuPy、驱动和设备环境。

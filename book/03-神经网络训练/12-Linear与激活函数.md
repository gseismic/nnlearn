# 12 Linear 与激活函数

最常用的层之一是线性变换。给定批量输入 `X`、权重 `W` 和偏置 `b`：

```text
Y = X @ W + b
```

若 `X` 的形状是 `(N, in_features)`，`W` 是 `(in_features, out_features)`，则输出 `Y` 是 `(N, out_features)`。偏置形状是 `(out_features,)`，通过广播加到每个样本上。

## 初始化和前向

配套 `Linear` 用均匀分布初始化权重，范围为 `±sqrt(6 / (in_features + out_features))`，偏置从 0 开始。这个初始化让信号在层间传递时保持较合适的尺度；更复杂的初始化会在后续需要时再加入。

调用 `layer(inputs)` 会执行矩阵乘法和偏置相加。因为这两步是 `mini_tensor` 已支持的可微运算，层本身不必手写线性层的反向公式。

## 激活函数

堆叠多个线性层仍然只得到一个线性变换。激活函数在层之间加入非线性：

- ReLU：`max(0, x)`，正数区间梯度为 1，非正区间梯度设为 0。
- Tanh：`tanh(x)`，导数为 `1 - tanh(x)²`。
- Sigmoid：`1 / (1 + exp(-x))`，导数为 `s(x)(1-s(x))`。

运行[配套脚本](../代码/03-神经网络训练/12-Linear与激活函数.py)：

```bash
python 'book/代码/03-神经网络训练/12-Linear与激活函数.py'
```

它打印输入与输出形状，并让 ReLU、Tanh 和 Sigmoid 的梯度经过 `Function` 回到原输入。

## 练习

1. 手算 `(2, 3)` 输入经过 `Linear(3, 4)` 后的输出形状。
2. 将 ReLU 在 0 处的梯度改为 1，讨论为什么常用约定会选择 0。
3. 把两层线性层直接串联与在线性层间加入 ReLU，比较两种网络表达能力。

## 对应到仓库

仓库的 `torch_1k/nn/linear.py` 和 `torch_1k/nn/activation.py` 提供对应模块；其中算子实现位于 `torch_1k/functional/` 与 `torch_1k/nn/functional.py`。

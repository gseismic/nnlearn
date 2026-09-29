# 10 组织 Tensor 和 Function

上一章完成了常用梯度公式。现在把它们组织成两个角色：`Tensor` 保存数据并指向产生它的运算；`Function` 负责一次运算的前向与反向。

## Function 的前向和反向

一次 `Function` 调用分为三步：

1. 把输入转换成 Tensor，取出它们的 NumPy 数据执行 `forward()`。
2. 如果任何输入需要梯度，就记录输入、输出和当前代数，并把 Function 设为输出的 creator。
3. 反向传播时，Function 接收输出梯度，通过 `backward()` 返回每个输入的梯度。

加法、乘法、矩阵乘法、重塑、求和与均值各自定义自己的梯度规则。`Tensor.backward()` 不需要知道每种公式，只需按代数从高到低调用这些规则。

## 代数保证反向顺序

叶子 Tensor 的 generation 是 0。某个操作的输入中 generation 最大值为 `g`，它输出的 generation 设为 `g + 1`。反向时优先处理代数最高的 Function，这保证来自后续运算的梯度先到齐，再继续传给它的输入。

同一输入可能出现在多条路径上，所以引擎用 Tensor 身份作为键累加梯度。最终只把梯度保存到需要求导的叶子 Tensor；中间节点的梯度在反传过程中临时保留。

## 跑通完整 Tensor 例子

运行[配套脚本](../代码/02-Tensor与自动微分/10-组织Tensor和Function.py)：

示例计算一个小批次线性层的预测和均方误差，再打印输入、权重和偏置的梯度。它尚未封装 `nn.Module` 或优化器，下一篇会在这个 Tensor 引擎之上增加这些组件。

默认数据下得到：

```text
预测值: [[0.5], [0.4]]
均方误差: 0.305000
dL/dinputs: [[-0.05, -0.1], [-0.06, -0.12]]
dL/dweight: [[-1.7], [-1.6]]
dL/dbias: [-1.1]
```

## 与 torch_1k 的关系和限制

本篇的 [`mini_tensor.py`](../代码/02-Tensor与自动微分/mini_tensor.py) 是教学实现，使用 NumPy 数组和 Python 对象组成动态图。仓库的 `torch_1k/function.py`、`torch_1k/tensor.py` 与 `torch_1k/functional/` 采用相近的职责划分，但还处理设备、上下文、高阶梯度和更多算子。

本篇引擎仅实现单输出算子、稠密实数数组、一次反向模式梯度和二维矩阵乘法。它不实现完整 PyTorch API、CUDA、稀疏张量或高阶导数。

## 练习

1. 为 `Tensor` 增加 `tanh()`，写出并实现它的反向公式。
2. 添加矩阵乘法的数值梯度检查，分别扰动左矩阵和右矩阵中的一个元素。
3. 让一个非叶子 Tensor 也能选择保留 `.grad`，并说明何时需要它。

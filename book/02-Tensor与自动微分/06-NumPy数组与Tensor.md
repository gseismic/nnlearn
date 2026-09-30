# 06 NumPy 数组与 Tensor

标量自动微分篇里，一个 `Value` 保存一个 Python 浮点数。神经网络计算往往一次处理一批样本，因此我们需要能表示多维数据的数组。

NumPy 的 `ndarray` 已经提供存储、索引和数值运算。本书的 `Tensor` 先包住一个 `ndarray`，再逐步增加形状操作、算子和梯度记录。

## 形状告诉我们数据如何排列

同样是 6 个数，可以有不同形状：

```text
[1, 2, 3, 4, 5, 6]           shape = (6,)
[[1, 2, 3], [4, 5, 6]]       shape = (2, 3)
```

形状是每个轴长度组成的元组。`(2, 3)` 表示两个样本、每个样本有三个特征。一个 NumPy 标量的形状是 `()`，只有一个元素但有一个轴的数组形状是 `(1,)`；二者不相同。

NumPy 数据还带有 `dtype`，例如 `int64`、`float32` 和 `float64`。整数适合索引和计数，训练参数与梯度通常使用浮点类型。配套实现允许 Tensor 保存整数数组，但只允许浮点 Tensor 设置 `requires_grad=True`。

## Tensor 先负责保存和说明数据

运行[配套脚本](../代码/02-Tensor与自动微分/06-NumPy数组与Tensor.py)：

```bash
python 'book/代码/02-Tensor与自动微分/06-NumPy数组与Tensor.py'
```

它展示标量、向量、矩阵的形状和数据类型，也演示 `Tensor` 与原始数组之间的拷贝边界。`Tensor.numpy()` 返回数据副本；这让外部数组修改不会悄悄绕过 Tensor 的状态。

关键输出：

```text
标量: shape=(), ndim=0, item=3.0
向量: shape=(3,), ndim=1, dtype=float64, values=[1.0, 2.0, 3.0]
矩阵: shape=(2, 3), ndim=2, dtype=int64
显式类型: float32
原数组和导出副本修改后，Tensor 仍是: [1.0, 2.0, 3.0]
```

本书的小型 Tensor 不会复制 PyTorch 的全部规则。它只实现当前章节所需的能力，后续再加上梯度和计算图。

## 练习

1. 创建形状分别为 `()`、`(1,)`、`(2, 3)` 和 `(2, 1, 3)` 的数据，解释每个轴表示什么。
2. 比较 `np.array([1, 2])` 与 `np.array([1.0, 2.0])` 的 `dtype`。
3. 修改代码中的原始 NumPy 数组，再读取 Tensor，确认拷贝边界。

## 对应到仓库

仓库中的 [Tensor 定义](../../nlearn/tensor.py)同样把数据交给后端管理，并提供 `shape`、`ndim`、`dtype` 等信息。本篇先直接使用 NumPy，暂不引入 CPU/CUDA 后端抽象。

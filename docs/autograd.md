# 自动微分与高阶导数

普通训练调用 `loss.backward()`。需要继续对梯度求导时，第一次调用 `loss.backward(create_graph=True)`，保存所需梯度，再清空叶子梯度，对保存的梯度表达式调用 `backward()`。

```python
import nnlearn as nl

x = nl.tensor([0.5, 1.5], requires_grad=True)
loss = (x ** 3).sum()
loss.backward(create_graph=True)
first = x.grad
x.zero_grad()
first.sum().backward()
print(x.grad.tolist())  # [3.0, 9.0]
```

## 已验证的范围

| 路径 | 验证范围 |
| --- | --- |
| 基础算术、激活、均值、形状、索引和复制 | 以 `tests/test_06_highorder.py`、`tests/test_48_review_autograd_api.py`、`tests/test_50_training_dtype_consistency.py` 中具体用例为准；不推导为所有组合都已验证。 |
| LayerNorm | 一维和多轴 normalized_shape；输入、weight、bias 的一阶及联合方向二阶导；非线性上游、重复 backward 和 float16/32/64。 |
| CrossEntropy | 整数类别标签，(N,C) 和 (N,C,...) 输入，三种 reduction、ignore_index、类别权重及 label_smoothing；logits 和非线性上游的一阶及方向二阶导。类别标签、类别权重按常量处理。 |
| Conv2d | 现有 NCHW 子集，整数或二元组 stride/padding、可选 bias；输入、weight、bias 一阶与交叉二阶导；float16/32/64。groups、dilation 等未提供的参数不在范围内。 |

`tests/test_53_higher_order_nn.py` 同时对照 PyTorch 和独立中央差分，比较全部输入的梯度及 Hessian-vector product，避免仅检查梯度形状。CUDA 对照见 `tests/test_55_cuda_training.py` 与 [CUDA 验收](cuda.md)。半精度中间统计量使用 float32，公开输出及每条回传边恢复目标 dtype。

## 使用边界

- `create_graph=True` 会保留更多 Tensor 运算和计算图，卷积通过可微窗口展开实现高阶反向，开销高于普通一阶数组路径。
- 非标量输出的 `backward()` 沿用本库的全一向量种子；需要特定方向时先构造 `(output * direction).sum()`。
- `retain_grad` 控制中间 Tensor 的 `.grad` 是否保留，不等价于 PyTorch 的 `retain_graph` 参数。本库同一计算图重复反向会累积叶子梯度，需按训练意图调用 `zero_grad()`。
- ReLU、极值、池化和离散索引包含不可微点；本文不承诺这些点的数学二阶导数，也不承诺所有其他算子的高阶导数。未在表中列出的模块需要按其具体实现和测试核实。
- 本库不提供 PyTorch 的完整 `autograd.grad`、Jacobian/Hessian 构造器或梯度图原地修改检查。前向到反向之间应保持参与计算的参数及输入数值不变。

设计参考：[PyTorch 自定义算子的二阶反向教程](https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html)。教程说明了缓存裸数组时需要在反向中重新建立可微表达式；本库采用自己的 Tensor/Function 引擎实现这一机制。

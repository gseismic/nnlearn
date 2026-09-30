# 11 Parameter 与 Module

前一篇的 `Tensor` 能计算梯度，但一个模型通常含有很多需要训练的 Tensor。我们需要明确区分可训练参数和普通输入，也需要一种方式把多个层组合成模型。

## Parameter 表示可训练状态

`Parameter` 是默认开启 `requires_grad` 的 Tensor。它仍然保存数组、梯度和计算图信息；这个类型主要告诉模块：“请把我当作模型状态登记起来”。本篇把整数输入转换为浮点参数，因为整数不能接受梯度。

## Module 组合层和参数

一个 `Module` 实现 `forward()` 描述前向计算。把 `Parameter` 或另一个 `Module` 赋给它的属性时，模块会登记这个对象：

```python
class TwoLayerModel(Module):
    def __init__(self):
        super().__init__()
        self.hidden = Linear(2, 3)
        self.output = Linear(3, 1)

    def forward(self, inputs):
        return self.output(self.hidden(inputs))
```

调用 `model(inputs)` 会进入 `forward()`。`named_parameters()` 递归列出参数路径，`parameters()` 提供更新器需要的参数序列，`state_dict()` 返回可保存的参数副本。

运行[配套脚本](../代码/03-神经网络训练/11-Parameter与Module.py)：

```bash
python 'book/代码/03-神经网络训练/11-Parameter与Module.py'
```

它会列出 `hidden.weight`、`hidden.bias` 等参数路径和形状。两个线性层一共登记 13 个可训练数值。

## 为什么参数登记很重要

如果靠手写列表管理参数，新增一层时容易漏掉它的权重或偏置。模块注册让参数列表由模型结构自动生成，也为梯度清理、优化器更新和保存模型状态提供统一入口。

本篇的 `state_dict` 只保存参数，不支持 buffer、共享参数别名或任意对象状态。后续模型不需要这些高级能力；仓库的 `nnlearn.nn.Module` 还支持 buffer 等更完整的路径。

## 练习

1. 给模型增加一个输出特征为 2 的线性层，确认新参数自动出现在 `named_parameters()` 中。
2. 对模型调用 `zero_grad()`，检查每个参数的 `.grad` 是否为 `None`。
3. 把某个参数属性替换成 `None`，观察它是否从参数枚举中移除。

## 对应到仓库

参数和模块职责分别对应 `nnlearn/nn/parameter.py` 与 `nnlearn/nn/module.py`。本篇实现保留这两个核心概念，不覆盖完整 PyTorch 模块注册规则。

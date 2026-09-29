# 《自己动手实现 PyTorch》目录设计

日期：2026-09-29

## 写作目标

本书带读者从最小的标量自动微分程序出发，逐步构建一个能完成基础模型训练的小型 PyTorch 风格框架。每一阶段先提出具体问题，再给出可运行实现，最后说明实现如何映射到仓库中的 `torch_1k`。

本书面向会使用 Python 和 NumPy、接触过基础神经网络训练，希望理解 Tensor、计算图、梯度和参数更新过程的读者。无需预先了解框架内部实现。

## 递进原则

1. **先标量，后张量**：先用纯 Python 数值理解导数、局部梯度、计算图和链式法则，再引入 NumPy 数组与多维形状。
2. **先机制，后接口**：先实现前向运算和梯度，再组织成 `Tensor`、`Function`、`Module` 等框架接口。
3. **每阶段能运行**：小章节尽量形成可单独运行的代码；关键梯度用手算、有限差分或 PyTorch 对照。
4. **先训练闭环，后进阶模型**：先完成线性回归和 MLP，再进入 CNN、Transformer 与设备后端。
5. **区分教学实现与 PyTorch**：使用 PyTorch 做结果对照，不把小型实现描述为完整兼容层或性能替代品。

## 章节安排

| 部分 | 内容 | 对应现有代码入口 |
| --- | --- | --- |
| 00 阅读指南 | 本书目标、环境、代码运行方式与项目边界 | 根目录 `README.md`、`docs/tutorial/torch-training-mechanism-20260929.md` |
| 01 标量自动微分 | 数值、导数、计算图、反向传播、单神经元训练 | `torch_1k/function.py`、`tests/test_02_autograd.py`（概念映射；当前实现从 Tensor 开始） |
| 02 Tensor 与自动微分 | NumPy 数组、形状、算子、广播、归约、张量反向传播 | `torch_1k/tensor.py`、`torch_1k/functional/`、`examples/example38_beginner_tutorial.py` |
| 03 神经网络训练 | Parameter、Module、层、损失、优化器、数据加载与训练 | `torch_1k/nn/`、`torch_1k/optim/`、`torch_1k/utils/data/`、`examples/example1_linear_reg_simple.py`、`examples/example7_mnist_dataloader_train_compare.py` |
| 04 进阶模型与运行时 | CNN、Transformer、CPU/CUDA 后端及 PyTorch 对照 | `torch_1k/nn/conv.py`、`torch_1k/nn/transformer.py`、`torch_1k/backend.py`、`torch_1k/cuda.py`、`examples/example39_training_mechanism_comparison.py` |
| 附录 | 源码地图、梯度与形状速查表、术语表和问题排查 | `torch_1k/`、`examples/` |

第一部分的纯 Python 标量程序是为讲清机制准备的教学阶梯。后续章节会转向 NumPy Tensor，再逐步对应仓库实现；不要求仓库现有代码必须沿用标量程序的内部结构。

## 章节依赖

```text
Python 数值与函数
        ↓
标量导数 → 计算图 → 反向传播
        ↓
NumPy Tensor → 张量算子及其梯度
        ↓
Parameter / Module → 损失 / 优化器
        ↓
线性回归 → mini-batch → MLP
        ↓
CNN / Transformer / 设备后端
```

## 单章写作约定

每章按需要包含：本章目标、运行前提、问题与直觉、逐步实现、完整代码、运行结果、梯度或数值核对、练习，以及与 `torch_1k` 的源码对应。练习答案可在读者版书稿稳定后另行安排。

## 目录和代码组织

书稿正文按篇放在 `book/` 的编号目录；阅读指南、每篇目录说明和附录各有入口页。可独立运行的教学代码集中在 `book/代码/`，按对应篇或章节组织，避免把临时教学版本混入 `torch_1k` 包。

具体章节文件名见 `book/目录规划.md`。篇入口保留在各目录的 `README.md`；章节正文和代码随写作逐步加入，不预建空章节文件。

# torch_1k：看懂一次训练如何发生

`torch_1k` 是一个小型、可阅读的 PyTorch 风格教学实现。它把张量运算、动态计算图、反向传播和参数更新放在一套可以运行的代码里，帮助已经会使用 PyTorch 的学习者理解训练过程。

PyTorch 已经适合实际模型开发和训练。本项目的价值在于可以直接阅读和修改自动微分及训练链路的实现，并用 PyTorch 对照前向值、梯度和参数更新。

## 适合谁

- 会写基础 PyTorch 训练代码，想理解 `loss.backward()` 如何算出梯度的人。
- 想练习实现张量算子、梯度公式和优化器更新的人。
- 想用小型实现观察深度学习框架内部行为的人。

## 从这里开始

先运行最短的训练机制对照：

```bash
pip install .
python examples/example39_training_mechanism_comparison.py
```

该例子要求另外安装 PyTorch 作为参考实现。它使用相同的线性模型参数、输入数据和学习率，逐项比较前向结果、损失、权重梯度、偏置梯度和一次 SGD 更新。

接下来可以按这条路径阅读：

1. [训练机制对照教程](docs/tutorial/torch-training-mechanism-20260929.md)
2. [基础 API 新手教程](examples/example38_beginner_tutorial.py)
3. [MLP、CNN 与 Transformer 训练基线](examples/example36_pytorch_training_baseline.py)
4. [核心实现说明](docs/design/torch-1k-20260624-principles.md)

`example36` 默认用 `torch_1k`。安装 PyTorch 后，可运行 `USE_TORCH_1K=0 python examples/example36_pytorch_training_baseline.py`，用相同训练主体切换到 PyTorch。

## 包含什么

- Tensor 运算和反向模式自动微分。
- `nn.Module`、常见网络层、损失函数和优化器。
- 数据集、批次加载、模型与优化器状态保存。
- 基于 NumPy 的 CPU 路径，以及依赖 CuPy 的可选 CUDA 路径。
- 覆盖 MLP、CNN 和 Transformer 训练流程的示例。

仓库保留了一部分 PyTorch 风格 API，目的是让核心训练例子容易比较和迁移。具体 API 范围以代码和示例为准。

## 支持范围

本项目不实现完整 PyTorch API，不承诺任意 PyTorch 脚本可以直接替换运行，也不提供 PyTorch 的训练性能或内存效率。示例通过只能证明该示例覆盖的路径可运行。

CPU 路径使用 NumPy。CUDA 路径需要安装与本机 CUDA 环境匹配的 CuPy，并需要可用的 CUDA 设备；没有设备实测的结果不应视为已验证。

## 项目目的和开发边界

项目以“读者能看懂训练机制、跑通例子、修改实现并对照结果”为判断标准。新增功能应服务于教学目标、重要训练示例或已承诺链路的正确性；不以追齐 PyTorch API 数量或固定代码行数为目标。

详细说明见[项目目的与范围](docs/design/torch-20260929-project-purpose.md)和[核心路线图](docs/design/torch-3k-20260624-core-cuda-roadmap.md)。

## 安装

```bash
pip install .
```

核心依赖由 `setup.py` 声明。PyTorch 是对照教程的可选依赖，需要按本机环境单独安装：<https://pytorch.org/get-started/locally/>。

## 参考资料

- 《深度学习入门：基于 Python 的理论与实现》及相关自制框架实践。
- PyTorch 自动微分机制说明：<https://docs.pytorch.org/docs/stable/notes/autograd.html>
- Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)，标量级自动微分教学实现。

## 更新记录

- 2024-08：创建项目并实现最初的张量、自动微分和训练示例。
- 2026-06：扩展到小型训练链路、CPU/CUDA 后端、PyTorch 风格 API 和教程示例。
- 2026-09：明确项目定位为训练机制教学实现，新增同参数的 PyTorch 数值对照教程。

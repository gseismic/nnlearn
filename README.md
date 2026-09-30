# nlearn

`nlearn` 是一套独立实现的张量计算与自动微分框架，提供熟悉的 PyTorch 风格 Python API。它有自己的 `Tensor`、动态计算图、反向传播和运行时实现，不把 PyTorch 作为执行后端。

CPU 路径使用 NumPy，CUDA 路径可选使用 CuPy。代码按张量、算子、神经网络层、优化器和设备后端分层，方便按项目需要阅读、替换和扩展实现。

## 快速开始

在仓库根目录安装：

```bash
python -m pip install .
```

常用入口与 PyTorch 风格相近：

```python
import nlearn as nl
import nlearn.nn as nn
import nlearn.optim as optim

model = nn.Linear(3, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

运行一个完整的训练机制对照例子：

```bash
python examples/example39_training_mechanism_comparison.py
```

该例子要求另外安装官方 PyTorch，用相同参数和数据比较前向结果、损失、梯度与一次 SGD 更新。`nlearn` 本身不依赖 PyTorch 运行。

## 项目定位

- 独立实现张量、自动微分、常见神经网络模块、优化器、数据加载和状态保存。
- 提供熟悉的 PyTorch 风格 API，降低从现有训练代码迁移概念的成本。
- 允许开发者检查并按需要扩展算子、后端和训练组件。
- 以 NumPy CPU 路径为基础，提供可选 CuPy CUDA 路径。

本项目不实现完整 PyTorch API，也不承诺任意 PyTorch 脚本可以直接替换运行。可用范围以代码、文档和示例明确覆盖的语义为准。

## 示例与文档

1. [训练机制对照教程](docs/tutorial/torch-training-mechanism-20260929.md)
2. [基础 API 示例](examples/example38_beginner_tutorial.py)
3. [MLP、CNN 与 Transformer 训练基线](examples/example36_pytorch_training_baseline.py)
4. [核心实现说明](docs/design/nlearn-20260624-principles.md)
5. [项目目的与范围](docs/design/nlearn-20260930-project-purpose.md)
6. [CUDA 后端路线图](docs/design/nlearn-20260624-core-cuda-roadmap.md)

`example36` 默认使用 nlearn。安装 PyTorch 后，可以运行：

```bash
USE_NLEARN=0 python examples/example36_pytorch_training_baseline.py
```

同一训练主体会切换到官方 PyTorch 路径。

## 0.1.0 命名迁移

0.1.0 统一使用 `nlearn` 发行包名、导入路径和 `USE_NLEARN` 示例开关。此前版本的调用方应按本 README 中的导入示例更新项目配置；函数、类和张量语义不因名称迁移而改变。

## 安装与依赖

核心依赖由 `setup.py` 声明。CPU 路径需要 NumPy 和日志依赖；CUDA 路径需要与本机 CUDA 环境匹配的 CuPy，以及可用的 CUDA 设备。PyTorch 只用于部分对照示例，需要按本机环境单独安装：<https://pytorch.org/get-started/locally/>。

## 参考资料

- 《深度学习入门：基于 Python 的理论与实现》及相关框架实现实践。
- PyTorch 自动微分机制说明：<https://docs.pytorch.org/docs/stable/notes/autograd.html>
- Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)，标量级自动微分实现。

## 更新记录

- 2024-08：创建项目并实现最初的张量、自动微分和训练示例。
- 2026-06：扩展到训练链路、CPU/CUDA 后端、PyTorch 风格 API 和教程示例。
- 2026-09：统一项目名为 `nlearn`，发行版本更新为 `0.1.0`，明确独立张量与自动微分框架定位。

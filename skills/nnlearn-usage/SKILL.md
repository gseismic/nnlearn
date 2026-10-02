---
name: nnlearn-usage
description: 使用和排查 nnlearn 的 Tensor、自动微分、神经网络训练、数据加载与设备 API；需要编写基于 nnlearn 的 Python 代码时使用。
---

# nnlearn 用法

帮助用户用仓库里的 `nnlearn` 编写或调整可运行的张量与模型训练代码，也用于定位这类代码的用法问题。先根据任务找到最接近的教程或示例，再按需核对实现和测试中的精确 API。

## 项目边界

- `nnlearn` 是独立实现的张量和自动微分框架。常规 CPU 路径依赖 NumPy；PyTorch 不作为它的运行后端。
- 项目提供部分 PyTorch 风格的训练接口，不是完整 PyTorch 替代品。不要承诺任意 PyTorch 程序可以直接运行，也不要从单个对照示例推断全面兼容。
- CUDA 是可选路径，依赖安装了匹配 CUDA 环境的 CuPy 及可用设备。只在检查到当前环境支持时建议或声称能够运行 CUDA 代码。

## 使用流程

1. 从仓库根目录的 [`README.zh-CN.md`](../../README.zh-CN.md) 确认安装和入门入口。仓库安装命令是 `python -m pip install .`；依赖与 Python 版本要求以 [`pyproject.toml`](../../pyproject.toml) 为准。
2. 按用户场景选择最接近的材料，避免从零猜 API：
   - Tensor、自动微分和完整入门流程：[`examples/000_tutorial_beginner_tutorial.py`](../../examples/000_tutorial_beginner_tutorial.py)
   - 线性回归训练：[`examples/102_training_pytorch_compatible_train.py`](../../examples/102_training_pytorch_compatible_train.py)
   - MLP、CNN、Transformer：[`examples/103_training_mlp_train_compare.py`](../../examples/103_training_mlp_train_compare.py)、[`examples/104_training_mnist_cnn_train_compare.py`](../../examples/104_training_mnist_cnn_train_compare.py)、[`examples/105_training_transformer_train_compare.py`](../../examples/105_training_transformer_train_compare.py)
   - `DataLoader` 与批次采样：[`examples/200_data_mnist_dataloader_train_compare.py`](../../examples/200_data_mnist_dataloader_train_compare.py)、[`examples/201_data_dataloader_sampler_compare.py`](../../examples/201_data_dataloader_sampler_compare.py)
   - PyTorch 风格训练与协议范围：[`examples/700_compat_pytorch_training_baseline.py`](../../examples/700_compat_pytorch_training_baseline.py)、[`examples/703_compat_torch_protocol.py`](../../examples/703_compat_torch_protocol.py)、[`torch_protocol/README.md`](../../torch_protocol/README.md)。
3. 对于示例没有覆盖的符号、参数或行为，直接查看对应的 `nnlearn/` 实现、`torch_protocol/` 协议和相关 `tests/`，并说明哪些结论来自实现核对。若资料不能确认行为，就明确指出尚未确认，不要虚构兼容性。
4. 按用户目标给出最小完整示例、必要的运行命令和关键限制。用户要求排错时，结合错误信息和实际调用链解释原因，并优先复用仓库示例中的写法。

## 常用训练写法

以当前仓库示例为准，典型训练代码使用这些模块：

```python
import nnlearn as nl
import nnlearn.nn as nn
import nnlearn.optim as optim
from nnlearn.utils.data import DataLoader, TensorDataset
```

模型训练通常遵循：创建模型、损失函数和优化器；每个批次依次调用 `optimizer.zero_grad()`、前向计算、`loss.backward()` 和 `optimizer.step()`。评估时调用 `model.eval()`，并在 `with nl.no_grad():` 中计算预测。只摘取与用户任务有关的部分；不需要 DataLoader 时直接使用 Tensor。

若用户明确需要张量创建或梯度示例，可参考 `nl.tensor(..., requires_grad=True)`、标量损失上的 `backward()`，并从 `.grad` 读取结果。数据加载入口是 `TensorDataset` 和 `DataLoader`；采样器与批次选项以对应示例和实现为准。

## 设备和兼容性问题

- 提供最小示例时优先从 CPU 工作流开始。讨论设备迁移时，参照入门示例中的 `nl.device(...)`、`nl.cuda.is_available()` 和 `.to(device=...)` 用法；不要假设仅安装 `nnlearn` 就有 CUDA 支持。
- 用户要迁移已有 PyTorch 代码时，先列出代码实际使用的模块与操作，再逐项对照仓库公开实现或 `torch_protocol/`。把不支持或尚未核实的部分标出来，并提供所需范围内的改写方案。
- 除非用户要求，否则不要把使用咨询扩展成修改框架实现。用户若要求新增 API 或修复库本身，再按代码变更处理。

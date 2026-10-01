# Torch Protocol v1

`torch_protocol` 定义 PyTorch 风格张量训练 API 的共同子集。程序只调用本页列出的接口时，可以将 `nnlearn` 或 PyTorch 的 `torch` 模块作为后端，运行同一段代码。

协议包不导入 PyTorch 或 nnlearn。nnlearn 显式实现这些 Protocol；PyTorch 无需修改，按 Python 结构化类型满足协议。这里的 `torch` 是 PyTorch 的导入名。协议只保证列出的接口，不表示任意 PyTorch 程序都能在 nnlearn 上运行。

## 安装

可从 nnlearn 仓库根目录安装独立协议包：

```bash
python -m pip install ./torch_protocol
```

或进入本目录后安装：

```bash
python -m pip install .
```

在 nnlearn 仓库中，`torch_protocol` 也随 nnlearn 一起提供。

## v1 接口

下面的签名表示可以依赖的共同调用形式。未列出的参数即使某个后端支持，也不属于 v1 保证范围。

| 命名空间 | 共同调用形式 |
| --- | --- |
| 根模块 | `Tensor`、`tensor(data, *, dtype=None, device=None, requires_grad=False)`、`float32`、`manual_seed(seed)`、`no_grad()` |
| `nn` | `Module`、`Linear(in_features, out_features, bias=True)`、`ReLU()`、`Sequential(*modules)`、`MSELoss(reduction="mean")`，reduction 可取 `"none"`、`"mean"` 或 `"sum"` |
| `optim` | `SGD(params, lr)` |
| `utils.data` | `TensorDataset(*tensors)`、`DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)` |

`Tensor` 是后端的实际张量类型，可用于 `isinstance(value, backend.Tensor)`。跨后端创建张量统一调用 `backend.tensor(...)`，不要直接调用 `Tensor(...)` 构造器。

张量对象实现 `TensorProtocol`，包含以下接口：

| 类别 | 成员 |
| --- | --- |
| 元信息 | `shape`、`ndim`、`dtype`、`device`、`requires_grad`、`grad` |
| 运算符 | `+`、`-`、`*`、`/`、`**`、`@`、索引 `[]` |
| 运算方法 | `reshape(*shape)`、`sum(dim=None, keepdim=False)`、`mean(dim=None, keepdim=False)`、`clone()` |
| 梯度与转换 | 标量 `backward()`、`detach()`、`to(*, device=None, dtype=None)`、`cpu()`、`numpy()`、`tolist()`、`item()`、`size(dim=None)` |

`Module` 是可调用模型的基类，支持 `__call__(...)`、`parameters()`、`train(mode=True)`、`eval()`、`to(*, device=None, dtype=None)`、`state_dict()` 和 `load_state_dict(state_dict)`。`MSELoss` 是接收预测与目标并返回张量的可调用对象。创建 `SGD` 时应显式提供 `lr`。优化器支持 `zero_grad()` 与 `step()`；显式传入 `set_to_none=True` 或 `False` 时，两边都接受该选项。数据集支持 `len(dataset)` 和 `dataset[index]`；数据加载器支持 `len(loader)` 与批次迭代。

## 训练示例

示例只使用 v1 接口。传入 `nnlearn` 或 PyTorch 模块即可切换后端：

```python
import nnlearn as backend
# 使用 PyTorch 时，改成：
# import torch as backend
# import torch.nn
# import torch.optim
# import torch.utils.data

from torch_protocol import TorchBackendProtocol, validate_backend

backend: TorchBackendProtocol = validate_backend(backend)
backend.manual_seed(0)

x = backend.tensor([[0.0], [1.0], [2.0]], dtype=backend.float32)
y = backend.tensor([[1.0], [3.0], [5.0]], dtype=backend.float32)
model = backend.nn.Linear(1, 1)
loss_fn = backend.nn.MSELoss()
optimizer = backend.optim.SGD(model.parameters(), lr=0.05)

for _ in range(20):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with backend.no_grad():
    predictions = model(x)
```

仓库中的 [`703_compat_torch_protocol.py`](../examples/703_compat_torch_protocol.py) 展示了包含 `TensorDataset` 与 `DataLoader` 的完整训练流程。可分别运行：

```bash
python examples/703_compat_torch_protocol.py
TORCH_BACKEND=torch python examples/703_compat_torch_protocol.py
```

使用第二个命令前需要安装 PyTorch。两种后端的随机数序列、参数初值和浮点结果不保证相同。

## 可移植代码的语义约定

- 创建张量时显式使用 `dtype=backend.float32` 等后端类型。未指定类型时，后端可能采用不同的推断默认值。
- `device` 和 `dtype` 对象由后端提供。v1 代码不假设 PyTorch 与 nnlearn 的设备对象可互换；需要最高可移植性时使用默认 CPU 设备。
- `backward()` 用于只有一个元素的损失张量。梯度会累积；更新前调用 `optimizer.zero_grad()`。不依赖清零后 `parameter.grad` 是 `None` 还是零张量。
- `no_grad()` 暂停上下文中的梯度记录。推理完成后仍可用 `detach().cpu().numpy()` 导出 NumPy 数组。
- `TensorDataset` 的张量首维长度必须相同；`DataLoader` 将样本按 `batch_size` 组合，支持 `shuffle` 和 `drop_last`。v1 只要求单进程批次迭代。
- 模型状态可以用 `state_dict()` 和 `load_state_dict()` 在同一后端内保存和恢复。v1 不保证一个后端生成的状态字典能直接加载到另一个后端。
- `manual_seed(seed)` 初始化后端随机数状态，但不承诺不同后端产生相同随机数。

`validate_backend()` 检查必需入口是否存在且可调用，不验证运算结果、梯度或其他语义。协议不包含完整 PyTorch API、多进程 DataLoader、性能保证或任意脚本替换。

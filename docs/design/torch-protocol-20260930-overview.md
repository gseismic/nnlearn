# Torch Protocol v1 设计

日期：2026-09-30

## 目标

提供一个独立、可安装的 `torch_protocol` Python 包，明确一组跨后端的 Python 张量与训练接口。使用方只依赖协议列出的公共成员，可以把 PyTorch 的 `torch` 模块或 `nnlearn` 模块作为后端传入同一段程序。nnlearn 的核心类型显式实现协议；PyTorch 通过 Python 结构化类型满足协议，不需要修改 PyTorch，也不需要让协议包依赖任一后端。

这里的“torch”指 PyTorch 的 Python 导入名 `torch`。协议定义共同支持面，不声称任意 PyTorch 程序都能在 nnlearn 上运行。

## 代表性场景

1. 创建张量、执行算术和矩阵乘法，并计算标量损失的梯度。
2. 使用 `nn.Module`、`nn.Linear` 和损失函数定义及训练模型。
3. 使用 `optim.SGD` 更新参数，并在推理时关闭梯度记录。
4. 使用 `TensorDataset` 和 `DataLoader` 以小批次读取数据。
5. 不改动训练主体，仅替换传入的后端模块，在 PyTorch 与 nnlearn 上运行。

## 包与依赖方向

```text
应用程序 ──依赖接口──> torch_protocol
                         ▲       ▲
                         │       │
                      nnlearn   PyTorch/torch
```

- `torch_protocol` 只依赖 Python 标准库 `typing`、`contextlib`，不导入 `nnlearn` 或 `torch`。
- nnlearn 的 Tensor、Module、Optimizer 和数据集类型显式继承相应 Protocol，并在包初始化时检查顶层后端入口。
- PyTorch 不修改、不 monkey-patch；`torch` 模块及其对象按结构满足协议。`validate_backend()` 用于检查必需成员是否存在，完整语义由共用契约测试验证。
- 协议包可单独构建和安装。当前 nnlearn 仓库作为 monorepo 将它一并纳入 nnlearn 发行包，避免依赖未发布到公共索引的包；未来拆分发行时保持 `import torch_protocol` 不变。

## v1 后端接口

| 层级 | 必需成员 | 用途 |
| --- | --- | --- |
| 根模块 | `Tensor`、`tensor`、`float32`、`manual_seed`、`no_grad` | 创建张量、选择常用类型、初始化随机数、推理上下文 |
| `nn` | `Module`、`Linear`、`ReLU`、`Sequential`、`MSELoss` | 定义可训练模型和均方误差 |
| `optim` | `SGD` | 以参数迭代器和学习率创建优化器 |
| `utils.data` | `TensorDataset`、`DataLoader` | 张量数据集和单进程批次遍历 |

根模块的 `Tensor` 是后端张量类型，可用于 `isinstance(value, backend.Tensor)`；跨后端创建张量统一调用 `backend.tensor(...)`，不直接调用两边构造行为不同的 `Tensor(...)`。

共同的构造调用形式为：

```text
tensor(data, *, dtype=None, device=None, requires_grad=False)
nn.Linear(in_features, out_features, bias=True)
nn.ReLU()
nn.Sequential(*modules)
nn.MSELoss(reduction="mean")  # reduction: "none" | "mean" | "sum"
optim.SGD(params, lr)
utils.data.TensorDataset(*tensors)
utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
```

跨后端训练代码应显式传入 `dtype=backend.float32` 和 `lr`。PyTorch 与 nnlearn 的浮点推断默认值、优化器默认参数并不完全相同。

张量与模块的设备转换使用共同形式 `to(*, device=None, dtype=None)`，传入的 device/dtype 值仍由当前后端提供。跨后端代码应使用关键字传参。

### TensorProtocol

- 元信息：`shape`、`ndim`、`dtype`、`device`、`requires_grad`、`grad`。
- 运算：加、减、乘、除、乘方、矩阵乘法；`reshape`、`sum`、`mean`。
- 自动微分与转换：标量 `backward()`、`detach()`、`to()`、`cpu()`、`numpy()`、`tolist()`、`item()`。
- dtype 与 device 对象由后端持有。跨后端代码使用 `backend.float32` 等常量，不假设 NumPy 与 PyTorch dtype 对象相同；不直接依赖 `.data`。

### ModuleProtocol、LossProtocol 和 OptimizerProtocol

- `ModuleProtocol`：`__call__`、`parameters()`、`train(mode=True)`、`eval()`、`to()`、`state_dict()`、`load_state_dict()`。
- `LossProtocol`：可调用，输入预测与目标并返回可反向传播的 Tensor。nnlearn 的损失对象目前不是 `nn.Module` 子类，协议因此不要求损失对象具有模块生命周期方法。
- `OptimizerProtocol`：`zero_grad()` 和 `step()`。后端可将清零后的梯度保存为 `None` 或零张量；协议代码不能依赖其具体表示。
- `backward()` 的可移植用法仅针对单元素损失张量，不传入显式梯度。梯度会累积，优化器更新前清零。

### DatasetProtocol 与 DataLoaderProtocol

- Dataset 提供 `__len__` 和按索引读取样本。
- DataLoader 支持 `dataset`、`batch_size`、`shuffle` 和 `drop_last`，返回由张量组成的批次。
- v1 只承诺单进程批次遍历，不包括 worker、多进程、pin memory、sampler 扩展和自定义 batch sampler。
- `Module.state_dict()` 与 `load_state_dict()` 保证在同一后端内保存和恢复；不保证一端生成的状态能直接加载到另一端。

## 兼容规则与范围

- 只有调用 v1 表列出的成员并遵守上述语义的程序，才属于跨后端协议程序。
- nnlearn 现有其他接口可继续使用，但未列入 v1 的接口是后端扩展，不因此获得 PyTorch 兼容承诺。
- 协议不承诺参数初始化随机序列一致、浮点逐位一致、设备完全相同或训练性能一致；数值测试使用容差。
- `validate_backend()` 只验证入口成员存在且可调用，不代替数值与行为测试。
- 新增兼容成员使用向后兼容方式演进；删除或改变 v1 必需语义需要新主版本协议。

## PyTorch 官方接口参考

- [`torch.tensor`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html)
- [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [`torch.nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [`torch.nn.MSELoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
- [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- [`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
- [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)

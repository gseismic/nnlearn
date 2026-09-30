# 20 CPU 与 CUDA 后端

Tensor 保存数据，也需要知道这些数据由哪个数组库管理。CPU 路径通常使用 NumPy；CUDA 路径可以使用 CuPy。后端负责选择数组模块、创建目标设备上的数据，并在需要时搬运数组。

## 运算要分发到数据所在的后端

如果把数组模块写死为 `numpy`，CUDA 数组就无法直接参与运算。一个简单的分发规则是：根据设备选择 `xp`，然后调用 `xp.asarray`、`xp.matmul` 等接口。NumPy 和 CuPy 提供相似的数组操作，因此同一段前向代码可以运行在不同设备上。

设备搬运应当显式发生。两个数组处于不同设备时，不能假定矩阵乘法会自动复制数据；先把其中一个搬到另一个设备，才开始计算。CPU 与 GPU 之间的复制也会消耗时间，频繁往返可能抵消并行计算的收益。

## 运行设备示例

运行[配套脚本](../代码/04-进阶模型与运行时/20-CPU与CUDA后端.py)：

```bash
python 'book/代码/04-进阶模型与运行时/20-CPU与CUDA后端.py'
```

脚本先用 NumPy 完成矩阵乘法；当 CuPy 和可用 CUDA 设备都存在时，再将输入搬到 GPU 计算并搬回 CPU 比较。当前写作环境没有安装 CuPy，因此实际输出为：

```text
CPU 输入设备: cpu
CPU 矩阵乘法结果: [0.0, 2.0]
CUDA/CuPy 不可用，跳过可选设备路径。
```

CUDA 路径需要安装与本机 CUDA 版本匹配的 CuPy，并具备可用的 CUDA 设备。本书不会在 CPU 环境中安装或模拟 GPU 驱动。

## 数据搬运与自动微分

本篇的 `DeviceArray` 只演示数组存储、后端分发、同设备运算和显式 `.to()`，它没有自动微分。若要让设备转换保留计算图，反向传播时必须把梯度送回源设备。仓库中的 [`Tensor.to()`](../../nlearn/tensor.py) 通过 [`CopyTo`](../../nlearn/functional/device.py) 记录这条转换边；底层 NumPy/CuPy 选择可见于 [`backend.py`](../../nlearn/backend.py)。

即使底层都使用数组，也要留意数据类型、同步时机和数据传输；支持多 GPU 时还要明确设备索引。CUDA 运算可能异步执行；计时前后需要正确同步，否则测到的可能只是提交任务的时间。

## 练习

1. 在有 CUDA 的机器上运行脚本，确认 GPU 结果搬回 CPU 后与 NumPy 结果一致。
2. 给 `DeviceArray` 增加 `sum()`，并保证结果仍保留原设备标记。
3. 阅读仓库中的 `CopyTo.backward()`，解释为什么返回梯度时要复制回源设备。

## 对应到仓库

仓库的 [`nlearn/backend.py`](../../nlearn/backend.py) 负责识别 NumPy/CuPy 数组、设备转换和 CUDA 可用性；[`nlearn/cuda.py`](../../nlearn/cuda.py) 提供公开查询入口。本章的封装仅保留最小数据路径，不覆盖完整张量后端。

# CUDA 安装与实机验收

nnlearn 用 CuPy 数组执行 CUDA 计算，用 NumPy 数组执行 CPU 计算。PyTorch 仅用于测试对照，训练框架本身不依赖 PyTorch。

## 选择兼容环境

先运行 `nvidia-smi` 和 `nvcc --version`：前者显示驱动支持的 CUDA 上限，后者显示实际 Toolkit。安装包必须匹配可用运行时，不能仅按 `nvidia-smi` 显示的数字选择。

[CuPy 13.6 官方安装文档](https://docs.cupy.dev/en/v13.6.0/install.html)列明：CUDA 11.2–11.8 使用 `cupy-cuda11x`，CUDA 12.x 使用 `cupy-cuda12x`；Python 支持 3.9–3.13。每个环境只装一个 CuPy 发行包。CUDA 12.2 及以上还需要匹配的 CUDA Runtime 头文件，详情见该文档的 NVRTC 错误排查说明。

本次机器的驱动支持 CUDA 12.4，但系统 Toolkit 是 11.5，因此使用 `cupy-cuda11x==13.6.0`，无需修改驱动。环境准备采用 `uv`，所有新依赖放进隔离环境。以下命令中的 Python 路径替换为本机已有、包含 NumPy/pytest/loguru/PyTorch 的解释器即可：

```bash
uv venv --system-site-packages --python /home/lsl/miniconda3/bin/python /tmp/nnlearn-cuda-validation-venv
UV_HTTP_TIMEOUT=180 uv pip install --python /tmp/nnlearn-cuda-validation-venv/bin/python 'numpy==2.2.6' 'fastrlock==0.8.4' 'cupy-cuda11x==13.6.0'
```

`--system-site-packages` 只复用已有测试依赖，新安装的 CuPy 不会写入原环境。首次下载 wheel 或首次编译 CUDA kernel 较慢；网络超时后先确认进程已退出，再重试同一命令。

## 执行硬件门禁

在项目根目录执行：

```bash
NNLEARN_REQUIRE_CUDA=1 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python -m pytest tests/test_55_cuda_training.py -q -rs
```

测试先实际分配 GPU 数组、编译 kernel、执行矩阵乘法并同步。缺失 CuPy、不可见 GPU、无法运行 PyTorch CUDA 对照时，普通 CPU 作业会明确显示跳过原因；设置 `NNLEARN_REQUIRE_CUDA=1` 后这些条件会使作业失败。已经发现 GPU 后的 kernel 或动态库错误始终失败。

测试覆盖：

- CPU→CUDA→CPU 可微复制与一、二阶梯度；Module 参数、已有梯度、子模块、浮点/整数/布尔 buffer 的迁移；优化器持有的 Parameter 引用保持不变。
- MLP、包含 Conv2d/MaxPool2d 的 CNN、包含重复 token Embedding 的 Transformer：8 步连续训练，逐步核对 CPU/CUDA 输出、损失、输入/参数梯度和更新，且损失下降。
- 三类模型的 PyTorch CUDA 前向和全部参数梯度对照，覆盖 float32/float64；causal 与 padding 掩码的布尔/浮点形式及梯度隔离。
- SGD、MomentumSGD、Adam、AdamW 的 CPU/CUDA/PyTorch 连续更新；CPU/CUDA 检查点写入文件再恢复到 CUDA，继续训练与不中断结果一致。
- Conv2d、LayerNorm、CrossEntropy 的 CUDA Hessian 向量积；float16 的 70000 类 Softmax/交叉熵及大方差 LayerNorm。
- Max/Min/TopK/Sort 的三维多轴前后向，以及空 TopK 的完整零梯度。

测试同时约束 dtype、物理数组后端和有限值，不能用 CPU 回退替代 CUDA 验收。数值容差按测试目的设置：训练路径考虑 GPU 规约顺序，高阶 float64 路径使用更严格容差，大类别半精度梯度使用足够小的绝对误差以发现下溢。

三模型 CPU/CUDA 多步参数对照使用 `Adam(lr=0.03, eps=1e-4)`。Transformer 的 key 投影偏置具有数学零梯度，浮点规约约 `1e-9` 的舍入会被很小的 Adam 分母放大；显式设置 eps 使该测试能约束各参数的连续更新。四优化器的固定非零梯度 PyTorch 对照及检查点恢复使用默认 eps。

## 已执行的实机记录

2026-10-25 已在以下环境执行：

| 项目 | 版本 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060，12 GiB，sm_86 |
| 驱动 / 系统 CUDA | 550.144.03 / Toolkit 11.5、NVRTC 11.5 |
| Python / NumPy / CuPy | 3.12.8 / 2.2.6 / 13.6.0 |
| PyTorch / pytest | 2.6.0+cu124 / 9.0.2 |

CuPy kernel、GPU 内存、cuBLAS 矩阵乘法和同步已实际通过，完整硬件门禁 **72 项全部通过、没有跳过**。半精度 CPU 回归 38 项、高阶导数 CPU 回归 33 项也在该 CuPy 环境通过。

PyTorch 参考明确使用原生 CUDA 卷积。现有系统 Python 的 cuDNN 包元数据为 9.1.0.70，但实际头文件和加载版本为 9.20，独立 cuDNN 卷积报初始化错误；原生 CUDA 卷积前后向正常。参考子进程同样固定原生 CUDA，保留真实 GPU 训练。

全量实机发现旧代码对 CuPy 的 `put_along_axis` 支持有误，已接入原生 GPU 兼容散射并用 Max/Min/TopK/Sort 多轴及空索引回归验证。

最终启用 `NNLEARN_REQUIRE_CUDA=1` 的完整回归 **947 项全部通过、没有跳过**，执行命令：

```bash
NNLEARN_REQUIRE_CUDA=1 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python -m pytest tests -q -rs
```

掩码 Transformer 示例也已在 CUDA 上完成 220 步：训练损失从 `0.63198376` 降至 `0.00011298`，验证损失 `0.00027135`，padding 和未来位置扰动的输出误差均为 0。可复现命令：

```bash
uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python examples/106_training_masked_transformer_compare.py --device cuda
```

将 `CUDA_VISIBLE_DEVICES` 设为空后，已验证普通运行明确报告 72 项跳过，强制硬件门禁非零退出。完整明细见 [PLAN-067 结果](dev/PLAN-067-cuda-validation-OUTCOME.md)。

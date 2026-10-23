# PLAN-062 实施结果：常用训练运算的精度一致性修复

日期：2026-10-24。

## 完成情况

已修复 Python 常量、反向公式、均值与极值规约及优化器零维更新中的精度传递问题。新增 292 项回归，CPU 全量 **702 passed**，原有 410 项全部通过；NumPy 1.24.4 的独立兼容回归 **25 passed**。

四条验收路径在零维和矩阵 float32 参数上均保持以下结果，SGD、MomentumSGD、Adam、AdamW 连续八步训练及中途恢复状态后继续训练均通过。

| 路径 | 输出 | 参数梯度 | 更新后参数 | 优化器浮点状态 |
| --- | --- | --- | --- | --- |
| x.mean() | float32 | float32 | float32 | float32 |
| (x * 2).sum() | float32 | float32 | float32 | float32 |
| (x ** 2).sum() | float32 | float32 | float32 | float32 |
| x.sigmoid().sum() | float32 | float32 | float32 | float32 |

SGD 本身没有浮点状态；其余三种优化器检查内部数组和 state_dict 中的 Tensor。

## 实际变更

1. **标量精度契约**：`nnlearn/function.py:18` 在浮点 Tensor 上下文中将原生 Python int/float 常量包装为同精度、不求导的 Tensor；多浮点输入选择公共精度。bool、显式 Tensor、NumPy 标量/数组保留原有类型或推断，独立 Tensor 构造默认值不变。
2. **可微的回传转换**：`nnlearn/tensor.py:89` 在每条浮点回传边汇总前将梯度恢复输入 dtype，使用 Tensor.to 保留高阶求导。`nnlearn/function.py:65` 建图时保存输入 dtype，兼容 linear 释放中间 data。
3. **规约与激活**：`nnlearn/functional/matrix.py:712` 引入直接求均值的 Mean 节点，复用可微的轴恢复与广播；半精度梯度在 float32 中归一化。`nnlearn/functional/numeric.py:16` 为幂指数、sigmoid、ReLU、clamp 内部前向常量明确 dtype；反向中的 Python 系数通过共用入参规则保持精度。
4. **极值反向**：`nnlearn/functional/numeric.py:30` 将掩码计数精度与上游梯度相乘分开处理，半精度计数使用 float32。max/min 的按轴单位置分配、全局平分，以及 amax/amin、maximum/minimum 的重复位置分配规则不变；Tensor 运算保留非线性上游的高阶依赖。
5. **优化器更新系数**：`nnlearn/optim/optimizer.py:112` 提供内部系数转换，SGD、MomentumSGD、Adam、AdamW 在参数设备和 dtype 中更新。Adam 的 1-beta 和偏差修正仍先按原超参数计算，再转换系数；保留原公式。非浮点参数沿用原行为，避免整数学习率被截断。

## 回归与复现

- 首版 246 项回归在修复前为 **169 failed, 77 passed**，修复后全部通过。失败证据包括 float32 前向、梯度、参数升精度，显式混合精度回传不一致，以及极值高阶依赖缺失。
- 复查补充大尺寸半精度极值、float32 数组指数高阶、零维内部常量等用例，`tests/test_50_training_dtype_consistency.py` 最终共 **267 项**。包含三种浮点精度、标量正反序算术、规约轴与 keepdim、激活、重复极值、显式零维高精度 Tensor、NumPy 类型、混合输入交叉二阶导、三阶导、重复反向及八步训练的 PyTorch 对照。
- NumPy 1.24.4 复查确认零维 sigmoid/ReLU 输出以及优化器参数、状态仍会升为 float64。实施前先更新设计和计划，再补齐内部常量及更新系数的显式精度。修复后的独立脚本通过 144 组八步训练检查。
- 新增不依赖 PyTorch 的 `tests/test_51_dtype_backend_compat.py`，**25 项**同时在 NumPy 1.24.4 和 2.2.6 下通过。检查组合路径的解析梯度、八步训练、中途状态恢复、内部状态及序列化精度，并约束整数优化器的历史提升行为。
- 数组指数的 0**0 二阶导使用解析值零验收；PyTorch 此处产生 NaN，其余非零指数位置继续对照 PyTorch。保留 PLAN-061 已实现的零指数保护。

## 验证命令与结果

- `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest -q`：**702 passed in 13.68s**，无警告；Python 3.12、NumPy 2.2.6、PyTorch 2.6.0+cu124。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project --python 3.8 --offline --with numpy==1.24.4 --with loguru --with pytest python -m pytest tests/test_51_dtype_backend_compat.py -q`：**25 passed in 0.39s**。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python examples/704_compat_training_performance_benchmark.py --device cpu --steps 50 --warmup 2 --repeats 1`：通过。两端 dtype 均为 float32，loss 均约 0.000751，accuracy 均为 1.0；loss 差值 1.74623e-10。本次验证训练闭环，未将并行测试期间的耗时作为稳定性能结论。
- `ruff check tests/test_50_training_dtype_consistency.py tests/test_51_dtype_backend_compat.py`：通过。
- `git diff --check`：通过。
- 未屏蔽设备的检查确认未安装 CuPy，`nnlearn.cuda.is_available()` 为 False，CUDA 未实测。

## 代码复查

- 区分原生 Python 常量与显式 Tensor，保留 bool 条件、独立构造和显式混合精度；未加入完整 PyTorch 类型提升系统。
- 梯度在每条边汇总前对齐 dtype，转换处于 create_graph 上下文中；首次克隆、分支累积及重复反向仍按原规则运行。
- 均值没有先求半精度 sum，大尺寸均值和极值计数都在溢出发生前使用适当的中间精度。100000 个相同半精度值的均值与平分梯度保持有限且非零。
- 原有 MSE 半精度与数组指数 21 项回归全部通过；整数和布尔类型系统未扩展，优化器对非浮点系数转换明确跳过。

## 交付文件

- 设计：`docs/design/dtype-20261024-training-consistency.md`。
- 计划：`docs/dev/PLAN-062-training-dtype-consistency.md`。
- 结果：`docs/dev/PLAN-062-training-dtype-consistency-OUTCOME.md`。
- 历史索引：`docs/dev/INDEX.md`。
- 实现：`nnlearn/function.py`、`nnlearn/tensor.py`、`nnlearn/functional/matrix.py`、`nnlearn/functional/numeric.py`、`nnlearn/optim/optimizer.py`、`nnlearn/optim/sgd.py`、`nnlearn/optim/momentum_sgd.py`、`nnlearn/optim/adam.py`。
- 测试：`tests/test_50_training_dtype_consistency.py`、`tests/test_51_dtype_backend_compat.py`。

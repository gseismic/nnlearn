# 训练运算的精度一致性

日期：2026-10-24。对应计划：`docs/dev/PLAN-062-training-dtype-consistency.md`。

## 问题与目标

`nnlearn/function.py` 在调用前将所有常量独立包装为 Tensor，Python int/float 因此成为 int64/float64 的零维数组。`nnlearn/functional/matrix.py` 的 mean 通过 sum/count 实现；`nnlearn/functional/numeric.py` 的幂、sigmoid、tanh、sqrt 反向也使用 Python 常量。这些路径会将 float32 输出或梯度提升为 float64，随后优化器的数组运算将参数和动量状态一并提升。

修复应使用户选择的浮点精度贯穿前向、反向和连续训练，同时保留显式 Tensor 混合精度以及可微转换的高阶梯度。

## 方案比较

| 方案 | 优点 | 问题 |
| --- | --- | --- |
| 修改 Tensor 构造器的默认 dtype | 实现入口集中 | 改变所有独立张量的推断，仍不能表达常量随 float16/float64 输入变化 |
| 在各反向公式和优化器末尾单独转换 | 容易局部修补 | 容易遗漏前向、函数式调用和后续新增公式，优化器转换不能修复错误的中间精度 |
| 在 Function 入参处识别 Python 常量，并约束浮点输入的回传梯度 | 正向与反向共享规则，保留显式 Tensor 的 dtype，转换支持高阶求导 | 仍需单独处理均值累加及极值计数，不能用末尾转换补救中间溢出 |

采用第三种方案，并修复规约中的中间计算。

## 精度契约

1. Function 实参中存在浮点 Tensor 时，原生 Python int/float 常量使用这些浮点 Tensor 的公共 dtype 包装，多个浮点 Tensor 使用 NumPy result_type 选择公共浮点精度。常量不需要梯度，设备与浮点上下文一致；存在 CUDA 浮点输入时常量在 CUDA 创建。
2. bool 保留布尔类型，避免破坏 where 条件。NumPy 标量、数组、列表以及已经构造的 Tensor 保持原有 dtype 或构造推断；不能仅凭零维形状将显式 Tensor 当作弱类型常量。无浮点 Tensor 上下文时不改变现有规则。
3. 显式 Tensor 之间仍由 NumPy/CuPy 执行类型提升，包括显式 float64 零维 Tensor。此次不实现完整 PyTorch 类型提升，不改变 Tensor 构造器默认值。
4. 每条回传边的梯度在汇总前恢复对应浮点输入的 dtype，使用 Tensor.to 的可微转换保留 create_graph 依赖；整数、布尔输入的历史行为不在此次范围内。Function 建图时缓存输入 dtype，使 linear 释放中间 data 后仍可恢复精度。不能仅在优化器中强制转换参数掩盖前向错误。
5. sum 保留浮点输入 dtype。mean 使用后端直接求均值，float16 梯度使用 float32 归一化后再在回传边恢复 float16，避免大张量先 sum 溢出和分母转半精度溢出。轴、keepdims、广播恢复沿用可微的规约反向。
6. 极值规约的掩码与计数使用可控精度，并通过 Tensor 运算应用上游梯度；保留 max/min 的按轴单位置分配、全局及 amax/amin 的平分规则。逐元素 maximum/minimum 的平分系数也使用 Tensor 运算。
7. SGD、MomentumSGD、Adam、AdamW 的更新公式保持不变，参与数组运算的超参数系数显式转换为参数设备及 dtype。Adam 的 1-beta 和偏差修正系数先按原超参数计算，再转换，保留原公式的舍入次序。不能依赖 NumPy 2.x 的弱类型标量行为保证零维参数精度。
8. 数值 Function 内部前向的 Python 常量也使用浮点上下文构造，覆盖幂指数、sigmoid、ReLU 和 clamp 边界，避免 NumPy 1.x/CuPy 的零维提升差异。

## 验证与复查

- 对照 PyTorch 验证基础算术、正反序标量、sum/mean、激活和极值的输出、梯度数值；显式混合精度的前向 dtype 按 NumPy/CuPy 规则验证。
- 覆盖 float16/float32/float64、零维/矩阵、广播、多轴规约、梯度累积，以及二阶或三阶导数。
- 大尺寸 float16 均值必须保持有限输出和非零梯度；不得回退 PLAN-061 的 MSE 与数组指数保护。
- 连续多步训练逐步检查输出、梯度、参数、优化器内部状态及 state_dict 的 dtype 和数值，并验证状态恢复后继续训练。
- 自查一：保留 bool 条件、显式零维 Tensor、NumPy 标量和默认构造行为，避免过度套用常量规则。
- 自查二：在梯度汇总前转换，每条边均满足输入精度；mean 与计数的中间溢出需要在发生之前处理，可微转换不得替换成裸数组 astype。
- 兼容复查：NumPy 1.24.4 实测发现零维 sigmoid/ReLU 前向和优化器参数、状态仍提升到 float64，因此对内部常量和优化器系数显式规范精度；验证同时覆盖 NumPy 1.x 与 2.x。

# PLAN-065 实施结果：Transformer 掩码与变长序列训练

日期：2026-10-25。

## 完成情况

已为注意力增加 `attn_mask`、`key_padding_mask`、`is_causal`，编码层增加 `src_mask`、`src_key_padding_mask`、`is_causal` 并逐项转发。布尔 True 表示屏蔽，浮点偏置在 softmax 前相加，支持共享二维和按样本/头区分的三维掩码、自注意力与不同 query/key 长度的交叉注意力、两种 batch_first 布局。

原有默认调用、默认 batch_first、第 4 个位置参数 need_weights，以及 False 返回单 Tensor / True 返回输出和跨头平均权重的形式保持不变。`is_causal=True` 实际生成并合并因果限制，矩形序列左上对齐；这点在设计与使用教程中明确区别于 PyTorch 优化提示的用法。

掩码检查 Tensor 类型、精确形状、浮点/bool dtype 和设备；两种显式掩码同传时要求 dtype 一致，浮点掩码须与 query 相同。拒绝整数、复数、NaN、正无穷；拒绝 key 空序列，以及合并后任意样本/头/query 全部 key 被屏蔽的情况。不会通过全 -inf 的 softmax 行静默产生 NaN。浮点掩码保留求导路径，布尔掩码创建新的偏置数组，调用不原地修改用户输入。

## 数值与梯度验证

`tests/test_54_transformer_masks.py` 共 102 项：当前 CPU 环境 **101 passed, 1 skipped**，跳过项需要 CuPy/CUDA。

- 56 组注意力对照：float32/float64、两种布局、bool/浮点、二维/三维、padding/因果/显式限制及其组合。对照输出、跨头平均权重、独立 q/k/v 输入、八个投影参数和浮点掩码梯度；交叉注意力目标长度 3、源长度 5。
- 4 组编码层对照：两种布局和两种掩码 dtype，对照输出、输入、自注意力全部参数、两个前馈 Linear 和两个 LayerNorm 的全部参数梯度，覆盖共享输入上的多分支梯度汇总。
- 旧调用返回契约、padding key/value 扰动不改变输出、屏蔽输入梯度为零、未来/补齐 token 不泄漏到前缀输出与梯度。
- 全屏蔽二维/三维/纯 padding/合并/因果行、错误形状、dtype、输入 rank、维数、批次、key/value 长度与构造参数的错误语义。

对照使用相同参数和随机上游梯度；float64 容差 5e-10，float32 容差 4e-5。浮点掩码的负无穷位置仍对照梯度零，输入与原掩码内容逐项检查保持不变。

## 实际序列训练

新增 `examples/106_training_masked_transformer_compare.py`，可用 USE_NNLEARN 切换后端。无需下载数据，本地生成 30 条长度 2～6 的传感器观测；输入包含当前观测和固定采样时刻，每个有效位置预测其前缀累计均值。padding + causal 限制贯穿整个训练，损失只统计 120 个有效位置，连续 220 步 Adam 优化后再用独立序列验证。

| 后端 | 初始训练 MSE | 最终训练 MSE | 独立验证 MSE | padding 扰动最大误差 | 未来扰动最大误差 |
| --- | --- | --- | --- | --- | --- |
| nnlearn | 0.63198382 | 0.00011298 | 0.00027135 | 0 | 0 |
| PyTorch | 0.42462769 | 0.00006847 | 0.00011980 | 0 | 0 |

两端数据相同，各自随机初始化；该示例验证实际训练和泛化，严格相同参数的数值对照由上述回归负责。默认断言训练 MSE < 0.008 且低于初值 10%、验证 MSE < 0.02、两种扰动误差 < 1e-5，全部满足。

## 命令与环境

- `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest tests/test_54_transformer_masks.py tests/test_13_cnn_transformer.py -q`：**104 passed, 1 skipped**；Python 3.12、NumPy 2.2.6、PyTorch 2.6.0+cu124。包含 3 项原有 CNN/Transformer 回归。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python examples/106_training_masked_transformer_compare.py`：通过，指标见上表。
- `CUDA_VISIBLE_DEVICES='' USE_NNLEARN=0 uv run --no-project python examples/106_training_masked_transformer_compare.py`：通过，指标见上表。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project --python 3.8 --offline --with numpy==1.24.4 --with loguru --with pytest python -m pytest tests/test_54_transformer_masks.py -q`：**41 passed, 61 skipped**。60 项 PyTorch 对照因未安装 PyTorch 显式跳过，1 项 CUDA 显式跳过；其余独立边界和无泄漏验证全部通过。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project --python 3.8 --offline --with numpy==1.24.4 --with loguru python examples/106_training_masked_transformer_compare.py`：通过，训练/验证结果与 NumPy 2.2.6 相同，证明不依赖 PyTorch 可完成训练。
- `ruff check nnlearn/nn/transformer.py tests/test_54_transformer_masks.py examples/106_training_masked_transformer_compare.py` 和 `git diff --check`：通过。

## 代码 review

- 检查 query/key/value 的精确批次、特征、源长度、浮点 dtype 和设备，避免默认广播造成跨样本语义错误。
- 检查二维、三维、padding 及因果偏置的 reshape 和广播轴，确认 batch_first 不改变掩码形状或权重返回布局。
- 检查浮点掩码未经 data 包装、copy 或 NumPy 重建，梯度经过 reshape 和相加回到原掩码；布尔掩码从独立数组生成偏置。
- 检查全部屏蔽判断在三种限制合并之后、softmax 之前；key padding 不自动清零 query 输出，教程明确要求有效位置损失和右侧 padding。
- 检查无掩码路径仅增加输入校验，保留现有计算公式、构造默认值和 need_weights 参数顺序。
- 本次未遇到需要修改其他并行任务所负责文件的底层问题。CUDA 实测由单独的 CUDA 验收任务记录；当前结果不将跳过算作通过。

## 交付文件

- 设计：`docs/design/attention-20261025-masks.md`。
- 计划：`docs/dev/PLAN-065-transformer-masks.md`。
- 结果：`docs/dev/PLAN-065-transformer-masks-OUTCOME.md`。
- 实现：`nnlearn/nn/transformer.py`。
- 测试：`tests/test_54_transformer_masks.py`。
- 示例：`examples/106_training_masked_transformer_compare.py`。
- 教程：`docs/tutorial/transformer-masks-20261025.md`。

文档和代码均已准备好，由主代理统一执行集成测试、索引维护、中文提交和 push。

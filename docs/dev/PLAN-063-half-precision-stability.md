# PLAN-063：半精度数值稳定性

## 目标

修复 Softmax/LogSoftmax、交叉熵和归一化层在半精度中间规约溢出的问题，保持输出、输入梯度、参数及优化器状态的精度契约。

设计：`docs/design/numerics-20261025-accumulation.md`。

## 步骤

1. 新增 `tests/test_52_half_precision_stability.py`，先记录未修复时的大类别 Softmax、LayerNorm 等失败。
2. 半精度前向和反向在 float32 中累加，输出及回传边恢复目标精度；复查显式混合精度、标签平滑、running statistics。
3. 运行新增、既有 CPU 回归，NumPy 1.24 兼容检查；CUDA 由 PLAN-067 联合验收。
4. 自行复查并修复，编写 `PLAN-063-half-precision-stability-OUTCOME.md`，更新索引，中文提交后立即推送。

## 验收

- 70000 类均匀 Softmax 概率和接近 1，LogSoftmax 有限；相应梯度与高精度参考一致。
- 有限大幅值半精度归一化不产生全零或 NaN，交叉熵大批次均值与梯度保持有限。
- 三种浮点精度、连续训练与缓冲区状态保持正确；现有测试无回归。

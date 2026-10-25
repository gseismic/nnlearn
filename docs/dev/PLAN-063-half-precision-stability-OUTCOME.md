# PLAN-063 实施结果：半精度数值稳定性

日期：2026-10-25。

## 完成内容

1. Softmax/LogSoftmax 在 float32 中完成半精度输入的平移、指数和规约；反向也在单精度中求乘积与和，输出与回传边保持调用方精度。
2. 交叉熵的半精度概率、加权损失、标签计数和均值分母使用 float32，最终才恢复输出精度；保留 ignore_index、三种 reduction 和 label_smoothing 语义。
3. LayerNorm 的半精度均值、方差和反向中间量使用 float32；显式 float32 仿射参数仍产生 float32 输出。
4. BatchNorm 通过可微转换计算半精度统计量，running_mean/running_var 按缓冲区原 dtype 写回，连续训练不提升参数或状态精度。
5. 新增 38 项回归，覆盖 70000 类、100000 样本、三种浮点精度、大幅值归一化、四优化器连续训练及整数 logits 的历史浮点输出。

## 验证与复查

- 修复前首版 37 项为 **14 failed, 23 passed**，复现类别指数和、损失/分母、LayerNorm/BatchNorm 的半精度中间溢出。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest tests/test_52_half_precision_stability.py -q`：**38 passed**。
- Python 3.8 / NumPy 1.24.4、无 PyTorch，同一文件 **20 passed, 18 skipped**；跳过项为显式参考框架对照。
- CPU 既有测试加本任务回归（排除并行中的新 test_53/54/55）：**718 passed, 22 skipped**。22 项为 CI 改造后显式报告的 CUDA 条件测试，过去以 early return 计入通过。
- `ruff check tests/test_52_half_precision_stability.py` 与 `git diff --check` 通过。
- 复查确认仅半精度恢复输出 dtype；整数 logits 继续产生浮点结果，显式混合精度没有被覆盖。最终输出超过目标精度表示范围时仍可能为无穷；本任务修复中间溢出，不改变表示范围。
- CUDA 真实设备及全部任务集成验收另见 PLAN-067，不将 CPU 结果视为 CUDA 验证。

## 交付

- 设计：`docs/design/numerics-20261025-accumulation.md`。
- 计划：`docs/dev/PLAN-063-half-precision-stability.md`。
- 结果：`docs/dev/PLAN-063-half-precision-stability-OUTCOME.md`。

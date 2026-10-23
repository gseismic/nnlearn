# PLAN-061 实施结果：MSE 半精度归一化与数组指数回归

日期：2026-10-24

## 完成情况

两个 P2 回归均已修复。新增 21 项回归测试，CPU 全量 410 项通过。

## 修复内容

1. **MSE 半精度归一化**：均值反向对 float16 差值使用可微的 float32 转换，在 float32 中构造元素数量分母、应用上游梯度并还原广播，最后恢复 float16。100000 个 1 对零目标的梯度恢复为约 2e-5；避免分母溢出和梯度提前转回半精度导致的下溢，保留二阶求导依赖。float32、float64 路径沿用原精度。
2. **数组指数反向**：将标量零指数判断与数组路径分开。数组按元素将零指数对应的反向幂设为零，避免零底数处先计算负次幂；数组常量迁移至输入设备，并在反向末尾将广播梯度还原到输入形状。
3. **回归测试**：覆盖 MSE 在 65000、65520、100000 个元素下的三种浮点精度，半精度广播目标和二阶梯度、上游梯度缩放；覆盖浮点与整数数组指数、混合零指数、分数与负指数、标量输入广播、零维指数、空数组及三阶导数。

## 复现与验证

- 修复前，首版 18 项新增回归运行结果：**10 failed, 8 passed**。失败包括半精度分母转换溢出、最终梯度清零和数组指数布尔判断异常。
- 修复后，首版新增回归加上一轮 92 项回归：**110 passed**。
- 代码复核时补充整数指数、分数及负指数、空数组三项用例，最终新增 **21 项**。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest -q`：**410 passed**，无警告。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python examples/704_compat_training_performance_benchmark.py --device cpu --steps 50 --warmup 2 --repeats 1`：通过。两端 dtype 为 float32，loss 均约 0.000751，accuracy 均为 1.0；loss 差值为 1.74623e-10。本次仅验证训练闭环，不作为稳定性能测量。
- `git diff --check`：通过。
- `nnlearn.cuda.is_available()` 为 False，CUDA 未实测。

## 代码复核

- MSE 的精度转换使用 Tensor 运算，在 create_graph 模式下保留输入与目标的依赖；恢复半精度发生在上游梯度缩放和广播求和之后。
- Pow 未对数组比较结果执行 Python 布尔判断；零指数保护发生在幂运算之前，混合零指数的一阶至三阶导数均通过严格浮点异常检查。
- 检查了数组指数引起的形状扩展，反向通过可微的 sum_to 恢复输入形状。
- 本次仅处理 MSE 均值反向精度和常量数组指数，不改变公开参数签名。

## 交付文件

- 计划：`docs/dev/PLAN-061-mse-pow-regressions.md`。
- 结果：`docs/dev/PLAN-061-mse-pow-regressions-OUTCOME.md`。
- 实现：`nnlearn/nn/loss.py`、`nnlearn/functional/numeric.py`。
- 测试：`tests/test_49_mse_pow_regressions.py`。

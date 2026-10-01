# PLAN-057 实施结果：nnlearn 与 PyTorch 训练性能对照示例

日期：2026-10-01

## 实施内容

1. 新增 `examples/704_compat_training_performance_benchmark.py`：
   - 在同一进程中运行 `nnlearn` 和 PyTorch 的两层 MLP 分类训练。
   - 使用固定合成数据、相同批次顺序、共享初始参数和 SGD 配置，不依赖外部数据下载。
   - 显式适配 `nnlearn.Linear` 权重矩阵的存储方向，确保两边初始线性变换一致。
   - 支持 `--device`、`--steps`、`--warmup`、`--repeats`；默认在 CPU 上各运行 1000 个计时步，结果取 3 次耗时中位数。
   - 输出耗时、步/秒、样本/秒、最终 loss、accuracy 及两边指标差值；CUDA 模式会在计时边界同步设备。
2. 更新 `docs/tutorial/torch-training-mechanism-20260929.md`，加入运行命令、指标说明和结果适用范围。

## 验证结果

- `python examples/704_compat_training_performance_benchmark.py`：通过，退出码为 0。
- 本次环境：NumPy 2.2.6、PyTorch 2.6.0+cu124，CPU 对照。
- 本次结果：PyTorch 使用 6 个 CPU 线程；`nnlearn` 中位耗时 0.274866 秒（3638.14 steps/s，232840.65 samples/s），PyTorch 中位耗时 0.204615 秒（4887.24 steps/s，312783.18 samples/s）；最终 loss 分别为 0.000066，accuracy 均为 1.0。
- 本次耗时比 `nnlearn / PyTorch` 为 1.34x；loss 差值为 `3.15442e-09`，accuracy 差值为 0。数据仅描述当前环境与此示例工作负载。
- `git diff --check`：通过。
- CUDA 路径未运行：当前环境缺少 CuPy，`nnlearn.cuda.is_available()` 不满足运行条件。

## Review 结论

- 训练输入、批次、初始参数、模型结构、学习率和更新步数在两边一致；`nnlearn` 的权重方向在载入参数时转置以保持同一数学变换。
- 预热、数据准备、模型创建和完整数据集评估都不计入训练耗时；计时段只包含前向、损失、反向与 SGD 更新，CUDA 计时边界包含设备同步。
- 未运行完整测试套件；本任务的验收通过实际运行示例完成。

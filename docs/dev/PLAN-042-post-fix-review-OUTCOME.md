# PLAN-042 首轮修复后独立复审结果

日期：2026-09-29

计划文件：`docs/dev/PLAN-042-post-fix-review.md`

发现文件：`docs/dev/PLAN-042-post-fix-review-FINDINGS.md`

## 审查与修复

以已推送的 `5de7a47` 为基线复审首轮修复。固定种子下，稳定排序的 12 组不同 dtype、维度及方向输入和索引梯度的 4 组输入均与 PyTorch 一致。另发现 3 项问题，已在修改代码前写入发现文件，再逐项修复：

1. 状态加载保留目标 dtype，同时允许数值状态按 PyTorch 行为转换；转换前验证数值类型，并在所有键验证及转换完成后统一赋值，避免部分改写。
2. 交叉熵类别标签只接受 PyTorch 当前支持的 `int64` 和 `uint8`，拒绝 `int32` 等不支持的类型。
3. `LayerNorm(())` 和直接构造 `LayerNormFunction(())` 均在初始化时拒绝空归一化形状。

新增 `tests/test_46_post_fix_review.py` 覆盖这三项及状态加载的原子性；调整首轮回归测试中对交叉熵错误消息的断言。随后复核了本轮 diff，移除无用导入，未发现新的可复现问题。

## 验证结果

- `CUDA_VISIBLE_DEVICES='' python -m pytest -q`：**292 passed**。
- `CUDA_VISIBLE_DEVICES='' USE_NLEARN=1 python examples/700_compat_pytorch_training_baseline.py`：MLP、CNN、Transformer 准确率均为 `1.0`，MLP 检查点往返误差 `0`。
- 当前环境没有 CuPy，CUDA 运行路径仍无法实测；本轮结论限于 CPU 实测、代码审查及已记录的 PyTorch 对照。

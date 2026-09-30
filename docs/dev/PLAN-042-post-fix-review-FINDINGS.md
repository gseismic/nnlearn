# PLAN-042 提交后复审发现

日期：2026-09-29

审查基线：`5de7a47`。本文件先于本轮修复创建，记录可复现的问题。

## P2

1. **状态加载错误拒绝可转换的整数缓冲区。** `nlearn/nn/module.py:212` 使用 `np.can_cast(..., casting='same_kind')`，禁止将浮点来源写入整数目标缓冲区。注册 `int64` 缓冲区 `count=[0]` 后加载 `count=[2.9]`，当前实现抛 `TypeError`；PyTorch 将值转换为 `2`，并保持目标 dtype 为 `int64`。这使合法的 PyTorch 检查点在 `nlearn` 中无法恢复。修复应保留目标 dtype、在提交赋值前完成全部转换，并拒绝非数值状态。

## P3

2. **交叉熵仍静默接受 `int32` 类别标签。** `nlearn/nn/loss.py:90` 只检查“整数 dtype”，随后转换为 `int64`。`int32` 标签 `[0]` 可得到损失 `0.313261...`；同一输入在 PyTorch 中报 `expected scalar type Long but found Int`。修复应只接受 PyTorch 支持的 `int64` 和 `uint8` 索引类型，不将其他整数类型悄悄转换为合法标签。
3. **空的 LayerNorm 归一化形状在构造时未被拒绝。** `nlearn/nn/normalization.py:55-56` 允许 `LayerNorm(())` 构造，直到前向调用才因尾部形状不匹配而报错；PyTorch 在构造时即拒绝空形状。配置错误应在创建层时暴露，直接调用 `LayerNormFunction(())` 也应同样校验。

## 已复核而未列为问题的路径

- 固定种子下，稳定排序在三种数值 dtype、两个维度和两个方向共 12 组输入中与 PyTorch 索引一致。
- 基础、布尔及重复高级索引的梯度在四组复现中与 PyTorch 一致。
- 对重复位置的直接赋值，PyTorch 与当前 Pad 反向均给两个写入值传梯度；该行为不作为本轮兼容性问题。

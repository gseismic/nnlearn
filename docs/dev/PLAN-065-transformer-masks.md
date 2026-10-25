# PLAN-065：Transformer 注意力掩码与真实序列训练

日期：2026-10-25。

## 目标与设计

补齐变长序列与前缀预测所需的 padding、causal 和显式注意力掩码；保持旧接口默认行为，形成可运行、无需下载的真实训练示例。

设计：`docs/design/attention-20261025-masks.md`。实现范围限于 `nnlearn/nn/transformer.py`、新增 `tests/test_54_transformer_masks.py`、`examples/106_training_masked_transformer_compare.py` 及相关中文文档；不改其他并行任务负责的数值运算和配置。

## 实施步骤

1. 核对官方 PyTorch 语义并写入设计及计划，确认形状、dtype、设备、全屏蔽行和兼容性契约。
2. 为 MultiheadAttention 增加 attn_mask、key_padding_mask、is_causal，保留 need_weights 的位置及返回约定；编码层转发 src_mask、src_key_padding_mask、is_causal。
3. 补 PyTorch 前向、权重、输入/全部参数/浮点掩码梯度对照；增加布局、交叉注意力、错误输入、全屏蔽行和无泄漏回归。测试注释写清验证目的。
4. 增加变长序列前缀均值预测示例：padding + causal、有效位置监督、连续训练、独立验证序列和扰动检查；支持 USE_NNLEARN 切换，并提供中文使用教程。
5. 逐项 review 代码及接口，运行新增回归、原 Transformer 回归和两端训练示例，修复问题后重复必要验证。全仓集成和完整测试由主代理统一执行。
6. 写入 `docs/dev/PLAN-065-transformer-masks-OUTCOME.md`，记录实测证据、差异和环境限制。按主代理统一安排创建中文提交并 push。

## 验收标准

- bool/浮点、2D/3D、padding/因果及其组合、两种 batch_first 布局和 cross-attention 数值正确，梯度覆盖输入、全部参数及可微浮点掩码。
- 全部屏蔽行明确报错，无静默 NaN；形状、dtype、设备错误易定位；调用不修改用户 Tensor。
- padding 与未来位置扰动不泄漏；旧 need_weights 调用和编码层默认调用保持兼容。
- 两个框架的真实变长序列训练收敛，示例无需网络或数据下载。
- 新增测试、原有相关测试、代码 review 和 diff 检查通过，计划与结果文档齐全。

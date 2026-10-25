# 注意力掩码与变长序列训练设计

日期：2026-10-25。关联计划：`docs/dev/PLAN-065-transformer-masks.md`。

## 目标与场景

现有注意力总是读取所有位置，不能直接用于带 padding 的批次或自回归预测。本次补齐三个常见场景：变长序列忽略 padding、前缀预测隔离未来位置、交叉注意力按样本和头设置不同可见区域。保持现有不传掩码的计算及返回形式。

语义依据为 PyTorch 官方 [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) 和 [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) 文档：布尔值 True 表示屏蔽；浮点掩码加在 softmax 之前的分数上；二维注意力掩码跨批次和头共享，三维掩码区分每个样本及头。

## 对外接口与兼容性

```python
MultiheadAttention.forward(
    query, key=None, value=None, need_weights=False,
    attn_mask=None, key_padding_mask=None, is_causal=False,
)
TransformerEncoderLayer.forward(
    src, src_mask=None, src_key_padding_mask=None, is_causal=False,
)
```

注意力保留原有第 4 个位置参数 `need_weights`；False 仍返回单个输出 Tensor，True 仍返回 `(output, 跨头平均的权重)`。默认 `batch_first=True` 也不改变；编码层原有默认 `batch_first=False` 不改变。编码层直接把三个参数转交自注意力。

输入继续支持三维批次：`(N,L,E)` 或 `(L,N,E)`，key/value 的源长度 S 可以不同于 query 的长度 L；仅要求 key/value 源长度相同。所有输入须为相同设备和浮点 dtype，E 必须等于 embed_dim。明确检查参数和形状，避免通过广播静默接受错误批次。

| 参数 | 形状 | dtype 和行为 |
| --- | --- | --- |
| attn_mask | `(L,S)` 或 `(N*num_heads,L,S)` | bool：True 屏蔽；与 query 同 dtype 的浮点：加法偏置 |
| key_padding_mask | `(N,S)` | 同上，屏蔽的是 key，不自动清零 query 输出 |
| is_causal | Python bool | True 自动生成 `j > i` 的屏蔽区域，矩形交叉注意力使用左上对齐 |

掩码为 nnlearn Tensor，不接受隐式数组转换；设备须与 query 相同；两种显式掩码同时存在时 dtype 须一致。浮点掩码允许有限值及负无穷，拒绝 NaN、正无穷、整数和复数。浮点掩码可以求导，reshape 和相加保留其计算图。所有路径均不原地修改输入掩码。

与 PyTorch `is_causal` 的优化提示不同，本库的 True 保证实际生成因果限制；即使显式 attn_mask 没包含因果区域也会合并。该规则使纯 `is_causal=True` 调用有确定语义；对照测试在 PyTorch 侧提供等价显式因果掩码。

## 内部实现与错误边界

布尔掩码转换为同设备、同输入 dtype 的 `0/-inf` 偏置；二维掩码 reshape 为 `(1,1,L,S)`，三维为 `(N,H,L,S)`，padding 为 `(N,1,1,S)`。通过普通 Tensor 运算相加后，在 softmax 前累加到分数中，不引入独立的注意力自动微分节点。

因果、注意力、padding 限制合并后，如任意样本/头/query 的全部 key 都为负无穷，抛出 `ValueError`，提示每行至少保留一个 key。空 key 序列也拒绝。这是明确的接口边界，避免 softmax 产生 NaN 或训练中悄悄传播无效值。padding 不移除 query，因此调用方应在损失中排除 padding，并避免左侧 padding 与因果限制形成空行。

中间掩码偏置最多具有注意力分数同阶的空间开销 `O(N*H*L*S)`；检查全部屏蔽行在 CUDA 上需要一次标量同步。普通注意力本身的时间和空间复杂度不改变。本次不增加未批次化输入、dropout、稀疏注意力或返回每头权重的新 API。

## 验证

1. 无掩码默认行为和原有位置参数约定继续通过。
2. bool/浮点、二维/三维、padding/因果/组合、self/cross、两种布局对照 PyTorch 前向、返回权重、q/k/v 与全部投影参数梯度；浮点掩码梯度也对照。
3. 编码层对照输出、输入与全部参数梯度；验证掩码确实转发。
4. 干扰被屏蔽 key/value、padding、未来位置，不得改变可见位置输出或梯度；输入和掩码内容保持不变。
5. 错误维度、形状、dtype、设备、非有限掩码和全部屏蔽行明确失败；有 CUDA 的环境补设备检查。
6. 提供无需下载数据的变长序列前缀均值预测示例，同时使用 padding/causal 掩码、有效位置损失和连续优化，可切换 nnlearn/PyTorch；评估训练损失和屏蔽位置扰动不变性。

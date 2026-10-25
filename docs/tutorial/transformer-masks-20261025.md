# 使用 Transformer 处理变长序列和前缀预测

日期：2026-10-25。对应实现：`nnlearn/nn/transformer.py`；完整训练示例：`examples/106_training_masked_transformer_compare.py`。

## 最常用的调用

同一批次的序列长度不同，把短序列在右侧补齐，用 `src_key_padding_mask` 屏蔽补齐的 key。预测当前时刻只能读取当前和过去观测时，再开启 `is_causal=True`。

```python
import numpy as np
import nnlearn as torch
from nnlearn import nn

encoder = nn.TransformerEncoderLayer(
    d_model=4, nhead=2, dim_feedforward=8, batch_first=True,
).to(dtype=torch.float32)
src = torch.tensor(np.random.default_rng(0).normal(size=(2, 5, 4)).astype('float32'))
padding = torch.tensor([
    [False, False, False, True, True],
    [False, False, False, False, True],
])
output = encoder(src, src_key_padding_mask=padding, is_causal=True)
assert output.shape == (2, 5, 4)
```

布尔掩码中 True 表示屏蔽。padding 限制的是 key，输出仍保留所有 query 位置，损失应只统计有效位置。例如回归任务的损失可以写成 `(((prediction - target) ** 2) * valid).sum() / valid.sum()`，其中 valid 是与输出可广播的 0/1 浮点 Tensor。

`MultiheadAttention` 默认 `batch_first=True`；`TransformerEncoderLayer` 默认 False。False 时输入和输出形状为 `(序列长度, 批次, 特征)`；padding 掩码始终为 `(批次, key长度)`，无需转置。

## 显式注意力掩码

直接调用注意力时，`attn_mask` 为 `(query长度,key长度)`，可跨批次和头共享；也可为 `(批次*num_heads,query长度,key长度)`，为每个样本和头分别指定限制。`key_padding_mask` 为 `(批次,key长度)`。交叉注意力的 query 与 key 长度可以不同，key 和 value 的长度必须相同。

```python
attention = nn.MultiheadAttention(embed_dim=4, num_heads=2)
causal = torch.tensor(np.triu(np.ones((5, 5), dtype=bool), k=1))
out, weights = attention(
    src, need_weights=True, attn_mask=causal, key_padding_mask=padding,
)
assert weights.shape == (2, 5, 5)  # 权重已经跨头取平均
```

注意力默认返回单个输出 Tensor，只有 `need_weights=True` 才返回 `(output, weights)`。原有第 4 个位置参数仍为 need_weights，使用新增参数时建议按名称传入。编码层对应的参数名是 `src_mask` 和 `src_key_padding_mask`。

浮点掩码会加到 softmax 之前的分数上，0 保持原分数，负无穷完全屏蔽，其他有限值表达注意力偏置。浮点 dtype 必须与 query 相同；同时使用两种显式掩码时，其 dtype 也必须相同。浮点掩码支持求导，布尔掩码不参与求导。

## 边界与错误处理

- query、key、value 都是三维 nnlearn Tensor，浮点 dtype、设备、批次数相同；每个 key 至少对应一个 value，特征维须等于 embed_dim。
- 掩码也必须显式构造成 nnlearn Tensor，设备与输入一致。本库不自动转换 NumPy 掩码、搬设备或改变浮点 dtype。
- 整数、复数、NaN、正无穷掩码会报错；形状不能靠额外广播代替规定的二维或三维形式。
- 任意样本/头/query 在合并掩码后没有可见 key 时会抛出 ValueError，避免归一化产生 NaN。因果训练建议右侧 padding，左侧 padding 会让早期 query 没有可见 key。
- `is_causal=True` 一定合并 `key位置 > query位置` 的限制，即使另传显式掩码也不会开放未来。矩形注意力按左上对齐处理。

布尔值、浮点偏置及掩码形状参考 PyTorch 官方 [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) 文档。nnlearn 保留自身的默认返回形式，并允许只用 `is_causal=True` 自动生成因果限制；使用 PyTorch 对照时应提供等价的显式因果掩码。

## 运行真实训练示例

示例在本地生成 30 条长度 2～6 的传感器观测，每个有效时刻预测当前前缀累计均值。训练 220 步 Adam，另用独立随机种子的序列验证泛化，同时分别扰动 padding 和未来观测，检查有效输出与前缀输出不变。

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project python examples/106_training_masked_transformer_compare.py
CUDA_VISIBLE_DEVICES='' USE_NNLEARN=0 uv run --no-project python examples/106_training_masked_transformer_compare.py
```

两端使用相同数据和任务，各自初始化后训练；示例不要求两端优化轨迹相同，严格相同参数的前向和全部梯度对照在 `tests/test_54_transformer_masks.py` 中验证。输出包含初始/最终训练损失、独立验证损失、padding 扰动误差和 causal 扰动误差。默认验收为训练 MSE 小于 0.008 且降至初值的 10% 以下，验证 MSE 小于 0.02，两种扰动误差均小于 1e-5。

具备 CuPy/CUDA 环境时可通过 `--device cuda` 运行；没有 CUDA 时默认 CPU 也可独立完成全部训练。

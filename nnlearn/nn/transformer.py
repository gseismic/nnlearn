import math

import numpy as np

from .module import Module
from .linear import Linear
from .activation import ReLU
from .normalization import LayerNorm
from nnlearn import backend, functional as F
from nnlearn.tensor import Tensor


class MultiheadAttention(Module):
    """多头注意力，支持用可见区域限制隔离 padding 和未来位置。"""

    def __init__(self, embed_dim, num_heads, batch_first=True):
        super().__init__()
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim <= 0:
            raise ValueError('embed_dim must be a positive integer')
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
            raise ValueError('num_heads must be a positive integer')
        if embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _validate_inputs(self, query, key, value):
        """在投影前检查批次契约，避免矩阵广播悄悄接受错误输入。"""
        for name, tensor in (('query', query), ('key', key), ('value', value)):
            if not isinstance(tensor, Tensor):
                raise TypeError(f'{name} must be an nnlearn Tensor')
            if tensor.ndim != 3:
                raise ValueError(f'{name} must be a 3D batched tensor')
            if tensor.shape[-1] != self.embed_dim:
                raise ValueError(f'{name} last dimension must equal embed_dim={self.embed_dim}')
            if not np.issubdtype(tensor.dtype, np.floating):
                raise TypeError(f'{name} must have a floating dtype')
            if tensor.dtype != query.dtype:
                raise TypeError('query, key and value must have the same dtype')
            if tensor.device != query.device:
                raise ValueError('query, key and value must be on the same device')
        batch_axis = 0 if self.batch_first else 1
        seq_axis = 1 if self.batch_first else 0
        if key.shape[batch_axis] != query.shape[batch_axis] or value.shape[batch_axis] != query.shape[batch_axis]:
            raise ValueError('query, key and value must have the same batch size')
        if key.shape[seq_axis] != value.shape[seq_axis]:
            raise ValueError('key and value must have the same sequence length')
        if key.shape[seq_axis] == 0:
            raise ValueError('key sequence must contain at least one position')

    def _mask_bias(self, mask, name, query):
        """检查掩码契约；布尔限制转成偏置，浮点偏置保留求导路径。"""
        if not isinstance(mask, Tensor):
            raise TypeError(f'{name} must be an nnlearn Tensor')
        if mask.device != query.device:
            raise ValueError(f'{name} must be on the same device as query')
        if mask.dtype != np.dtype(bool) and not np.issubdtype(mask.dtype, np.floating):
            raise TypeError(f'{name} must have bool or floating dtype')
        if mask.dtype != np.dtype(bool) and mask.dtype != query.dtype:
            raise TypeError(f'{name} floating dtype must match query dtype')
        xp = backend.get_array_module(query.data)
        if mask.dtype == np.dtype(bool):
            # 创建新数组，用户掩码始终只读；常量明确 dtype，兼容 NumPy 1.x。
            data = xp.where(mask.data, xp.asarray(-math.inf, dtype=query.dtype),
                            xp.asarray(0, dtype=query.dtype))
            return Tensor(data, requires_grad=False)
        invalid = xp.any(xp.isnan(mask.data) | xp.isposinf(mask.data))
        if bool(backend.as_numpy(invalid).item()):
            raise ValueError(f'{name} must contain only finite values or -inf')
        return mask

    def _attention_bias(self, query, key, attn_mask, key_padding_mask, is_causal):
        """把不同形状的限制合并，并在 softmax 前拒绝没有可见 key 的行。"""
        b, t, _ = query.shape
        s = key.shape[1]
        bias = None
        if attn_mask is not None:
            bias = self._mask_bias(attn_mask, 'attn_mask', query)
            if attn_mask.shape == (t, s):
                bias = bias.reshape(1, 1, t, s)
            elif attn_mask.shape == (b * self.num_heads, t, s):
                bias = bias.reshape(b, self.num_heads, t, s)
            else:
                raise ValueError(
                    f'attn_mask shape must be {(t, s)} or '
                    f'{(b * self.num_heads, t, s)}, got {attn_mask.shape}'
                )
        if key_padding_mask is not None:
            padding_bias = self._mask_bias(key_padding_mask, 'key_padding_mask', query)
            if key_padding_mask.shape != (b, s):
                raise ValueError(f'key_padding_mask shape must be {(b, s)}, got {key_padding_mask.shape}')
            if attn_mask is not None and attn_mask.dtype != key_padding_mask.dtype:
                raise TypeError('attn_mask and key_padding_mask must have the same dtype')
            padding_bias = padding_bias.reshape(b, 1, 1, s)
            bias = padding_bias if bias is None else bias + padding_bias
        xp = backend.get_array_module(query.data)
        if is_causal:
            future = xp.arange(s)[None, :] > xp.arange(t)[:, None]
            causal_data = xp.where(future, xp.asarray(-math.inf, dtype=query.dtype),
                                   xp.asarray(0, dtype=query.dtype))
            causal_bias = Tensor(causal_data.reshape(1, 1, t, s), requires_grad=False)
            bias = causal_bias if bias is None else bias + causal_bias
        if bias is not None:
            empty_rows = xp.any(xp.all(xp.isneginf(bias.data), axis=-1))
            if bool(backend.as_numpy(empty_rows).item()):
                raise ValueError('attention masks must leave at least one visible key in every row')
        return bias

    def forward(self, query, key=None, value=None, need_weights=False,
                attn_mask=None, key_padding_mask=None, is_causal=False):
        """计算注意力；bool 掩码 True 屏蔽，浮点掩码在 softmax 前相加。

        attn_mask 为 (L,S) 或 (N*num_heads,L,S)，key_padding_mask 为
        (N,S)。浮点掩码须与 query 同 dtype，同传的两种掩码 dtype 须一致。
        is_causal=True 自动合并 j>i 的限制，矩形序列左上对齐。全部屏蔽
        的行会报 ValueError；padding 仅限制 key，不清零 query 的输出。
        保留旧接口：need_weights=False 返回 Tensor，True 返回输出和跨头平均权重。
        """
        if key is None:
            key = query
        if value is None:
            value = key
        self._validate_inputs(query, key, value)
        if not isinstance(is_causal, bool):
            raise TypeError('is_causal must be a bool')
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        bias = self._attention_bias(query, key, attn_mask, key_padding_mask, is_causal)
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        scores = F.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if bias is not None:
            scores = scores + bias
        attn = F.softmax(scores, axis=-1)
        context = F.matmul(attn, v)
        b, _, t, _ = context.shape
        out = context.transpose(1, 2).reshape(b, t, self.embed_dim)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        if need_weights:
            return out, attn.mean(axis=1)
        return out


class TransformerEncoderLayer(Module):
    """包含自注意力和前馈网络的编码层，可隔离 padding 和未来 token。"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, batch_first=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.activation = ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """转发三种可见区域限制；padding 输出仍需在下游损失中排除。"""
        attn = self.self_attn(src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask,
                              is_causal=is_causal)
        src = self.norm1(src + attn)
        ff = self.linear2(self.activation(self.linear1(src)))
        return self.norm2(src + ff)

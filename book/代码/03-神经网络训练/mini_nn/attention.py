"""序列嵌入、注意力和最小 Transformer 编码器组件。"""

import numpy as np

from mini_tensor import Function

from .core import Module, Parameter
from .layers import Linear, ReLU


class _EmbeddingLookup(Function):
    def forward(self, token_ids, weight):
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise TypeError("Embedding 的 token 必须是整数")
        if np.any(token_ids < 0) or np.any(token_ids >= weight.shape[0]):
            raise ValueError("token id 超出词表范围")
        self.token_ids = token_ids.astype(np.int64, copy=False)
        self.weight_shape = weight.shape
        return weight[self.token_ids]

    def backward(self, output_grad):
        grad_weight = np.zeros(self.weight_shape, dtype=output_grad.dtype)
        np.add.at(grad_weight, self.token_ids, output_grad)
        return None, grad_weight


class Embedding(Module):
    """可训练词向量查表；反向会合并重复 token 的行梯度。"""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        if (isinstance(num_embeddings, bool)
                or isinstance(embedding_dim, bool)
                or int(num_embeddings) != num_embeddings
                or int(embedding_dim) != embedding_dim
                or num_embeddings <= 0 or embedding_dim <= 0):
            raise ValueError("词表大小和向量维数必须是正整数")
        scale = 1.0 / np.sqrt(int(embedding_dim))
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = Parameter(
            np.random.randn(self.num_embeddings, self.embedding_dim) * scale
        )

    def forward(self, token_ids):
        return _EmbeddingLookup()(token_ids, self.weight)


class _MultiheadAttention(Function):
    def __init__(self, head_dim):
        self.scale = 1.0 / np.sqrt(head_dim)

    def forward(self, query, key, value):
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("多头注意力需要 (batch, time, heads, head_dim) 输入")
        if query.shape[:3] != key.shape[:3] or key.shape[:3] != value.shape[:3]:
            raise ValueError("Q、K、V 的 batch、time 和 head 维度必须一致")

        q = np.swapaxes(query, 1, 2)
        k = np.swapaxes(key, 1, 2)
        v = np.swapaxes(value, 1, 2)
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * self.scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        context = np.matmul(probabilities, v)

        self.query = q
        self.key = k
        self.value = v
        self.probabilities = probabilities
        return np.swapaxes(context, 1, 2)

    def backward(self, output_grad):
        grad_context = np.swapaxes(output_grad, 1, 2)
        probabilities = self.probabilities
        grad_value = np.matmul(np.swapaxes(probabilities, -1, -2), grad_context)
        grad_probabilities = np.matmul(
            grad_context, np.swapaxes(self.value, -1, -2)
        )
        grad_scores = probabilities * (
            grad_probabilities
            - np.sum(grad_probabilities * probabilities, axis=-1, keepdims=True)
        )
        grad_query = np.matmul(grad_scores, self.key) * self.scale
        grad_key = np.matmul(
            np.swapaxes(grad_scores, -1, -2), self.query
        ) * self.scale
        return (
            np.swapaxes(grad_query, 1, 2),
            np.swapaxes(grad_key, 1, 2),
            np.swapaxes(grad_value, 1, 2),
        )


class MultiheadSelfAttention(Module):
    """batch-first 自注意力；不含 mask、dropout 或位置编码。"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if (isinstance(embed_dim, bool) or isinstance(num_heads, bool)
                or int(embed_dim) != embed_dim or int(num_heads) != num_heads
                or embed_dim <= 0 or num_heads <= 0
                or embed_dim % num_heads != 0):
            raise ValueError("embed_dim 和 num_heads 必须为正整数且前者可被后者整除")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = Linear(self.embed_dim, self.embed_dim)
        self.k_proj = Linear(self.embed_dim, self.embed_dim)
        self.v_proj = Linear(self.embed_dim, self.embed_dim)
        self.out_proj = Linear(self.embed_dim, self.embed_dim)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.embed_dim:
            raise ValueError("注意力输入必须是 (batch, time, embed_dim)")
        batch, time, _ = inputs.shape
        flat = inputs.reshape(batch * time, self.embed_dim)

        def project(layer):
            return layer(flat).reshape(
                batch, time, self.num_heads, self.head_dim
            )

        query = project(self.q_proj)
        key = project(self.k_proj)
        value = project(self.v_proj)
        context = _MultiheadAttention(self.head_dim)(query, key, value)
        context = context.reshape(batch * time, self.embed_dim)
        return self.out_proj(context).reshape(batch, time, self.embed_dim)


class _LayerNorm(Function):
    def __init__(self, eps):
        self.eps = eps

    def forward(self, inputs, gamma, beta):
        if inputs.ndim < 1 or inputs.shape[-1] != gamma.shape[0]:
            raise ValueError("LayerNorm 的最后一维必须与参数长度相同")
        if beta.shape != gamma.shape:
            raise ValueError("LayerNorm 的 gamma 和 beta 形状必须相同")
        mean = np.mean(inputs, axis=-1, keepdims=True)
        centered = inputs - mean
        variance = np.mean(centered**2, axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(variance + self.eps)
        self.normalized = centered * self.inv_std
        self.gamma = gamma
        self.feature_count = inputs.shape[-1]
        return self.normalized * gamma + beta

    def backward(self, output_grad):
        reduce_axes = tuple(range(output_grad.ndim - 1))
        grad_gamma = output_grad * self.normalized
        grad_beta = output_grad
        if reduce_axes:
            grad_gamma = np.sum(grad_gamma, axis=reduce_axes)
            grad_beta = np.sum(grad_beta, axis=reduce_axes)

        grad_normalized = output_grad * self.gamma
        sum_grad = np.sum(grad_normalized, axis=-1, keepdims=True)
        sum_grad_normalized = np.sum(
            grad_normalized * self.normalized, axis=-1, keepdims=True
        )
        grad_inputs = (self.inv_std / self.feature_count) * (
            self.feature_count * grad_normalized
            - sum_grad
            - self.normalized * sum_grad_normalized
        )
        return grad_inputs, grad_gamma, grad_beta


class LayerNorm(Module):
    """只沿最后一维归一化的 LayerNorm。"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if (isinstance(normalized_shape, bool)
                or int(normalized_shape) != normalized_shape
                or normalized_shape <= 0 or eps <= 0):
            raise ValueError("normalized_shape 和 eps 必须为正数")
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = Parameter(np.ones(self.normalized_shape))
        self.bias = Parameter(np.zeros(self.normalized_shape))

    def forward(self, inputs):
        return _LayerNorm(self.eps)(inputs, self.weight, self.bias)


class TransformerEncoderLayer(Module):
    """无 mask、dropout 和位置编码的 batch-first 编码器层。"""

    def __init__(self, d_model, num_heads, dim_feedforward=32):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.activation = ReLU()
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm2 = LayerNorm(d_model)
        self.d_model = int(d_model)

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[-1] != self.d_model:
            raise ValueError("编码器输入必须是 (batch, time, d_model)")
        hidden = self.norm1(inputs + self.self_attn(inputs))
        batch, time, _ = hidden.shape
        flat = hidden.reshape(batch * time, self.d_model)
        feedforward = self.linear2(self.activation(self.linear1(flat)))
        feedforward = feedforward.reshape(batch, time, self.d_model)
        return self.norm2(hidden + feedforward)

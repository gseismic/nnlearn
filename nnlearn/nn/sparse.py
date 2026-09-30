from nnlearn.function import Function
from nnlearn.tensor import Tensor
from nnlearn import backend
from .module import Module
from .parameter import Parameter
import numpy as np


class EmbeddingFunction(Function):
    def forward(self, indices, weight):
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError('embedding indices must be integers')
        xp = backend.get_array_module(indices, weight)
        if bool(backend.as_numpy(xp.any(
            (indices < 0) | (indices >= weight.shape[0])
        )).item()):
            raise IndexError('embedding index out of range')
        self.indices = indices.astype("int64")
        self.weight_shape = weight.shape
        return weight[self.indices]

    def backward(self, gy):
        xp = backend.get_array_module(gy.data)
        gweight = xp.zeros(self.weight_shape, dtype=gy.data.dtype)
        xp.add.at(gweight, self.indices, gy.data)
        return None, Tensor(gweight)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        scale = np.sqrt(1 / embedding_dim)
        self.weight = Parameter(
            np.random.randn(num_embeddings, embedding_dim) * scale,
            name='W'
        )

    def forward(self, x):
        return EmbeddingFunction()(x, self.weight)

"""线性层、激活函数和顺序容器。"""

import numpy as np

from mini_tensor import Function

from .core import Module, Parameter


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear 的输入和输出特征数必须为正数")
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        limit = np.sqrt(6.0 / (self.in_features + self.out_features))
        initial_weight = np.random.uniform(
            -limit, limit, size=(self.in_features, self.out_features)
        )
        self.weight = Parameter(initial_weight)
        self.bias = (
            Parameter(np.zeros(self.out_features, dtype=np.float64))
            if bias else None
        )

    def forward(self, inputs):
        if inputs.ndim != 2:
            raise ValueError("本篇 Linear 接受形状为 (batch, features) 的二维输入")
        if inputs.shape[1] != self.in_features:
            raise ValueError(
                f"Linear 需要 {self.in_features} 个输入特征，"
                f"实际收到 {inputs.shape[1]} 个"
            )
        output = inputs @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


class _ReLU(Function):
    def forward(self, value):
        self.mask = value > 0
        return np.maximum(value, 0)

    def backward(self, output_grad):
        return output_grad * self.mask


class ReLU(Module):
    def forward(self, inputs):
        return _ReLU()(inputs)


class _Tanh(Function):
    def forward(self, value):
        self.output_data = np.tanh(value)
        return self.output_data

    def backward(self, output_grad):
        return output_grad * (1.0 - self.output_data**2)


class Tanh(Module):
    def forward(self, inputs):
        return _Tanh()(inputs)


class _Sigmoid(Function):
    def forward(self, value):
        output = np.empty_like(value, dtype=np.result_type(value, np.float64))
        positive = value >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exp_value = np.exp(value[~positive])
        output[~positive] = exp_value / (1.0 + exp_value)
        self.output_data = output
        return output

    def backward(self, output_grad):
        return output_grad * self.output_data * (1.0 - self.output_data)


class Sigmoid(Module):
    def forward(self, inputs):
        return _Sigmoid()(inputs)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for index, layer in enumerate(layers):
            if not isinstance(layer, Module):
                raise TypeError("Sequential 中的每一项都必须是 Module")
            setattr(self, f"layer_{index}", layer)

    def forward(self, inputs):
        for layer in self._modules.values():
            inputs = layer(inputs)
        return inputs

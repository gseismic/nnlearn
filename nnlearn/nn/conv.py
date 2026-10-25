import numpy as np
from nnlearn.function import Function
from nnlearn.tensor import Tensor
from nnlearn import backend
from nnlearn.settings import Config
from .module import Module
from .parameter import Parameter


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class Conv2dFunction(Function):
    def __init__(self, stride=1, padding=0):
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, x, weight, bias=None):
        xp = backend.get_array_module(x, weight)
        self.has_bias = bias is not None
        self.x_shape = x.shape
        self.weight_shape = weight.shape
        n, _, h, w = x.shape
        out_channels, _, kh, kw = weight.shape
        sh, sw = self.stride
        ph, pw = self.padding

        self.x_pad = xp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1
        y = xp.zeros((n, out_channels, out_h, out_w), dtype=x.dtype)

        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                region = self.x_pad[:, :, hs:hs + kh, ws:ws + kw]
                y[:, :, i, j] = xp.tensordot(
                    region, weight, axes=((1, 2, 3), (1, 2, 3))
                )
        if bias is not None:
            y = y + bias.reshape(1, -1, 1, 1)
        return y

    def backward(self, gy):
        if Config.enable_backprop:
            return self._backward_with_graph(gy)
        xp = backend.get_array_module(gy.data)
        x, weight = self.inputs[:2]
        gy_data = gy.data
        n, _, h, w = self.x_shape
        out_channels, _, kh, kw = self.weight_shape
        sh, sw = self.stride
        ph, pw = self.padding

        gx_pad = xp.zeros_like(self.x_pad)
        gweight = xp.zeros_like(weight.data)
        gbias = xp.sum(gy_data, axis=(0, 2, 3)) if self.has_bias else None

        out_h, out_w = gy_data.shape[2], gy_data.shape[3]
        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                region = self.x_pad[:, :, hs:hs + kh, ws:ws + kw]
                for oc in range(out_channels):
                    g = gy_data[:, oc, i, j][:, None, None, None]
                    gweight[oc] += xp.sum(region * g, axis=0)
                    gx_pad[:, :, hs:hs + kh, ws:ws + kw] += weight.data[oc] * g

        gx = gx_pad[:, :, ph:ph + h, pw:pw + w] if ph or pw else gx_pad
        if self.has_bias:
            return Tensor(gx), Tensor(gweight), Tensor(gbias)
        return Tensor(gx), Tensor(gweight)

    def _backward_with_graph(self, gy):
        """用可微窗口和矩阵乘法表达卷积反向，保留输入/权重的交叉高阶导数。"""
        from nnlearn.functional.matrix import matmul, stack
        from nnlearn.functional.pad import pad

        x, weight = self.inputs[:2]
        n, channels, height, width = self.x_shape
        outputs, _, kh, kw = self.weight_shape
        sh, sw = self.stride
        ph, pw = self.padding
        padded_shape = (n, channels, height + 2 * ph, width + 2 * pw)
        inner = (slice(None), slice(None), slice(ph, ph + height), slice(pw, pw + width))
        padded = pad(x, padded_shape, inner) if ph or pw else x
        out_h, out_w = gy.shape[2:]
        windows = []
        selectors = []
        for i in range(out_h):
            for j in range(out_w):
                selector = (slice(None), slice(None), slice(i * sh, i * sh + kh),
                            slice(j * sw, j * sw + kw))
                selectors.append(selector)
                windows.append(padded[selector].reshape(n, -1))
        columns = stack(windows, dim=1)
        grad_matrix = gy.permute(0, 2, 3, 1).reshape(n, -1, outputs)
        gweight = matmul(grad_matrix.reshape(-1, outputs).T,
                        columns.reshape(-1, channels * kh * kw)).reshape(self.weight_shape)
        patches = matmul(grad_matrix, weight.reshape(outputs, -1))
        gx_padded = None
        for position, selector in enumerate(selectors):
            patch = patches[:, position, :].reshape(n, channels, kh, kw)
            contribution = pad(patch, padded_shape, selector)
            gx_padded = contribution if gx_padded is None else gx_padded + contribution
        gx = gx_padded[inner] if ph or pw else gx_padded
        if self.has_bias:
            return gx, gweight, gy.sum(axis=(0, 2, 3))
        return gx, gweight


def conv2d(x, weight, bias=None, stride=1, padding=0):
    if bias is None:
        return Conv2dFunction(stride, padding)(x, weight)
    return Conv2dFunction(stride, padding)(x, weight, bias)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super().__init__()
        kh, kw = _pair(kernel_size)
        scale = np.sqrt(2 / (in_channels * kh * kw))
        self.weight = Parameter(
            np.random.randn(out_channels, in_channels, kh, kw) * scale,
            name='W'
        )
        self.weight._fan_in = in_channels * kh * kw
        self.weight._fan_out = out_channels * kh * kw
        self.bias = Parameter(np.zeros(out_channels), name='b') if bias else None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

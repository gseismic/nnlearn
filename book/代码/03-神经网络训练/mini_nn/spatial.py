"""二维空间算子：卷积、最大池化和展平。"""

import numpy as np

from mini_tensor import Function

from .core import Module, Parameter


def _pair(value, name, allow_zero=False):
    values = value if isinstance(value, tuple) else (value, value)
    if (len(values) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, np.integer))
                   for item in values)
            or any(item < 0 if allow_zero else item <= 0 for item in values)):
        qualifier = "非负" if allow_zero else "正"
        raise ValueError(f"{name} 必须是两个{qualifier}整数")
    return tuple(int(item) for item in values)


class _Conv2d(Function):
    def __init__(self, stride, padding):
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding", allow_zero=True)

    def forward(self, inputs, weight, bias=None):
        if inputs.ndim != 4 or weight.ndim != 4:
            raise ValueError("Conv2d 输入和卷积核必须分别是 NCHW、OCHW 四维数组")
        batch, in_channels, height, width = inputs.shape
        out_channels, weight_channels, kh, kw = weight.shape
        if in_channels != weight_channels:
            raise ValueError("输入通道数与卷积核通道数不匹配")
        if bias is not None and bias.shape != (out_channels,):
            raise ValueError("bias 的形状必须是 (out_channels,)")

        sh, sw = self.stride
        ph, pw = self.padding
        out_h = (height + 2 * ph - kh) // sh + 1
        out_w = (width + 2 * pw - kw) // sw + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("卷积核大于填充后的输入尺寸")

        self.input_shape = inputs.shape
        self.weight_shape = weight.shape
        self.input_pad = np.pad(
            inputs, ((0, 0), (0, 0), (ph, ph), (pw, pw))
        )
        output = np.zeros(
            (batch, out_channels, out_h, out_w), dtype=inputs.dtype
        )
        for row in range(out_h):
            hs = row * sh
            for col in range(out_w):
                ws = col * sw
                patch = self.input_pad[:, :, hs:hs + kh, ws:ws + kw]
                output[:, :, row, col] = np.tensordot(
                    patch, weight, axes=((1, 2, 3), (1, 2, 3))
                )
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)
        self.has_bias = bias is not None
        self.out_shape = output.shape
        return output

    def backward(self, output_grad):
        batch, _, height, width = self.input_shape
        out_channels, _, kh, kw = self.weight_shape
        sh, sw = self.stride
        ph, pw = self.padding
        inputs, weight = self.inputs[:2]
        gx_pad = np.zeros_like(self.input_pad)
        gweight = np.zeros_like(weight.data)
        gbias = (output_grad.sum(axis=(0, 2, 3))
                 if self.has_bias else None)

        for row in range(self.out_shape[2]):
            hs = row * sh
            for col in range(self.out_shape[3]):
                ws = col * sw
                patch = self.input_pad[:, :, hs:hs + kh, ws:ws + kw]
                grad_at = output_grad[:, :, row, col]
                gweight += np.einsum("no,nchw->ochw", grad_at, patch)
                gx_pad[:, :, hs:hs + kh, ws:ws + kw] += np.einsum(
                    "no,ochw->nchw", grad_at, weight.data
                )

        gx = gx_pad[:, :, ph:ph + height, pw:pw + width]
        if self.has_bias:
            return gx, gweight, gbias
        return gx, gweight


class Conv2d(Module):
    """仅支持 NCHW 输入、对称 padding 和二维步幅的卷积层。"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()
        if (in_channels <= 0 or out_channels <= 0
                or isinstance(in_channels, bool) or isinstance(out_channels, bool)
                or int(in_channels) != in_channels
                or int(out_channels) != out_channels):
            raise ValueError("输入和输出通道数必须是正整数")
        kh, kw = _pair(kernel_size, "kernel_size")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (kh, kw)
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding", allow_zero=True)

        scale = np.sqrt(2.0 / (self.in_channels * kh * kw))
        weight = np.random.randn(
            self.out_channels, self.in_channels, kh, kw
        ) * scale
        self.weight = Parameter(weight)
        self.bias = (Parameter(np.zeros(self.out_channels, dtype=np.float64))
                     if bias else None)

    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("Conv2d 需要 NCHW 四维输入")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Conv2d 需要 {self.in_channels} 个输入通道，"
                f"实际收到 {inputs.shape[1]} 个"
            )
        if self.bias is None:
            return _Conv2d(self.stride, self.padding)(inputs, self.weight)
        return _Conv2d(self.stride, self.padding)(
            inputs, self.weight, self.bias
        )


class _MaxPool2d(Function):
    def __init__(self, kernel_size, stride):
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")

    def forward(self, inputs):
        if inputs.ndim != 4:
            raise ValueError("MaxPool2d 需要 NCHW 四维输入")
        self.input_shape = inputs.shape
        batch, channels, height, width = inputs.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        out_h = (height - kh) // sh + 1
        out_w = (width - kw) // sw + 1
        if out_h <= 0 or out_w <= 0:
            raise ValueError("池化窗口大于输入尺寸")

        output = np.empty((batch, channels, out_h, out_w), dtype=inputs.dtype)
        self.argmax = np.empty((batch, channels, out_h, out_w), dtype=np.int64)
        for row in range(out_h):
            hs = row * sh
            for col in range(out_w):
                ws = col * sw
                patch = inputs[:, :, hs:hs + kh, ws:ws + kw]
                flat = patch.reshape(batch, channels, kh * kw)
                self.argmax[:, :, row, col] = np.argmax(flat, axis=2)
                output[:, :, row, col] = np.max(flat, axis=2)
        return output

    def backward(self, output_grad):
        gx = np.zeros(self.input_shape, dtype=output_grad.dtype)
        batch, channels, _, _ = self.input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        for row in range(output_grad.shape[2]):
            hs = row * sh
            for col in range(output_grad.shape[3]):
                ws = col * sw
                winners = self.argmax[:, :, row, col]
                for n in range(batch):
                    for c in range(channels):
                        offset = winners[n, c]
                        gx[n, c, hs + offset // kw, ws + offset % kw] += (
                            output_grad[n, c, row, col]
                        )
        return gx


class MaxPool2d(Module):
    """无 padding 的二维最大池化；默认步幅等于窗口大小。"""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(
            kernel_size if stride is None else stride, "stride"
        )

    def forward(self, inputs):
        return _MaxPool2d(self.kernel_size, self.stride)(inputs)


class Flatten(Module):
    """合并 batch 之后的维度，保留第 0 维批次轴。"""

    def forward(self, inputs):
        if inputs.ndim < 2:
            raise ValueError("Flatten 至少需要保留批次轴和一个特征轴")
        return inputs.reshape(inputs.shape[0], -1)

"""本书第二篇使用的 NumPy Tensor 与反向模式自动微分实现。"""

import numpy as np


def sum_to(array, shape):
    """把广播后的梯度沿多出来或被扩展的轴求和，恢复目标形状。"""
    array = np.asarray(array)
    shape = tuple(shape)
    if array.shape == shape:
        return array
    if len(shape) > array.ndim:
        raise ValueError("目标形状的维数不能大于梯度维数")

    while array.ndim > len(shape):
        array = array.sum(axis=0)

    axes = []
    for axis, target_size in enumerate(shape):
        current_size = array.shape[axis]
        if target_size == 1 and current_size != 1:
            axes.append(axis)
        elif target_size != current_size:
            raise ValueError(
                f"无法把形状 {array.shape} 的梯度还原到 {shape}"
            )

    if axes:
        array = array.sum(axis=tuple(axes), keepdims=True)
    return array.reshape(shape)


class Tensor:
    """保存 NumPy 数组，并在需要时连接到自动微分计算图。"""

    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            data = data.data

        self.data = np.array(data, dtype=dtype, copy=True)
        if requires_grad and not np.issubdtype(self.data.dtype, np.floating):
            raise TypeError("只有浮点 Tensor 可以设置 requires_grad=True")

        self.requires_grad = bool(requires_grad)
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def numpy(self):
        """返回数据副本，避免外部改动绕过 Tensor。"""
        return self.data.copy()

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def zero_grad(self):
        self.grad = None

    def requires_grad_(self, requires_grad=True):
        requires_grad = bool(requires_grad)
        if requires_grad and not np.issubdtype(self.dtype, np.floating):
            raise TypeError("只有浮点 Tensor 可以设置 requires_grad=True")
        self.requires_grad = requires_grad
        return self

    def backward(self, grad=None):
        """反向传播并累积叶子 Tensor 的梯度。"""
        if not self.requires_grad:
            return

        if grad is None:
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=self.dtype)
            if seed.shape != self.shape:
                raise ValueError(
                    f"初始梯度形状应为 {self.shape}，实际为 {seed.shape}"
                )

        gradients = {id(self): seed}
        tensors = {id(self): self}
        functions = []
        seen_functions = set()

        def add_function(function):
            if id(function) not in seen_functions:
                seen_functions.add(id(function))
                functions.append(function)
                functions.sort(key=lambda item: item.generation)

        if self.creator is not None:
            add_function(self.creator)

        while functions:
            function = functions.pop()
            output_grad = gradients.get(id(function.output))
            if output_grad is None:
                continue

            input_grads = function.backward(output_grad)
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            if len(input_grads) != len(function.inputs):
                raise RuntimeError("Function.backward 返回的梯度数量不匹配")

            for tensor, input_grad in zip(function.inputs, input_grads):
                if input_grad is None or not tensor.requires_grad:
                    continue
                input_grad = np.asarray(input_grad)
                if input_grad.shape != tensor.shape:
                    raise RuntimeError(
                        f"反向结果形状 {input_grad.shape} 与输入形状 "
                        f"{tensor.shape} 不匹配"
                    )

                key = id(tensor)
                if key in gradients:
                    gradients[key] = gradients[key] + input_grad
                else:
                    gradients[key] = input_grad
                    tensors[key] = tensor
                if tensor.creator is not None:
                    add_function(tensor.creator)

        for key, tensor_grad in gradients.items():
            tensor = tensors[key]
            if tensor.creator is not None:
                continue
            if tensor.grad is None:
                tensor.grad = tensor_grad.copy()
            else:
                tensor.grad = tensor.grad + tensor_grad

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape(tuple(shape))(self)

    def sum(self, axis=None, keepdims=False):
        return Sum(axis=axis, keepdims=keepdims)(self)

    def mean(self, axis=None, keepdims=False):
        return Mean(axis=axis, keepdims=keepdims)(self)

    def sigmoid(self):
        return Sigmoid()(self)

    def matmul(self, other):
        return MatMul()(self, other)

    def mm(self, other):
        return self.matmul(other)

    def __add__(self, other):
        return Add()(self, other)

    def __radd__(self, other):
        return Add()(other, self)

    def __mul__(self, other):
        return Mul()(self, other)

    def __rmul__(self, other):
        return Mul()(other, self)

    def __neg__(self):
        return Mul()(self, -1.0)

    def __sub__(self, other):
        return self + (-as_tensor(other))

    def __rsub__(self, other):
        return as_tensor(other) + (-self)

    def __truediv__(self, other):
        return self * (as_tensor(other) ** -1)

    def __rtruediv__(self, other):
        return as_tensor(other) * (self ** -1)

    def __pow__(self, exponent):
        return Pow(exponent)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def __repr__(self):
        return (
            f"Tensor(data={self.data!r}, requires_grad="
            f"{self.requires_grad})"
        )


def as_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


class Function:
    """一次张量运算：前向接收数组，反向返回每个输入的梯度。"""

    def __call__(self, *inputs):
        inputs = tuple(as_tensor(value) for value in inputs)
        output_data = self.forward(*(value.data for value in inputs))
        requires_grad = any(value.requires_grad for value in inputs)
        output = Tensor(output_data, requires_grad=requires_grad)

        if requires_grad:
            self.inputs = inputs
            self.output = output
            self.generation = max(value.generation for value in inputs)
            output.creator = self
            output.generation = self.generation + 1
        return output

    def forward(self, *arrays):
        raise NotImplementedError

    def backward(self, output_grad):
        raise NotImplementedError


class Add(Function):
    def forward(self, left, right):
        self.left_shape = left.shape
        self.right_shape = right.shape
        return left + right

    def backward(self, output_grad):
        return (
            sum_to(output_grad, self.left_shape),
            sum_to(output_grad, self.right_shape),
        )


class Mul(Function):
    def forward(self, left, right):
        self.left_shape = left.shape
        self.right_shape = right.shape
        self.left = left
        self.right = right
        return left * right

    def backward(self, output_grad):
        return (
            sum_to(output_grad * self.right, self.left_shape),
            sum_to(output_grad * self.left, self.right_shape),
        )


class MatMul(Function):
    def forward(self, left, right):
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("本篇的矩阵乘法只接受二维 Tensor")
        if left.shape[1] != right.shape[0]:
            raise ValueError(
                f"矩阵形状不兼容：{left.shape} 与 {right.shape}"
            )
        self.left = left
        self.right = right
        return left @ right

    def backward(self, output_grad):
        return output_grad @ self.right.T, self.left.T @ output_grad


class Pow(Function):
    def __init__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("指数必须是 Python 数值")
        self.exponent = exponent

    def forward(self, value):
        self.value = value
        return value**self.exponent

    def backward(self, output_grad):
        exponent = self.exponent
        if exponent == 0:
            return np.zeros_like(self.value) * output_grad
        return exponent * self.value ** (exponent - 1) * output_grad


class Sigmoid(Function):
    """数值稳定的 sigmoid 及其反向梯度。"""

    def forward(self, value):
        output = np.empty_like(
            value, dtype=np.result_type(value.dtype, np.float32)
        )
        if value.ndim == 0:
            scalar = float(value)
            if scalar >= 0:
                output[...] = 1.0 / (1.0 + np.exp(-scalar))
            else:
                exp_value = np.exp(scalar)
                output[...] = exp_value / (1.0 + exp_value)
        else:
            positive = value >= 0
            output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
            exp_value = np.exp(value[~positive])
            output[~positive] = exp_value / (1.0 + exp_value)
        self.output_data = output
        return output

    def backward(self, output_grad):
        return output_grad * self.output_data * (1.0 - self.output_data)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, value):
        self.input_shape = value.shape
        return value.reshape(self.shape)

    def backward(self, output_grad):
        return output_grad.reshape(self.input_shape)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = bool(keepdims)

    def forward(self, value):
        self.input_shape = value.shape
        ndim = value.ndim
        if self.axis is None:
            self.axes = tuple(range(ndim))
        else:
            if isinstance(self.axis, (int, np.integer)):
                raw_axes = (int(self.axis),)
            elif isinstance(self.axis, (tuple, list)):
                raw_axes = tuple(int(axis) for axis in self.axis)
            else:
                raise TypeError("归约轴必须是整数、整数序列或 None")
            axes = []
            for axis in raw_axes:
                axis = axis + ndim if axis < 0 else axis
                if axis < 0 or axis >= ndim:
                    raise IndexError(
                        f"归约轴 {axis} 超出 {ndim} 维 Tensor 的范围"
                    )
                if axis in axes:
                    raise ValueError("归约轴不能重复")
                axes.append(axis)
            self.axes = tuple(axes)
        numpy_axis = None if self.axis is None else self.axes
        return value.sum(axis=numpy_axis, keepdims=self.keepdims)

    def backward(self, output_grad):
        grad = output_grad
        if not self.keepdims:
            for axis in sorted(self.axes):
                grad = np.expand_dims(grad, axis=axis)
        return np.broadcast_to(grad, self.input_shape)


class Mean(Sum):
    def forward(self, value):
        total = super().forward(value)
        if self.axis is None:
            self.count = value.size
        else:
            self.count = int(
                np.prod([value.shape[axis] for axis in self.axes], dtype=np.int64)
            )
        return total / self.count

    def backward(self, output_grad):
        return super().backward(output_grad) / self.count

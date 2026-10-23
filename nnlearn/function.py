import numpy as np
import weakref
from . import backend
from .log import log_function_call
from .settings import log_settings, runtime_settings, Config

class Function:

    def __init__(self, log_enabled=None):
        if log_enabled is None:
            self.log_enabled = log_settings.get('func_log_enabled', False)
        else:
            self.log_enabled = log_enabled
        self.generation = None

    def __call__(self, *inputs):
        from .tensor import Tensor, ensure_tensor
        floating_inputs = [
            input for input in inputs
            if isinstance(input, Tensor) and np.issubdtype(input.dtype, np.floating)
        ]
        scalar_dtype = (
            np.result_type(*[input.dtype for input in floating_inputs])
            if floating_inputs else None
        )
        scalar_device = (
            'cuda' if any(input.device == 'cuda' for input in floating_inputs)
            else 'cpu'
        )
        # 仅原生 Python 数字沿用浮点上下文；显式 Tensor、NumPy 类型和 bool 保留自身精度。
        inputs = [
            Tensor(input, dtype=scalar_dtype, device=scalar_device,
                   requires_grad=False)
            if scalar_dtype is not None and type(input) in (int, float)
            else ensure_tensor(input)
            for input in inputs
        ]
        data_devices = {input.device for input in inputs if input.data.shape != ()}
        if len(data_devices) > 1:
            raise ValueError(f'all non-scalar inputs must be on the same device: {data_devices}')
        if data_devices == {'cuda'}:
            for input in inputs:
                if input.device == 'cpu' and input.data.shape == ():
                    input.data = backend.to_device(input.data, 'cuda')
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        # `inputs` 仅仅在反向传播时才需要，不反向传播时，不用保留
        # outputs = [tensor.Tensor(y) for y in ys]
        requires_grad = (
            Config.enable_backprop
            and any(input.requires_grad for input in inputs)
        )
        outputs = [Tensor(y, requires_grad=requires_grad) for y in ys]
        if requires_grad:
            # 更新`代`, 为所有输入代的最大值
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            # linear 可能释放中间 data，回传时仍需要建图时的输入精度。
            self.input_dtypes = [input.dtype for input in inputs]
            # 解除输出的循环引用
            if runtime_settings.get('remove_recursive_ref', True):
                self.outputs = [weakref.ref(output) for output in outputs]
            else:
                self.outputs = [output for output in outputs]

        return outputs[0] if len(outputs) == 1 else outputs

    def get_outputs(self):
        """按实际存储类型解析输出，避免前后向切换引用配置时读取错误。"""
        return [
            output() if isinstance(output, weakref.ReferenceType) else output
            for output in self.outputs
        ]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

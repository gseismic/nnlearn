"""参数和模块注册。"""

import numpy as np

from mini_tensor import Tensor


class Parameter(Tensor):
    """默认需要梯度的 Tensor，用作可训练参数。"""

    def __init__(self, data, dtype=None):
        source = data.data if isinstance(data, Tensor) else np.asarray(data)
        if dtype is None and not np.issubdtype(source.dtype, np.floating):
            dtype = np.float64
        super().__init__(data, requires_grad=True, dtype=dtype)


class Module:
    """自动收集直接赋值的 Parameter 和子 Module。"""

    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        parameters = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")
        if parameters is not None:
            parameters.pop(name, None)
            modules.pop(name, None)
            if isinstance(value, Parameter):
                parameters[name] = value
            elif isinstance(value, Module):
                modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix="", memo=None):
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))

        for name, parameter in self._parameters.items():
            if id(parameter) not in memo:
                memo.add(id(parameter))
                yield f"{prefix}{name}", parameter

        for name, module in self._modules.items():
            yield from module.named_parameters(
                prefix=f"{prefix}{name}.", memo=memo
            )

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def children(self):
        return iter(self._modules.values())

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode=True):
        self.training = bool(mode)
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {
            name: parameter.data.copy()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state, strict=True):
        parameters = dict(self.named_parameters())
        missing = [name for name in parameters if name not in state]
        unexpected = [name for name in state if name not in parameters]
        if strict and (missing or unexpected):
            raise KeyError(
                f"状态键不匹配：缺少 {missing}，多出 {unexpected}"
            )

        pending = []
        for name, parameter in parameters.items():
            if name not in state:
                continue
            value = np.asarray(state[name])
            if value.shape != parameter.shape:
                raise ValueError(
                    f"参数 {name} 的形状应为 {parameter.shape}，"
                    f"实际为 {value.shape}"
                )
            if value.dtype.kind not in "fiu":
                raise TypeError(f"参数 {name} 的状态必须是实数数组")
            pending.append((parameter, value.astype(parameter.dtype, copy=True)))

        for parameter, value in pending:
            parameter.data = value
        return {"missing_keys": missing, "unexpected_keys": unexpected}

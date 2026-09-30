"""Torch 风格后端的轻量入口检查。"""

from typing import cast

from .protocols import TorchBackendProtocol


PROTOCOL_VERSION = "1.0"

_REQUIRED_ATTRIBUTES = (
    "float32",
)

_REQUIRED_CALLABLES = (
    "Tensor",
    "tensor",
    "manual_seed",
    "no_grad",
    "nn.Module",
    "nn.Linear",
    "nn.ReLU",
    "nn.Sequential",
    "nn.MSELoss",
    "optim.SGD",
    "utils.data.TensorDataset",
    "utils.data.DataLoader",
)


class BackendProtocolError(TypeError):
    """后端未暴露 Torch Protocol v1 所需成员。"""

    def __init__(self, missing_members):
        self.missing_members = tuple(missing_members)
        members = ", ".join(self.missing_members)
        super().__init__(
            "backend does not satisfy Torch Protocol v1; "
            f"missing or non-callable members: {members}"
        )


_MISSING = object()


def _resolve(candidate, dotted_path):
    value = candidate
    for part in dotted_path.split("."):
        try:
            value = getattr(value, part)
        except Exception:
            return _MISSING
    return value


def validate_backend(candidate) -> TorchBackendProtocol:
    """检查必需入口并返回候选后端。

    此检查只验证成员存在性与构造器可调用性；张量数值和自动微分语义由
    `torch_protocol` 的 conformance tests 验证。
    """

    missing = []
    for path in _REQUIRED_ATTRIBUTES:
        if _resolve(candidate, path) is _MISSING:
            missing.append(path)

    for path in _REQUIRED_CALLABLES:
        value = _resolve(candidate, path)
        if value is _MISSING or not callable(value):
            missing.append(path)

    if missing:
        raise BackendProtocolError(missing)
    return cast(TorchBackendProtocol, candidate)

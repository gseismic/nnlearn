"""独立的 Torch 风格结构化接口协议。"""

__version__ = "0.1.0"

from .protocols import (
    DataLoaderProtocol,
    DatasetProtocol,
    LossProtocol,
    ModuleProtocol,
    OptimizerProtocol,
    TensorProtocol,
    TorchBackendProtocol,
)
from .validation import (
    PROTOCOL_VERSION,
    BackendProtocolError,
    validate_backend,
)

__all__ = [
    "BackendProtocolError",
    "DataLoaderProtocol",
    "DatasetProtocol",
    "LossProtocol",
    "ModuleProtocol",
    "OptimizerProtocol",
    "PROTOCOL_VERSION",
    "TensorProtocol",
    "TorchBackendProtocol",
    "__version__",
    "validate_backend",
]

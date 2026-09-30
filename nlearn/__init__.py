from .log import log_function_call
from .function import Function

from .settings import log_settings, runtime_settings
from .functional import *
from .tensor import (
    Tensor, no_grad, allclose, arange, as_tensor, eye, from_numpy, full,
    full_like, rand, rand_like, randn, randn_like, randint, ones, ones_like,
    tensor, zeros, zeros_like,
    float32, float64, int64, long, register_ops
)
from .backend import device
from .misc import *
from . import cuda

from .version import __version__

register_ops()

# 兼容 `torch.utils.data.DataLoader` 形式的访问。
from .utils import data

# 暴露与 PyTorch 相同的后端命名空间，并在导入时检查 Torch Protocol v1。
from . import nn, optim
from torch_protocol import validate_backend as _validate_backend
import sys as _sys

_validate_backend(_sys.modules[__name__])
del _validate_backend, _sys

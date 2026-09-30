"""与具体张量框架无关的 Torch 风格公共接口。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    runtime_checkable,
)


_TensorT = TypeVar("_TensorT", bound="TensorProtocol")
_ModuleT = TypeVar("_ModuleT", bound="ModuleProtocol")


@runtime_checkable
class TensorProtocol(Protocol):
    """跨后端张量 v1。协议不暴露后端私有的 ``data`` 表示。"""

    shape: Sequence[int]
    ndim: int
    dtype: Any
    device: Any
    requires_grad: bool
    grad: Optional["TensorProtocol"]

    def __add__(self: _TensorT, other: Any) -> _TensorT: ...

    def __radd__(self: _TensorT, other: Any) -> _TensorT: ...

    def __sub__(self: _TensorT, other: Any) -> _TensorT: ...

    def __rsub__(self: _TensorT, other: Any) -> _TensorT: ...

    def __mul__(self: _TensorT, other: Any) -> _TensorT: ...

    def __rmul__(self: _TensorT, other: Any) -> _TensorT: ...

    def __truediv__(self: _TensorT, other: Any) -> _TensorT: ...

    def __rtruediv__(self: _TensorT, other: Any) -> _TensorT: ...

    def __pow__(self: _TensorT, exponent: Any) -> _TensorT: ...

    def __matmul__(self: _TensorT, other: Any) -> _TensorT: ...

    def __getitem__(self: _TensorT, index: Any) -> _TensorT: ...

    def backward(self) -> None:
        """对单元素张量反向传播；跨后端代码不传入显式梯度。"""
        ...

    def reshape(self: _TensorT, *shape: Any) -> _TensorT: ...

    def sum(self: _TensorT, dim: Any = None, keepdim: bool = False) -> _TensorT: ...

    def mean(self: _TensorT, dim: Any = None, keepdim: bool = False) -> _TensorT: ...

    def clone(self: _TensorT) -> _TensorT: ...

    def detach(self) -> "TensorProtocol": ...

    def to(
        self, *, device: Any = None, dtype: Any = None
    ) -> "TensorProtocol": ...

    def cpu(self) -> "TensorProtocol": ...

    def numpy(self) -> Any: ...

    def tolist(self) -> Any: ...

    def item(self) -> Any: ...

    def size(self, dim: Optional[int] = None) -> Any: ...


@runtime_checkable
class ModuleProtocol(Protocol):
    """可调用、可切换训练状态的模型接口。"""

    training: bool

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def parameters(self) -> Iterable[TensorProtocol]: ...

    def train(self: _ModuleT, mode: bool = True) -> _ModuleT: ...

    def eval(self: _ModuleT) -> _ModuleT: ...

    def to(
        self: _ModuleT, *, device: Any = None, dtype: Any = None
    ) -> _ModuleT: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs: Any) -> Any: ...


@runtime_checkable
class LossProtocol(Protocol):
    """将输入与目标映射为可反向传播损失的接口。"""

    def __call__(
        self, input: TensorProtocol, target: TensorProtocol
    ) -> TensorProtocol: ...


@runtime_checkable
class OptimizerProtocol(Protocol):
    """更新一组可训练参数的接口。"""

    def zero_grad(self, set_to_none: bool = ...) -> None:
        """清除参数梯度；不承诺省略参数时 grad 是 None 还是零张量。"""
        ...

    def step(self) -> None: ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """通过长度和索引提供样本的数据集接口。"""

    def __getitem__(self, index: Any) -> Any: ...

    def __len__(self) -> int: ...


@runtime_checkable
class DataLoaderProtocol(Protocol):
    """按批次遍历数据集的接口。"""

    def __iter__(self) -> Iterator[Any]: ...

    def __len__(self) -> int: ...


class TensorFactoryProtocol(Protocol):
    def __call__(
        self,
        data: Any,
        *,
        dtype: Any = None,
        device: Any = None,
        requires_grad: bool = False,
    ) -> TensorProtocol: ...


class LinearFactoryProtocol(Protocol):
    def __call__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> ModuleProtocol: ...


class NoArgumentModuleFactoryProtocol(Protocol):
    def __call__(self) -> ModuleProtocol: ...


class SequentialFactoryProtocol(Protocol):
    def __call__(self, *modules: ModuleProtocol) -> ModuleProtocol: ...


class MSELossFactoryProtocol(Protocol):
    def __call__(
        self, reduction: Literal["none", "mean", "sum"] = "mean"
    ) -> LossProtocol: ...


class SGDFactoryProtocol(Protocol):
    def __call__(
        self, params: Iterable[TensorProtocol], lr: float
    ) -> OptimizerProtocol: ...


class TensorDatasetFactoryProtocol(Protocol):
    def __call__(self, *tensors: TensorProtocol) -> DatasetProtocol: ...


class DataLoaderFactoryProtocol(Protocol):
    def __call__(
        self,
        dataset: DatasetProtocol,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoaderProtocol: ...


class NoGradFactoryProtocol(Protocol):
    def __call__(self) -> AbstractContextManager[Any]: ...


class NNNamespaceProtocol(Protocol):
    Module: Type[ModuleProtocol]
    Linear: LinearFactoryProtocol
    ReLU: NoArgumentModuleFactoryProtocol
    Sequential: SequentialFactoryProtocol
    MSELoss: MSELossFactoryProtocol


class OptimizerNamespaceProtocol(Protocol):
    SGD: SGDFactoryProtocol


class DataNamespaceProtocol(Protocol):
    TensorDataset: TensorDatasetFactoryProtocol
    DataLoader: DataLoaderFactoryProtocol


class UtilsNamespaceProtocol(Protocol):
    data: DataNamespaceProtocol


@runtime_checkable
class TorchBackendProtocol(Protocol):
    """应用程序可以接收的 Torch 风格后端命名空间。"""

    Tensor: Type[TensorProtocol]
    tensor: TensorFactoryProtocol
    float32: Any
    manual_seed: Callable[[int], Any]
    no_grad: NoGradFactoryProtocol
    nn: NNNamespaceProtocol
    optim: OptimizerNamespaceProtocol
    utils: UtilsNamespaceProtocol

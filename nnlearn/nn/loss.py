import numpy as np
from torch_protocol import LossProtocol
from nnlearn.function import Function
from nnlearn.tensor import Tensor
from nnlearn import backend
from nnlearn.settings import Config
from .module import Module


def _check_reduction(reduction):
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            "reduction must be one of 'none', 'mean', or 'sum'"
        )
    return reduction


def _sum_to_input_shape(grad, shape):
    if grad is None or grad.shape == shape:
        return grad
    from nnlearn.functional.matrix import sum_to

    return sum_to(grad, shape)


def _array_scalar_bool(value):
    return bool(backend.as_numpy(value).item())


def _class_weight_data(weight, input_data, num_classes, dtype):
    if weight is None:
        return None

    data = weight.data if isinstance(weight, Tensor) else weight
    data = backend.to_device(data, backend.device_of(input_data))
    if data.shape != (num_classes,):
        raise ValueError(
            "cross_entropy weight must have shape (C,), "
            f"got {data.shape}"
        )
    return data.astype(dtype)


class MSELoss(Module, LossProtocol):
    """可复用的均方误差模块，每次调用创建独立节点以隔离计算图。"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = _check_reduction(reduction)  # 配置规约方式，不保存前向缓存。

    def forward(self, input, target):
        return _MSELossFunction(self.reduction)(input, target)


class _MSELossFunction(Function):
    """保存单次 MSE 调用的状态，使模块复用不会覆盖待反传的输入。"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = _check_reduction(reduction)

    def forward(self, input, target):
        xp = backend.get_array_module(input)
        loss = (input - target) ** 2
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return xp.sum(loss)
        return xp.mean(loss)

    def backward(self, grad_output):
        input, target = self.inputs
        diff = input - target
        use_float32 = self.reduction == "mean" and diff.dtype == np.float16
        if use_float32:
            # 元素数可能超出半精度范围；可微转换保留高阶求导依赖。
            diff = diff.to(dtype=np.float32)
        grad_input = diff + diff
        if self.reduction == "mean":
            # 分母沿用中间计算精度，避免半精度溢出或 float32 被提升为 float64。
            normalizer = Tensor(
                diff.numel(), device=diff.device, dtype=diff.dtype,
                requires_grad=False,
            )
            grad_input = grad_input / normalizer
        grad_input = grad_input * grad_output
        grad_target = -grad_input
        grads = (
            _sum_to_input_shape(grad_input, input.shape),
            _sum_to_input_shape(grad_target, target.shape),
        )
        if use_float32:
            # 应用上游梯度并还原广播之后再恢复精度，避免提前下溢或舍入。
            return tuple(grad.to(dtype=np.float16) for grad in grads)
        return grads


class CrossEntropyLoss(Module, LossProtocol):
    """可复用的交叉熵模块，将概率与标签缓存保留在各次调用的节点中。"""

    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean",
        label_smoothing=0.0,
    ):
        super().__init__()
        if label_smoothing < 0.0 or label_smoothing > 1.0:
            raise ValueError("label_smoothing must be between 0.0 and 1.0")
        self.weight = weight  # 可选的类别权重，Tensor 权重可随模块迁移。
        self.ignore_index = ignore_index  # 不参与损失的标签值。
        self.reduction = _check_reduction(reduction)  # 输出的规约方式。
        self.label_smoothing = float(label_smoothing)  # 标签平滑系数。

    def forward(self, input, target):
        return _CrossEntropyLossFunction(
            self.weight, self.ignore_index, self.reduction, self.label_smoothing,
        )(input, target)


class _CrossEntropyLossFunction(Function):
    """保存单次交叉熵的前向缓存，避免合并多个损失时丢失梯度。"""
    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean",
        label_smoothing=0.0,
    ):
        super().__init__()
        if label_smoothing < 0.0 or label_smoothing > 1.0:
            raise ValueError("label_smoothing must be between 0.0 and 1.0")
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction)
        self.label_smoothing = float(label_smoothing)

    def forward(self, input, target):
        if input.ndim < 2:
            raise ValueError("cross_entropy expects input shape (N, C, ...)")

        xp = backend.get_array_module(input)
        num_classes = input.shape[1]
        if target.dtype not in (np.dtype('int64'), np.dtype('uint8')):
            raise TypeError('cross_entropy target must use int64 or uint8 class indices')
        target = target.astype("int64")
        expected_target_shape = (input.shape[0],) + input.shape[2:]
        if target.shape != expected_target_shape:
            raise ValueError(
                "cross_entropy target shape must match input without "
                f"class dimension: expected {expected_target_shape}, "
                f"got {target.shape}"
            )

        if input.ndim == 2:
            logits = input.reshape((-1, num_classes))
        else:
            logits = xp.moveaxis(input, 1, -1).reshape((-1, num_classes))

        target_flat = target.reshape(-1)
        valid_mask = target_flat != self.ignore_index
        invalid_target = valid_mask & (
            (target_flat < 0) | (target_flat >= num_classes)
        )
        if _array_scalar_bool(xp.any(invalid_target)):
            raise ValueError(
                "cross_entropy target values must be in [0, C) or "
                "equal to ignore_index"
            )

        safe_target = target_flat.copy()
        safe_target = xp.where(valid_mask, safe_target, xp.zeros_like(safe_target))

        # 指数和、损失和及均值分母使用单精度，不能在溢出后才恢复 dtype。
        work_logits = logits.astype(np.float32) if logits.dtype == np.float16 else logits
        shifted = work_logits - xp.max(work_logits, axis=1, keepdims=True)
        exp_logits = xp.exp(shifted)
        self.probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
        log_probs = shifted - xp.log(xp.sum(exp_logits, axis=1, keepdims=True))

        weight_data = _class_weight_data(
            self.weight, input, num_classes, log_probs.dtype,
        )
        if weight_data is None:
            sample_weight = xp.ones(target_flat.shape, dtype=log_probs.dtype)
        else:
            sample_weight = weight_data[safe_target]
        sample_weight = xp.where(
            valid_mask, sample_weight, xp.zeros_like(sample_weight),
        )

        batch_index = xp.arange(target_flat.shape[0])
        nll = -log_probs[batch_index, safe_target]
        if weight_data is not None:
            nll = nll * sample_weight

        if self.label_smoothing:
            if weight_data is None:
                smooth_loss = -xp.sum(log_probs, axis=1)
            else:
                smooth_loss = -xp.sum(
                    log_probs * weight_data.reshape((1, num_classes)),
                    axis=1,
                )
            losses = (
                (1.0 - self.label_smoothing) * nll
                + (self.label_smoothing / num_classes) * smooth_loss
            )
        else:
            losses = nll
        losses = xp.where(valid_mask, losses, xp.zeros_like(losses))

        self.input_shape = input.shape
        self.target_shape = target.shape
        self.num_classes = num_classes
        self.target = target_flat
        self.safe_target = safe_target
        self.valid_mask = valid_mask
        self.weight_data = weight_data
        self.sample_weight = sample_weight
        self.mean_normalizer = None
        self.mean_normalizer_nonzero = True

        if self.reduction == "none":
            output = losses.reshape(target.shape)
            return output.astype(input.dtype) if input.dtype == np.float16 else output
        if self.reduction == "sum":
            output = xp.sum(losses)
            return output.astype(input.dtype) if input.dtype == np.float16 else output

        if weight_data is None:
            normalizer = xp.sum(valid_mask.astype(log_probs.dtype))
        else:
            normalizer = xp.sum(sample_weight)
        self.mean_normalizer = normalizer
        self.mean_normalizer_nonzero = _array_scalar_bool(normalizer != 0)
        output = xp.sum(losses) / normalizer
        return output.astype(input.dtype) if input.dtype == np.float16 else output

    def backward(self, grad_output):
        if Config.enable_backprop:
            return self._backward_with_graph(grad_output)
        xp = backend.get_array_module(self.probs)
        batch_index = xp.arange(self.target.shape[0])

        if self.label_smoothing:
            smooth = self.label_smoothing / self.num_classes
            target_dist = xp.full(self.probs.shape, smooth,
                                  dtype=self.probs.dtype)
            target_dist[batch_index, self.safe_target] += (
                1.0 - self.label_smoothing
            )
            if self.weight_data is None:
                grad = self.probs - target_dist
            else:
                target_dist = target_dist * self.weight_data.reshape(
                    (1, self.num_classes)
                )
                grad = (
                    self.probs * xp.sum(target_dist, axis=1, keepdims=True)
                    - target_dist
                )
        else:
            grad = self.probs.copy()
            grad[batch_index, self.safe_target] -= 1.0
            if self.weight_data is not None:
                grad = grad * self.sample_weight.reshape((-1, 1))

        grad = xp.where(
            self.valid_mask.reshape((-1, 1)), grad, xp.zeros_like(grad),
        )
        if self.reduction == "mean":
            if self.mean_normalizer_nonzero:
                grad = grad / self.mean_normalizer
            else:
                grad = xp.zeros_like(grad)
        elif self.reduction == "none":
            grad = grad * grad_output.data.reshape((-1, 1))

        if self.reduction != "none":
            grad = grad * grad_output.data

        if len(self.input_shape) == 2:
            grad_input = grad.reshape(self.input_shape)
        else:
            grad_input = grad.reshape(self.target_shape + (self.num_classes,))
            grad_input = xp.moveaxis(grad_input, -1, 1)
        return Tensor(grad_input), None

    def _backward_with_graph(self, grad_output):
        """重算可微概率，并用常量标签分布保留 logits 和上游的二阶依赖。"""
        from nnlearn.functional.numeric import softmax

        x = self.inputs[0]
        if x.dtype == np.float16:
            x, grad_output = x.to(dtype=np.float32), grad_output.to(dtype=np.float32)
        if x.ndim > 2:
            x = x.permute((0,) + tuple(range(2, x.ndim)) + (1,))
        probs = softmax(x.reshape(-1, self.num_classes), axis=1)
        xp = backend.get_array_module(self.probs)
        distribution = xp.full(self.probs.shape, self.label_smoothing / self.num_classes,
                               dtype=self.probs.dtype)
        distribution[xp.arange(self.target.size), self.safe_target] += 1 - self.label_smoothing
        if self.weight_data is not None:
            distribution *= self.weight_data.reshape(1, -1)
        distribution *= self.valid_mask.reshape(-1, 1)
        target = Tensor(distribution, requires_grad=False)
        grad = probs * target.sum(axis=1, keepdims=True) - target
        if self.reduction == 'mean':
            if self.mean_normalizer_nonzero:
                grad = grad / Tensor(self.mean_normalizer, requires_grad=False)
            else:
                grad = probs * 0
        if self.reduction == 'none':
            grad = grad * grad_output.reshape(-1, 1)
        else:
            grad = grad * grad_output
        if len(self.input_shape) == 2:
            return grad.reshape(self.input_shape), None
        grad = grad.reshape(self.target_shape + (self.num_classes,))
        return grad.permute((0, grad.ndim - 1) + tuple(range(1, grad.ndim - 1))), None

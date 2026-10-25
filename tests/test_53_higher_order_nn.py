"""对照 PyTorch 和独立有限差分，验证神经网络算子的高阶依赖。"""

import numpy as np
import pytest

import nnlearn as nl
from nnlearn import nn
from nnlearn.nn.conv import conv2d
from nnlearn.nn.normalization import LayerNormFunction

torch = pytest.importorskip('torch')


def _compare_derivatives(data, nl_function, torch_function, dtype=np.float64):
    """比较全部输入的一阶导和联合方向二阶导，包含跨参数的 Hessian 项。"""
    rng = np.random.default_rng(64)
    inputs = [nl.tensor(value.astype(dtype), requires_grad=True) for value in data]
    references = [torch.tensor(value.astype(dtype), requires_grad=True) for value in data]
    result, reference = nl_function(*inputs), torch_function(*references)
    loss = (result * result).sum()
    reference_loss = (reference * reference).sum()
    loss.backward(create_graph=True)
    gradients = [value.grad for value in inputs]
    reference_gradients = torch.autograd.grad(reference_loss, references, create_graph=True)
    tolerance = 0.03 if dtype == np.float16 else (3e-4 if dtype == np.float32 else 2e-8)
    for value, gradient, expected in zip(inputs, gradients, reference_gradients):
        assert gradient is not None and gradient.requires_grad
        assert gradient.dtype == value.dtype
        np.testing.assert_allclose(gradient.numpy(), expected.detach().numpy(),
                                   atol=tolerance, rtol=tolerance)
        value.zero_grad()
    directions = [rng.normal(size=value.shape).astype(dtype) for value in data]
    seed = sum((gradient * nl.tensor(direction, requires_grad=False)).sum()
               for gradient, direction in zip(gradients, directions))
    reference_seed = sum((gradient * torch.tensor(direction)).sum()
                         for gradient, direction in zip(reference_gradients, directions))
    seed.backward()
    reference_second = torch.autograd.grad(reference_seed, references)
    for value, expected in zip(inputs, reference_second):
        assert value.grad is not None
        assert value.grad.dtype == value.dtype
        np.testing.assert_allclose(value.grad.numpy(), expected.detach().numpy(),
                                   atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('shape', [(4,), (2, 2)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_layernorm_first_and_cross_second_derivatives(shape, dtype):
    """多轴 LayerNorm 的输入、缩放和偏置均参与联合二阶求导。"""
    rng = np.random.default_rng(641)
    data = [rng.normal(size=(2, 3) + shape), rng.normal(size=shape), rng.normal(size=shape)]
    _compare_derivatives(
        data,
        lambda x, weight, bias: LayerNormFunction(shape)(x, weight, bias),
        lambda x, weight, bias: torch.nn.functional.layer_norm(x, shape, weight, bias),
        dtype,
    )


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('smoothing', [0., 0.2])
@pytest.mark.parametrize('spatial', [False, True])
def test_cross_entropy_second_derivatives(reduction, smoothing, spatial):
    """类别权重、忽略标签及平滑后的交叉熵保留 logits 和非线性上游依赖。"""
    rng = np.random.default_rng(642)
    shape = (2, 3, 2) if spatial else (4, 3)
    labels = np.array([[0, -100], [2, 1]]) if spatial else np.array([0, 2, -100, 1])
    weights = np.array([0.5, 1.5, 2.])
    _compare_derivatives(
        [rng.normal(size=shape)],
        lambda x: nn.functional.cross_entropy(x, nl.tensor(labels), weight=weights,
                                             reduction=reduction, label_smoothing=smoothing),
        lambda x: torch.nn.functional.cross_entropy(
            x, torch.tensor(labels), weight=torch.tensor(weights),
            reduction=reduction, label_smoothing=smoothing),
    )


@pytest.mark.parametrize('stride,padding,bias', [(1, 0, True), (2, 1, False),
                                               ((1, 2), (1, 1), True)])
def test_conv2d_first_and_cross_second_derivatives(stride, padding, bias):
    """卷积的 stride、padding 和可选偏置路径均检查输入/权重交叉二阶导。"""
    rng = np.random.default_rng(643)
    data = [rng.normal(size=(1, 2, 4, 5)), rng.normal(size=(2, 2, 2, 3))]
    if bias:
        data.append(rng.normal(size=(2,)))
    _compare_derivatives(
        data,
        lambda *values: conv2d(*values, stride=stride, padding=padding),
        lambda *values: torch.nn.functional.conv2d(*values, stride=stride, padding=padding),
    )


@pytest.mark.parametrize('kind', ['layernorm', 'cross_entropy', 'conv2d'])
def test_directional_derivatives_against_finite_differences(kind):
    """独立中央差分检查一阶及二阶方向导数，避免只复刻参考框架的假设。"""
    rng = np.random.default_rng(644)
    shape = (2, 3) if kind != 'conv2d' else (1, 1, 3, 3)
    data, direction = rng.normal(size=shape), rng.normal(size=shape)
    if kind == 'layernorm':
        def function(x):
            return LayerNormFunction(3)(
                x, nl.tensor([0.7, 1.2, -0.5], requires_grad=False),
                nl.tensor([0.1, -0.2, 0.3], requires_grad=False))
    elif kind == 'cross_entropy':
        def function(x):
            return nn.CrossEntropyLoss()(x, nl.tensor([0, 2]))
    else:
        def function(x):
            return conv2d(x, nl.tensor(np.array([[[[0.5, -0.2], [0.8, 0.3]]]]),
                                      requires_grad=False), padding=1)

    def scalar(value):
        with nl.no_grad():
            output = function(nl.tensor(value))
            return (output * output).sum().item()

    x = nl.tensor(data, requires_grad=True)
    output = function(x)
    (output * output).sum().backward(create_graph=True)
    first = (x.grad * nl.tensor(direction, requires_grad=False)).sum()
    first_value = first.item()
    x.zero_grad()
    first.backward()
    second_value = float(np.sum(x.grad.numpy() * direction))
    step = 1e-4
    center, plus, minus = scalar(data), scalar(data + step * direction), scalar(data - step * direction)
    assert first_value == pytest.approx((plus - minus) / (2 * step), rel=1e-5, abs=1e-6)
    assert second_value == pytest.approx((plus - 2 * center + minus) / step ** 2,
                                        rel=2e-5, abs=3e-6)


def test_layernorm_repeated_backward_accumulates_without_reusing_old_seed():
    """同一计算图重复反向仍只传播本次新梯度。"""
    x = nl.tensor([[0.2, 1.1, -0.5]], requires_grad=True)
    model = nn.LayerNorm(3)
    output = model(x)
    loss = (output * output).sum()
    loss.backward(create_graph=True)
    first = x.grad.numpy().copy()
    loss.backward(create_graph=True)
    np.testing.assert_allclose(x.grad.numpy(), first * 2, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('kind', ['layernorm', 'cross_entropy', 'conv2d'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_higher_order_dtype_and_parameter_dependencies(kind, dtype):
    """较低精度下仍保留输入和参数的二阶依赖，并恢复每条边的梯度精度。"""
    rng = np.random.default_rng(645)
    if kind == 'layernorm':
        _compare_derivatives(
            [rng.normal(size=(2, 4)), np.ones(4), np.zeros(4)],
            lambda x, weight, bias: LayerNormFunction(4)(x, weight, bias),
            lambda x, weight, bias: torch.nn.functional.layer_norm(x, (4,), weight, bias),
            dtype,
        )
    elif kind == 'cross_entropy':
        _compare_derivatives(
            [rng.normal(size=(2, 4))],
            lambda x: nn.CrossEntropyLoss()(x, nl.tensor([1, 3])),
            lambda x: torch.nn.functional.cross_entropy(x, torch.tensor([1, 3])),
            dtype,
        )
    else:
        _compare_derivatives(
            [rng.normal(size=(1, 1, 3, 3)), rng.normal(size=(1, 1, 2, 2)), np.zeros(1)],
            lambda x, weight, bias: conv2d(x, weight, bias),
            lambda x, weight, bias: torch.nn.functional.conv2d(x, weight, bias),
            dtype,
        )

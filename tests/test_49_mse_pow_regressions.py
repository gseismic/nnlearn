"""覆盖复审发现的 MSE 半精度归一化与数组指数反向回归。"""

import numpy as np
import pytest
pytorch = pytest.importorskip("torch")

import nnlearn as torch


@pytest.mark.parametrize('size', [65000, 65520, 100000])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_large_mse_mean_keeps_nonzero_gradient_and_dtype(size, dtype):
    """元素数量越过半精度上限后，均值梯度仍应非零且保持输入精度。"""
    x = torch.tensor(np.ones(size, dtype=dtype), requires_grad=True)
    target = torch.tensor(np.zeros(size, dtype=dtype))
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        loss = torch.nn.MSELoss()(x, target)
        loss.backward()
    assert loss.item() == 1
    assert x.grad.dtype == dtype
    np.testing.assert_allclose(
        x.grad.numpy(), np.full(size, 2 / size, dtype=dtype), rtol=1e-6,
    )


def test_large_half_mse_keeps_broadcasted_second_derivative():
    """半精度中间转换与梯度还原必须保留输入、广播目标的二阶依赖。"""
    size = 100000
    x = torch.tensor(np.ones(size, dtype=np.float16), requires_grad=True)
    target = torch.tensor(np.float16(0), requires_grad=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        torch.nn.MSELoss()(x, target).backward(create_graph=True)
        assert x.grad.dtype == np.float16
        assert target.grad.dtype == np.float16
        np.testing.assert_allclose(target.grad.item(), -2, rtol=0.002)
        gradient = x.grad
        x.zero_grad()
        target.zero_grad()
        gradient.sum().backward()
    assert x.grad.dtype == np.float16
    assert target.grad.dtype == np.float16
    np.testing.assert_allclose(
        x.grad.numpy(), np.full(size, 2 / size, dtype=np.float16), rtol=0.002,
    )
    np.testing.assert_allclose(target.grad.item(), -2, rtol=0.002)


def test_half_mse_applies_upstream_gradient_before_casting_back():
    """先应用上游梯度再恢复半精度，避免可表示的最终梯度提前下溢。"""
    x = torch.tensor(np.full(100000, 1e-4, dtype=np.float16), requires_grad=True)
    target = torch.tensor(np.zeros(x.shape, dtype=np.float16))
    scale = torch.tensor(np.float16(1000))
    (torch.nn.MSELoss()(x, target) * scale).backward()
    expected = (2 * x.numpy().astype(np.float32) / x.numel() * 1000).astype(np.float16)
    assert x.grad.dtype == np.float16
    np.testing.assert_array_equal(x.grad.numpy(), expected)


@pytest.mark.parametrize('data,exponent', [
    ([2., 3.], np.array([2., 3.])),
    ([0., 3., -2.], np.array([0., 2., 3.])),
    ([0., 3., -2.], np.array([0, 2, 3], dtype=np.int64)),
    ([4., 2.], np.array([0.5, -1.])),
    ([[0., 2.], [0., 3.]], np.array([0., 3.])),
    ([[2.], [3.]], np.array([0., 1., 2.])),
    (2., np.array([0., 1., 2.])),
    ([0., 2.], np.array(0.)),
    ([], np.array([])),
])
def test_array_exponent_forward_and_gradient_match_pytorch(data, exponent):
    """数组指数支持逐元素零指数和广播，梯度应还原到原输入形状。"""
    data = np.asarray(data, dtype=np.float64)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = x ** exponent
        result.sum().backward()
    expected = reference ** pytorch.tensor(exponent)
    expected.sum().backward()
    np.testing.assert_allclose(result.numpy(), expected.detach().numpy())
    assert x.grad.shape == x.shape
    np.testing.assert_allclose(x.grad.numpy(), reference.grad.numpy())


def test_mixed_zero_array_exponent_keeps_higher_derivatives():
    """混合零指数在零底数处既不能计算负次幂，也不能切断其余元素的图。"""
    x = torch.tensor([0., 2., 3.], requires_grad=True)
    exponent = np.array([0., 2., 3.])
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        (x ** exponent).sum().backward(create_graph=True)
        np.testing.assert_allclose(x.grad.numpy(), [0., 4., 27.])
        gradient = x.grad
        x.zero_grad()
        gradient.sum().backward(create_graph=True)
        np.testing.assert_allclose(x.grad.numpy(), [0., 2., 18.])
        gradient = x.grad
        x.zero_grad()
        gradient.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0., 0., 6.])

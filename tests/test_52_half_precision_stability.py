"""验证半精度规约的真实溢出边界、解析梯度及连续训练。"""

import numpy as np
import pytest

import nnlearn as nl
from nnlearn import nn


def test_integer_logits_keep_historical_floating_output():
    """仅恢复半精度输出，不能把整数输入的既有浮点结果截断。"""
    x = nl.tensor([[0, 0, 0]], dtype=np.int64, requires_grad=False)
    for name in ['softmax', 'log_softmax']:
        output = getattr(x, name)(dim=-1)
        assert np.issubdtype(output.dtype, np.floating)
    for reduction in ['none', 'sum', 'mean']:
        output = nn.CrossEntropyLoss(reduction=reduction)(x, nl.tensor([0]))
        assert np.issubdtype(output.dtype, np.floating)
        np.testing.assert_allclose(output.numpy(), np.log(3))


@pytest.mark.parametrize('name', ['softmax', 'log_softmax'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_large_uniform_distribution_and_gradient(name, dtype):
    """类别数超过半精度最大有限值时，均匀分布及其梯度仍正确。"""
    count = 70000
    x = nl.tensor(np.zeros((2, count), dtype=dtype), requires_grad=True)
    output = getattr(x, name)(dim=-1)
    expected = 1 / count if name == 'softmax' else -np.log(count)
    assert output.dtype == dtype
    assert np.isfinite(output.numpy()).all()
    np.testing.assert_allclose(output.numpy(), expected, rtol=0.004, atol=1e-7)
    if name == 'softmax':
        np.testing.assert_allclose(output.numpy().astype(np.float64).sum(-1), 1,
                                   rtol=0.004)
    upstream = np.linspace(-0.25, 0.25, count).astype(dtype)[None, :]
    (output * nl.tensor(upstream, requires_grad=False)).sum().backward()
    expected_grad = upstream.astype(np.float64)
    if name == 'softmax':
        expected_grad = (expected_grad - expected_grad.mean()) / count
    else:
        expected_grad = expected_grad - expected_grad.sum() / count
    assert x.grad.dtype == dtype
    np.testing.assert_allclose(x.grad.numpy(), np.broadcast_to(expected_grad, x.shape),
                               rtol=0.008, atol=1e-7)


@pytest.mark.parametrize('weight', [False, True])
@pytest.mark.parametrize('smoothing', [0., 0.2])
@pytest.mark.parametrize('shape', [(2, 70000), (100000, 2)])
def test_half_cross_entropy_large_reductions(shape, weight, smoothing):
    """损失求和、标签计数及加权分母都必须在恢复半精度之前完成。"""
    batch, classes = shape
    x = nl.tensor(np.zeros(shape, dtype=np.float16), requires_grad=True)
    target_data = np.arange(batch) % classes
    target = nl.tensor(target_data.astype(np.int64), requires_grad=False)
    weights = np.ones(classes, dtype=np.float16) * 1.5 if weight else None
    loss = nn.functional.cross_entropy(x, target, weight=weights,
                                      label_smoothing=smoothing)
    assert loss.dtype == np.float16
    assert np.isfinite(loss.item())
    assert loss.item() == pytest.approx(np.log(classes), rel=0.003)
    loss.backward()
    expected = np.full(shape, (1 - smoothing) / classes, dtype=np.float64)
    expected[np.arange(batch), target_data] -= 1 - smoothing
    expected /= batch
    assert x.grad.dtype == np.float16
    np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-7, rtol=0.005)


@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_cross_entropy_weighted_ignored_smoothing_reference(reduction, dtype):
    """高维输入、忽略标签和类别权重组合不能因内部升精度改变语义。"""
    torch = pytest.importorskip('torch')
    data = np.linspace(-3, 3, 48).reshape(2, 3, 2, 4).astype(dtype)
    labels = np.array([[[0, 1, 2, -100], [1, 2, 0, 1]],
                       [[2, 0, 1, 2], [0, -100, 1, 2]]])
    weights = np.array([0.5, 1.5, 2.], dtype=dtype)
    x = nl.tensor(data, requires_grad=True)
    y = nn.functional.cross_entropy(x, nl.tensor(labels), weight=weights,
                                   reduction=reduction, label_smoothing=0.2)
    rx = torch.tensor(data.astype(np.float64), requires_grad=True)
    ry = torch.nn.functional.cross_entropy(
        rx, torch.tensor(labels), weight=torch.tensor(weights.astype(np.float64)),
        reduction=reduction, label_smoothing=0.2,
    )
    (y * y).sum().backward()
    (ry * ry).sum().backward()
    assert y.dtype == dtype
    assert x.grad.dtype == dtype
    np.testing.assert_allclose(y.numpy(), ry.detach().numpy(), rtol=0.004, atol=2e-3)
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), rtol=0.01, atol=3e-3)


@pytest.mark.parametrize('kind', ['layer', 'batch1d', 'batch2d'])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_normalization_large_values_matches_reference(kind, dtype):
    """对照高精度参考检查归一化的输出、输入和仿射参数梯度。"""
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(52)
    shape = (2, 128) if kind == 'layer' else ((128, 3) if kind == 'batch1d'
                                              else (4, 3, 8, 8))
    data = (rng.normal(size=shape) * 100).astype(dtype)
    if kind == 'layer':
        model, reference = nn.LayerNorm(128), torch.nn.LayerNorm(128)
    elif kind == 'batch1d':
        model, reference = nn.BatchNorm1d(3), torch.nn.BatchNorm1d(3)
    else:
        model, reference = nn.BatchNorm2d(3), torch.nn.BatchNorm2d(3)
    model.to(dtype=dtype)
    reference.to(dtype=torch.float64)
    x = nl.tensor(data, requires_grad=True)
    rx = torch.tensor(data.astype(np.float64), requires_grad=True)
    y, ry = model(x), reference(rx)
    upstream = rng.normal(size=shape).astype(dtype)
    (y * nl.tensor(upstream, requires_grad=False)).sum().backward()
    (ry * torch.tensor(upstream.astype(np.float64))).sum().backward()
    assert y.dtype == dtype
    assert x.grad.dtype == dtype
    np.testing.assert_allclose(y.numpy(), ry.detach().numpy(), atol=0.003, rtol=0.005)
    np.testing.assert_allclose(x.grad.numpy(), rx.grad.numpy(), atol=3e-5, rtol=0.02)
    for name in ['weight', 'bias']:
        parameter = getattr(model, name)
        assert parameter.grad.dtype == dtype
        np.testing.assert_allclose(parameter.grad.numpy(), getattr(reference, name).grad.numpy(),
                                   rtol=0.01, atol=0.03)
    if kind != 'layer':
        for name in ['running_mean', 'running_var']:
            buffer = getattr(model, name)
            assert buffer.dtype == dtype
            np.testing.assert_allclose(buffer.numpy(), getattr(reference, name).numpy(),
                                       rtol=0.003, atol=0.01)
        model.eval()
        with nl.no_grad():
            assert model(x).dtype == dtype


def test_half_layernorm_preserves_explicit_float32_affine():
    """内部累加精度不能覆盖调用方显式选择的混合精度参数。"""
    model = nn.LayerNorm(128).to(dtype=np.float32)
    x = nl.tensor(np.arange(256, dtype=np.float16).reshape(2, 128))
    output = model(x)
    assert output.dtype == np.float32
    assert np.isfinite(output.numpy()).all()
    assert np.max(np.abs(output.numpy())) > 1


@pytest.mark.parametrize('optimizer_name', ['SGD', 'MomentumSGD', 'Adam', 'AdamW'])
def test_half_normalized_classifier_multistep(optimizer_name):
    """真实组合层经过多次更新后，参数、梯度与优化器状态保持半精度。"""
    nl.manual_seed(52)
    model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.ReLU(), nn.Linear(8, 2))
    model.to(dtype=np.float16)
    options = {'lr': 0.02}
    if optimizer_name in ('Adam', 'AdamW'):
        options['eps'] = 1e-4
    optimizer = getattr(nl.optim, optimizer_name)(model.parameters(), **options)
    data = nl.tensor(np.array([[-100, 30, -20, 5], [100, -30, 20, -5]], dtype=np.float16))
    labels = nl.tensor([0, 1], dtype=np.int64)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, labels)
        losses.append(loss.item())
        assert output.dtype == np.float16
        assert np.isfinite(loss.item())
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad.dtype == np.float16
            assert np.isfinite(parameter.grad.numpy()).all()
        optimizer.step()
        for parameter in model.parameters():
            assert parameter.dtype == np.float16
            assert np.isfinite(parameter.numpy()).all()
        for state in optimizer.state_dict()['state'].values():
            for value in state.values():
                if isinstance(value, nl.Tensor):
                    assert value.dtype == np.float16
    assert losses[-1] < losses[0]

"""约束训练运算的常量推断、梯度精度、高阶依赖及连续优化状态。"""

import numpy as np
import pytest
import torch as pytorch

import nnlearn as torch
from nnlearn import functional as F


FLOAT_DTYPES = [np.float16, np.float32, np.float64]
ARITHMETIC = [
    ('add', lambda x, c: x + c),
    ('sub', lambda x, c: x - c),
    ('rsub', lambda x, c: c - x),
    ('mul', lambda x, c: x * c),
    ('div', lambda x, c: x / c),
    ('rdiv', lambda x, c: c / x),
    ('pow', lambda x, c: x ** c),
]
COMMON_OPS = [
    ('sum', lambda x: x.sum(), lambda x: x.sum()),
    ('mean', lambda x: x.mean(), lambda x: x.mean()),
    ('mean_axis', lambda x: x.mean(dim=1, keepdim=True),
     lambda x: x.mean(dim=1, keepdim=True)),
    ('mean_tuple', lambda x: x.mean(dim=(0, -1)),
     lambda x: x.mean(dim=(0, -1))),
    ('square', F.square, pytorch.square),
    ('sqrt', lambda x: (x + 2).sqrt(), lambda x: (x + 2).sqrt()),
    ('exp', lambda x: x.exp(), lambda x: x.exp()),
    ('log', lambda x: (x + 2).log(), lambda x: (x + 2).log()),
    ('abs', lambda x: x.abs(), lambda x: x.abs()),
    ('relu', lambda x: x.relu(), lambda x: x.relu()),
    ('tanh', lambda x: x.tanh(), lambda x: x.tanh()),
    ('sigmoid', lambda x: x.sigmoid(), lambda x: x.sigmoid()),
    ('softmax', lambda x: x.softmax(dim=1), lambda x: x.softmax(dim=1)),
    ('log_softmax', lambda x: x.log_softmax(dim=1),
     lambda x: x.log_softmax(dim=1)),
    ('clamp', lambda x: x.clamp(min=-0.5, max=1),
     lambda x: x.clamp(min=-0.5, max=1)),
]
EXTREMA = [
    ('max', lambda x: x.max(), lambda x: x.max()),
    ('min', lambda x: x.min(), lambda x: x.min()),
    ('max_axis', lambda x: x.max(dim=1).values,
     lambda x: x.max(dim=1).values),
    ('min_axis', lambda x: x.min(dim=1, keepdim=True).values,
     lambda x: x.min(dim=1, keepdim=True).values),
    ('amax', lambda x: x.amax(dim=1), lambda x: x.amax(dim=1)),
    ('amin', lambda x: x.amin(dim=(0, 1)), lambda x: x.amin(dim=(0, 1))),
    ('maximum', lambda x: F.maximum(x, 2.),
     lambda x: pytorch.maximum(x, x.new_tensor(2.))),
    ('minimum', lambda x: F.minimum(x, 2.),
     lambda x: pytorch.minimum(x, x.new_tensor(2.))),
]
TRAINING_PATHS = [
    ('mean', lambda x: x.mean()),
    ('mul_sum', lambda x: (x * 2).sum()),
    ('pow_sum', lambda x: (x ** 2).sum()),
    ('sigmoid_sum', lambda x: x.sigmoid().sum()),
]


def _assert_matches(actual, expected):
    """同时比较值与精度，避免数值接近掩盖无意的 dtype 提升。"""
    expected = expected.detach().numpy()
    assert actual.dtype == expected.dtype
    tolerances = {
        np.dtype('float16'): (0.02, 0.003),
        np.dtype('float32'): (2e-5, 2e-6),
        np.dtype('float64'): (1e-11, 1e-12),
    }
    rtol, atol = tolerances[actual.dtype]
    np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('shape', [(), (2, 3)])
@pytest.mark.parametrize('constant', [2, 2.5])
@pytest.mark.parametrize('name,operation', ARITHMETIC, ids=[x[0] for x in ARITHMETIC])
def test_python_scalar_arithmetic_preserves_float_context(dtype, shape, constant,
                                                         name, operation):
    """标量正反序算术应遵守浮点上下文，零维张量与矩阵行为一致。"""
    data = np.full(shape, 1.25, dtype=dtype)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    result = operation(x, constant)
    expected = operation(reference, constant)
    _assert_matches(result, expected)
    result.sum().backward()
    expected.sum().backward()
    _assert_matches(x.grad, reference.grad)


def _check_second_derivative(dtype, data, operation, reference_operation):
    """通过非线性上游验证反向公式保留计算图，而非只检查常数梯度。"""
    data = np.asarray(data, dtype=dtype)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    result = operation(x)
    expected = reference_operation(reference)
    _assert_matches(result, expected)
    (result ** 2).sum().backward(create_graph=True, retain_grad=True)
    expected_gradient, = pytorch.autograd.grad(
        (expected ** 2).sum(), reference, create_graph=True,
    )
    assert result.grad.dtype == dtype
    _assert_matches(x.grad, expected_gradient)
    gradient = x.grad
    x.zero_grad()
    gradient.sum().backward()
    expected_second, = pytorch.autograd.grad(expected_gradient.sum(), reference)
    assert x.grad is not None
    _assert_matches(x.grad, expected_second)


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('name,operation,reference_operation', COMMON_OPS,
                         ids=[x[0] for x in COMMON_OPS])
def test_common_ops_keep_output_gradient_and_second_derivative_dtype(
        dtype, name, operation, reference_operation):
    """规约和常用激活的一阶、二阶梯度均应匹配所选浮点精度。"""
    _check_second_derivative(
        dtype, [[-0.9, -0.25, 0.3], [0.8, 1.2, 1.4]],
        operation, reference_operation,
    )


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('operation', [lambda x: x ** 2.5, lambda x: x.sigmoid(),
                                      lambda x: x.relu(), lambda x: x.clamp(min=0., max=1.),
                                      F.square],
                         ids=['pow', 'sigmoid', 'relu', 'clamp', 'square'])
def test_zero_dim_internal_constants_keep_dtype_and_higher_gradients(dtype, operation):
    """零维前向内部常量也必须按输入精度构造，覆盖旧版后端的提升差异。"""
    reference_operation = pytorch.square if operation is F.square else operation
    _check_second_derivative(dtype, 0.5, operation, reference_operation)


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('name,operation,reference_operation', EXTREMA,
                         ids=[x[0] for x in EXTREMA])
def test_extrema_keep_tie_rules_dtype_and_upstream_graph(
        dtype, name, operation, reference_operation):
    """极值重复位置的分配规则、计数精度和非线性上游依赖均须保留。"""
    _check_second_derivative(
        dtype, [[-1, 2, 2], [3, -1, 3]], operation, reference_operation,
    )


@pytest.mark.parametrize('kind', ['tensor_scalar', 'tensor_vector', 'numpy_scalar',
                                 'numpy_0d', 'numpy_vector', 'list'])
@pytest.mark.parametrize('operation', [lambda x, y: x + y, lambda x, y: x * y,
                                      lambda x, y: x / y], ids=['add', 'mul', 'div'])
def test_explicit_tensor_and_array_dtypes_are_not_weak_scalars(kind, operation):
    """显式高精度输入沿用后端提升规则，回传梯度分别恢复每个输入精度。"""
    x = torch.tensor([1., 2.], dtype=np.float32, requires_grad=True)
    other = {
        'tensor_scalar': torch.tensor(2., dtype=np.float64, requires_grad=True),
        'tensor_vector': torch.tensor([2.], dtype=np.float64, requires_grad=True),
        'numpy_scalar': np.float64(2),
        'numpy_0d': np.array(2., dtype=np.float64),
        'numpy_vector': np.array([2.], dtype=np.float64),
        'list': [2.],
    }[kind]
    other_data = other.numpy() if isinstance(other, torch.Tensor) else np.asarray(other)
    expected = operation(x.numpy(), other_data)
    result = operation(x, other)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result.numpy(), expected)
    result.sum().backward()
    assert x.grad.dtype == np.float32
    if isinstance(other, torch.Tensor):
        assert other.dtype == np.float64
        assert other.grad.dtype == np.float64


@pytest.mark.parametrize('shape', [(), (1,)])
def test_mixed_explicit_dtypes_keep_cross_input_second_derivatives(shape):
    """可微的梯度转换不能切断混合精度输入之间的交叉二阶依赖。"""
    x = torch.tensor([0.5, 1.5], dtype=np.float32, requires_grad=True)
    scale = torch.tensor(np.full(shape, 2., dtype=np.float64), requires_grad=True)
    reference = pytorch.tensor(x.numpy(), requires_grad=True)
    # PyTorch 的零维提升规则不同，使用一维广播常量对照相同的混合精度计算。
    ref_scale = pytorch.tensor([2.], dtype=pytorch.float64, requires_grad=True)
    ((x * scale) ** 2).mean().backward(create_graph=True)
    first = pytorch.autograd.grad(
        ((reference * ref_scale) ** 2).mean(), (reference, ref_scale),
        create_graph=True,
    )
    _assert_matches(x.grad, first[0])
    np.testing.assert_allclose(scale.grad.numpy(), first[1].detach().numpy().reshape(shape))
    assert scale.grad.dtype == np.float64
    gradient = x.grad.sum() + scale.grad.sum()
    x.zero_grad()
    scale.zero_grad()
    gradient.backward()
    second = pytorch.autograd.grad(first[0].sum() + first[1].sum(), (reference, ref_scale))
    _assert_matches(x.grad, second[0])
    assert scale.grad.dtype == np.float64
    np.testing.assert_allclose(scale.grad.numpy(), second[1].numpy().reshape(shape))


def test_bool_integer_and_default_tensor_inference_remain_distinct():
    """常量上下文只适用于原生 int/float，布尔条件和独立构造保持原语义。"""
    x = torch.tensor([1., 2.], dtype=np.float32)
    assert torch.tensor(2.).dtype == np.float64
    assert torch.tensor(2).dtype == np.int64
    assert (x + torch.tensor(2.)).dtype == np.float64
    assert (x + np.float32(2.)).dtype == np.float32
    assert F.add(1., 2.).dtype == np.float64
    assert (torch.tensor([1, 2], dtype=np.int32) + 0.5).dtype == np.float64
    result = torch.where(True, x, 0)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result.numpy(), x.numpy())


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
def test_scalar_formulas_preserve_third_derivative(dtype):
    """三次幂的系数和多层规约不能在连续求导时升精度。"""
    data = np.array([0.5, 1.5], dtype=dtype)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    result = ((x ** 3 * 2 - x) * 0.5).mean()
    expected = ((reference ** 3 * 2 - reference) * 0.5).mean()
    for order in range(3):
        result.sum().backward(create_graph=order < 2)
        expected, = pytorch.autograd.grad(expected.sum(), reference, create_graph=order < 2)
        _assert_matches(x.grad, expected)
        result = x.grad
        x.zero_grad()


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
def test_branched_repeated_backward_keeps_accumulated_gradient_dtype(dtype):
    """分支合并和重复反向的累积都应保持叶张量精度及本次数值。"""
    data = np.array([0.5, 1.5], dtype=dtype)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    result = x.mean() + (x * 2).sum() + (x ** 3).mean()
    expected = reference.mean() + (reference * 2).sum() + (reference ** 3).mean()
    for _ in range(2):
        result.backward()
        expected.backward(retain_graph=True)
        _assert_matches(x.grad, reference.grad)


@pytest.mark.parametrize('value,scale', [(1., 1), (1e-4, 1000)])
def test_large_half_mean_avoids_sum_and_normalizer_overflow(value, scale):
    """直接求半精度均值，归一化在恢复半精度之前应用上游缩放。"""
    data = np.full(100000, value, dtype=np.float16)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = x.mean()
        (result * scale).backward()
    expected = reference.mean()
    (expected * scale).backward()
    _assert_matches(result, expected)
    _assert_matches(x.grad, reference.grad)
    np.testing.assert_array_equal(
        x.grad.numpy(), np.full(data.shape, scale / data.size, dtype=np.float16),
    )


@pytest.mark.parametrize('name', ['max', 'min', 'amax', 'amin'])
def test_large_half_extrema_keep_nonzero_tie_gradients(name):
    """大量相同极值不能使半精度计数溢出，最终梯度应按准确数量平分。"""
    x = torch.tensor(np.ones(100000, dtype=np.float16), requires_grad=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        (getattr(x, name)() * 2).backward()
    assert x.grad.dtype == np.float16
    np.testing.assert_array_equal(
        x.grad.numpy(), np.full(x.shape, 2 / x.numel(), dtype=np.float16),
    )


@pytest.mark.parametrize('exponent_dtype', [np.int64, np.float64])
def test_array_exponent_keeps_float32_input_higher_gradients(exponent_dtype):
    """显式数组指数保留前向提升，同时零指数保护及高阶梯度遵守输入精度。"""
    data = np.array([0., 2., 3.], dtype=np.float32)
    exponent = np.array([0, 2, 3], dtype=exponent_dtype)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = x ** exponent
        expected_data = data ** exponent
        assert result.dtype == expected_data.dtype
        np.testing.assert_array_equal(result.numpy(), expected_data)
        result.sum().backward(create_graph=True)
        expected = reference ** pytorch.tensor(exponent)
        first, = pytorch.autograd.grad(expected.sum(), reference, create_graph=True)
        _assert_matches(x.grad, first)
        gradient = x.grad
        x.zero_grad()
        gradient.sum().backward()
    second, = pytorch.autograd.grad(first.sum(), reference)
    # PyTorch 在 0**0 的数组二阶导中产生 NaN；这里按常量函数的解析导数验收。
    assert x.grad.dtype == np.float32
    assert x.grad.numpy()[0] == 0
    _assert_matches(x.grad[1:], second[1:])


def _optimizers(name, parameter, reference):
    """以相同超参数构造优化器，MomentumSGD 对应 PyTorch 的 momentum SGD。"""
    options = {'lr': 0.03}
    reference_name = name
    if name == 'MomentumSGD':
        options['momentum'] = 0.7
        reference_name = 'SGD'
    elif name in ('Adam', 'AdamW'):
        options.update(betas=(0.8, 0.9), eps=1e-6, weight_decay=0.02)
    return (getattr(torch.optim, name)([parameter], **options),
            getattr(pytorch.optim, reference_name)([reference], **options))


def _check_optimizer_state(name, optimizer, parameter, reference_optimizer, reference):
    """检查内部数组和导出 Tensor，防止状态悄然提升或序列化改变精度。"""
    state = optimizer.state_dict()['state']
    if name == 'SGD':
        assert state == {}
        return
    values = state[0]
    expected = reference_optimizer.state[reference]
    if name == 'MomentumSGD':
        assert optimizer.velocity[id(parameter)].dtype == parameter.dtype
        _assert_matches(values['velocity'], -optimizer.lr * expected['momentum_buffer'])
    else:
        assert optimizer.m[id(parameter)].dtype == parameter.dtype
        assert optimizer.v[id(parameter)].dtype == parameter.dtype
        _assert_matches(values['m'], expected['exp_avg'])
        _assert_matches(values['v'], expected['exp_avg_sq'])
        assert values['step'] == expected['step'].item()
    for value in values.values():
        if isinstance(value, torch.Tensor):
            assert value.dtype == parameter.dtype


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(), (2, 3)])
@pytest.mark.parametrize('name', ['SGD', 'MomentumSGD', 'Adam', 'AdamW'])
@pytest.mark.parametrize('path,operation', TRAINING_PATHS,
                         ids=[x[0] for x in TRAINING_PATHS])
def test_multistep_training_and_restored_optimizer_keep_dtype(
        dtype, shape, name, path, operation):
    """逐步对照八次训练，并在中途恢复参数及状态后继续验证精度和数值。"""
    data = np.full(shape, 0.5, dtype=dtype)
    parameter = torch.nn.Parameter(data.copy())
    reference = pytorch.nn.Parameter(pytorch.tensor(data))
    optimizer, reference_optimizer = _optimizers(name, parameter, reference)
    for step in range(8):
        optimizer.zero_grad()
        reference_optimizer.zero_grad(set_to_none=False)
        result = operation(parameter)
        expected = operation(reference)
        _assert_matches(result, expected)
        result.backward()
        expected.backward()
        _assert_matches(parameter.grad, reference.grad)
        optimizer.step()
        reference_optimizer.step()
        _assert_matches(parameter, reference)
        _check_optimizer_state(name, optimizer, parameter, reference_optimizer, reference)
        if step == 3:
            state = optimizer.state_dict()
            reference_state = reference_optimizer.state_dict()
            parameter = torch.nn.Parameter(parameter.numpy().copy())
            reference = pytorch.nn.Parameter(reference.detach().clone())
            optimizer, reference_optimizer = _optimizers(name, parameter, reference)
            optimizer.load_state_dict(state)
            reference_optimizer.load_state_dict(reference_state)
            _check_optimizer_state(name, optimizer, parameter, reference_optimizer, reference)

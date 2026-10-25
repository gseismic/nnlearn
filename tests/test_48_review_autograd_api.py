"""覆盖完整仓库 Review 中的梯度、索引、精度和 Tensor 接口问题。"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
pytorch = pytest.importorskip("torch")

import nnlearn as torch
from nnlearn import nn, optim


@pytest.mark.parametrize('loss_name', ['MSELoss', 'CrossEntropyLoss'])
@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_reused_loss_keeps_each_calls_graph(loss_name, reduction):
    """同一损失实例合并不同形状的批次时，每个输入都应收到各自梯度。"""
    criterion = getattr(nn, loss_name)(reduction=reduction)
    reference_criterion = getattr(pytorch.nn, loss_name)(reduction=reduction)
    inputs = [np.array([[1.0, 2.0]]), np.array([[2.0, -1.0], [0.5, 1.0]])]
    targets = [
        np.zeros_like(data) if loss_name == 'MSELoss'
        else np.arange(len(data), dtype=np.int64)
        for data in inputs
    ]
    xs = [torch.tensor(data, requires_grad=True) for data in inputs]
    refs = [pytorch.tensor(data, requires_grad=True) for data in inputs]
    losses = [criterion(x, torch.tensor(t)) for x, t in zip(xs, targets)]
    expected = [reference_criterion(x, pytorch.tensor(t))
                for x, t in zip(refs, targets)]
    (losses[0].sum() + losses[1].sum()).backward()
    (expected[0].sum() + expected[1].sum()).backward()
    for loss, ref in zip(losses, expected):
        np.testing.assert_allclose(loss.numpy(), ref.detach().numpy())
    for x, ref in zip(xs, refs):
        assert x.grad is not None
        np.testing.assert_allclose(x.grad.numpy(), ref.grad.numpy())


def test_loss_no_grad_call_does_not_overwrite_pending_backward():
    """同一实例的推理调用不能覆盖尚未反传的训练缓存。"""
    for criterion, data, target in [
        (nn.MSELoss(), [[2.0]], [[0.0]]),
        (nn.CrossEntropyLoss(), [[1.0, 2.0]], [0]),
    ]:
        x = torch.tensor(data, requires_grad=True)
        loss = criterion(x, torch.tensor(target))
        with torch.no_grad():
            criterion(torch.tensor(np.array(data) * 3), torch.tensor(target))
        loss.backward()
        independent = torch.tensor(data, requires_grad=True)
        criterion(independent, torch.tensor(target)).backward()
        np.testing.assert_allclose(x.grad.numpy(), independent.grad.numpy())


@pytest.mark.parametrize('create_graph', [False, True])
def test_reduction_gradient_can_be_zeroed(create_graph):
    """广播规约梯度必须可写，默认优化器清零应可重复使用。"""
    p = nn.Parameter(np.ones((2, 3)))
    p.sum().backward(create_graph=create_graph)
    assert p.grad.data.flags.writeable
    optimizer = optim.SGD([p])
    optimizer.zero_grad()
    np.testing.assert_array_equal(p.grad.numpy(), np.zeros((2, 3)))
    p.sum().backward()
    np.testing.assert_array_equal(p.grad.numpy(), np.ones((2, 3)))


@pytest.mark.parametrize('create_graph', [False, True])
def test_input_gradients_have_independent_storage(create_graph):
    """加法的两个输入不能共享梯度；清零一个参数不得影响另一个。"""
    a = nn.Parameter(np.array(1.0))
    b = nn.Parameter(np.array(2.0))
    output = a + b
    output.backward(retain_grad=True, create_graph=create_graph)
    assert a.grad is not b.grad
    assert not np.shares_memory(a.grad.data, b.grad.data)
    optim.SGD([a]).zero_grad()
    assert a.grad.item() == 0
    assert b.grad.item() == 1
    assert output.grad.item() == 1


@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_mse_second_derivatives_include_broadcasted_inputs(reduction, dtype):
    """MSE 的梯度拷贝与广播还原都必须保留对预测和目标的可微依赖。"""
    x = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=dtype, requires_grad=True)
    target = torch.tensor([0.0, 1.0], dtype=dtype, requires_grad=True)
    nn.MSELoss(reduction=reduction)(x, target).sum().backward(create_graph=True)
    factor = 1 / x.numel() if reduction == 'mean' else 1
    assert x.grad.dtype == dtype
    assert target.grad.dtype == dtype
    np.testing.assert_allclose(x.grad.numpy(), 2 * (x.numpy() - target.numpy()) * factor)
    grad = x.grad
    x.zero_grad()
    target.zero_grad()
    grad.sum().backward()
    assert x.grad is not None
    assert target.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.full(x.shape, 2 * factor))
    np.testing.assert_allclose(target.grad.numpy(), np.full(target.shape, -4 * factor))


@pytest.mark.parametrize('op_name', ['gather', 'scatter', 'scatter_add'])
@pytest.mark.parametrize('shape,index_data,dim', [
    ((3, 4), [[1, 1]], 1),
    ((3, 4), [[0, 2], [2, 1]], -1),
    ((3, 4), [[0], [2]], 0),
    ((2, 3, 4), [[[0, 2], [1, 3]]], -1),
    ((3, 4), np.empty((0, 2), dtype=np.int64), 1),
])
def test_index_ops_respect_index_shape(op_name, shape, index_data, dim):
    """非索引维度小于输入时，前向取值和反向梯度均与 PyTorch 一致。"""
    data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 1
    indices = np.asarray(index_data, dtype=np.int64)
    x = torch.tensor(data, requires_grad=True)
    ref = pytorch.tensor(data, requires_grad=True)
    args = [x, dim, torch.tensor(indices)]
    ref_args = [ref, dim, pytorch.tensor(indices)]
    if op_name != 'gather':
        src_data = np.arange(indices.size, dtype=np.float64).reshape(indices.shape) + 2
        src = torch.tensor(src_data, requires_grad=True)
        ref_src = pytorch.tensor(src_data, requires_grad=True)
        args.append(src)
        ref_args.append(ref_src)
    output = getattr(torch, op_name)(*args)
    expected = getattr(pytorch, op_name)(*ref_args)
    assert output.shape == tuple(expected.shape)
    np.testing.assert_allclose(output.numpy(), expected.detach().numpy())
    (output ** 2).sum().backward()
    (expected ** 2).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), ref.grad.numpy())
    if op_name != 'gather':
        assert src.grad.shape == src.shape
        np.testing.assert_allclose(src.grad.numpy(), ref_src.grad.numpy())


def test_module_precision_keeps_discrete_buffers():
    """递归转换浮点精度后，类别索引、布尔 mask 和计数器仍能直接使用。"""
    model = nn.Module()
    model.layer = nn.Embedding(4, 2)
    model.layer.register_buffer('ids', torch.tensor([0, 2], dtype=torch.int64))
    model.register_buffer('mask', torch.tensor([True, False]))
    model.register_buffer('count', torch.tensor(3, dtype='int32'))
    model.register_buffer('running', torch.tensor([1.0, 2.0], dtype=torch.float64))
    model.layer.weight.sum().backward()
    model.to(device='cpu', dtype=torch.float32)
    assert model.layer.ids.dtype == np.int64
    assert model.mask.dtype == np.bool_
    assert model.count.dtype == np.int32
    assert model.running.dtype == np.float32
    assert model.layer.weight.dtype == np.float32
    assert model.layer.weight.grad.dtype == np.float32
    assert model.layer(model.layer.ids).shape == (2, 2)


@pytest.mark.parametrize('optimizer_cls', [optim.Adam, optim.AdamW, optim.MomentumSGD])
def test_optimizer_restore_casts_state_to_parameter_dtype(optimizer_cls):
    """跨精度恢复状态后，动量与后续参数更新都应保持目标 float32 精度。"""
    old = nn.Parameter(np.array([1.0, 2.0], dtype=np.float64))
    optimizer = optimizer_cls([old])
    (old ** 2).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    p = nn.Parameter(old.numpy().astype(np.float32))
    restored = optimizer_cls([p])
    restored.load_state_dict(state)
    for value in restored.state_dict()['state'][0].values():
        if isinstance(value, torch.Tensor):
            assert value.dtype == p.dtype
            assert value.device == p.device
    for _ in range(2):
        # 用规约生成同精度梯度，单独验证恢复的动量不会提升参数类型。
        p.sum().backward()
        restored.step()
        restored.zero_grad()
        assert p.dtype == np.float32
    for value in state['state'][0].values():
        if isinstance(value, torch.Tensor):
            assert value.dtype == np.float64


@pytest.mark.parametrize('factory', [torch.zeros, torch.ones, torch.randn,
                                    torch.Tensor.zeros, torch.Tensor.ones,
                                    torch.Tensor.randn])
@pytest.mark.parametrize('shape', [(2, 3), [2, 3], ()])
def test_creation_accepts_shape_container(factory, shape):
    """工厂与类方法接受元组、列表及空形状，并保留 dtype 与梯度选项。"""
    value = factory(shape, dtype=torch.float32, requires_grad=True)
    assert value.shape == tuple(shape)
    assert value.dtype == np.float32
    assert value.requires_grad
    assert factory(*shape).shape == tuple(shape)


@pytest.mark.parametrize('method', ['sum', 'mean'])
def test_reduction_second_positional_argument_is_keepdim(method):
    """保留维度的位置参数应与关键字及 NumPy 风格别名一致。"""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    output = getattr(x, method)(1, True)
    assert output.shape == (2, 1)
    expected = getattr(x, method)(axis=1, keepdims=True)
    np.testing.assert_allclose(output.numpy(), expected.numpy())
    output.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), np.full((2, 2), 1 if method == 'sum' else 0.5))


@pytest.mark.parametrize('data', [True, [True], [[True]], [False]])
def test_single_element_item_and_bool(data):
    """单元素张量无论维数均可转成 Python 标量及布尔值。"""
    value = torch.tensor(data)
    assert value.item() == np.asarray(data).item()
    assert bool(value) == bool(np.asarray(data).item())


@pytest.mark.parametrize('data', [[], [1, 2]])
def test_item_rejects_non_single_element(data):
    """空张量和多元素张量应给出可理解的异常。"""
    with pytest.raises(ValueError, match='one element'):
        torch.tensor(data).item()


def test_zero_power_has_finite_zero_gradient():
    """零次幂在零点的反向不能先计算负次幂，否则会污染多项式梯度。"""
    x = torch.tensor([0.0, 2.0, -3.0], requires_grad=True)
    with np.errstate(all='raise'):
        (x ** 0).sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros(3))
    x.zero_grad()
    (x ** 2 + x ** 0).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), 2 * x.numpy())


@pytest.mark.parametrize('weak_forward,weak_backward', [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize('op', ['tanh', 'sigmoid', 'softmax', 'log_softmax',
                              'max', 'min', 'amax', 'amin'])
def test_backward_resolves_stored_output_reference(monkeypatch, weak_forward, weak_backward, op):
    """反向按实际存储的引用类型读取输出，包含前后向之间修改配置的场景。"""
    monkeypatch.setitem(torch.runtime_settings, 'remove_recursive_ref', weak_forward)
    data = np.array([[1.0, 3.0], [2.0, -1.0]])
    x = torch.tensor(data, requires_grad=True)
    ref = pytorch.tensor(data, requires_grad=True)
    kwargs = {'dim': -1} if op in ('softmax', 'log_softmax') else {}
    output = getattr(x, op)(**kwargs).sum()
    expected = getattr(ref, op)(**kwargs).sum()
    monkeypatch.setitem(torch.runtime_settings, 'remove_recursive_ref', weak_backward)
    output.backward()
    expected.backward()
    np.testing.assert_allclose(x.grad.numpy(), ref.grad.numpy(), atol=1e-12)


@pytest.mark.parametrize('shape', [(), (3,), (2, 2), (1, 2, 3)])
def test_global_argmax_keepdim(shape):
    """全局 argmax 保留全部原始轴，并保持 int64 无梯度索引。"""
    data = np.arange(max(1, np.prod(shape)), dtype=np.float64).reshape(shape)
    value = torch.tensor(data, requires_grad=True)
    for output in (value.argmax(keepdim=True), torch.argmax(value, keepdim=True),
                   value.argmax(keepdims=True)):
        assert output.shape == (1,) * len(shape)
        assert output.dtype == np.int64
        assert not output.requires_grad
        assert output.item() == data.size - 1


def test_benchmark_models_and_training_use_float32():
    """性能基准两端从建模到一次真实训练更新都使用相同精度。"""
    path = Path(__file__).resolve().parents[1] / 'examples/704_compat_training_performance_benchmark.py'
    spec = importlib.util.spec_from_file_location('training_benchmark', path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    features, labels = benchmark._make_dataset()
    state = benchmark._make_initial_state()
    for framework, dtype in [('nnlearn', np.float32), ('pytorch', pytorch.float32)]:
        model, optimizer, criterion = benchmark._build_training_parts(framework, 'cpu', state)
        batches = benchmark._make_batches(features, labels, 'cpu', framework)
        assert all(p.dtype == dtype for p in model.parameters())
        benchmark._train_steps(model, optimizer, criterion, batches, 1)
        assert all(p.dtype == dtype for p in model.parameters())

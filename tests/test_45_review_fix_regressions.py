"""覆盖 PLAN-039 与 PLAN-040 中真实失败的输入和状态。"""

import numpy as np
import pytest
import torch as pytorch

import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.optim as optim
from torch_1k.function import Function
from torch_1k.functional.pad import pad
from torch_1k.utils.data import default_collate


@pytest.mark.parametrize('x_shape,w_shape', [
    ((2,), (2, 3)),
    ((3, 2), (2,)),
    ((2,), (2,)),
    ((2, 3, 4), (1, 4, 5)),
    ((4,), (2, 4, 3)),
    ((2, 3, 4), (4,)),
    ((1, 3, 4), (2, 4, 5)),
])
def test_matmul_vector_and_broadcast_gradients_match_pytorch(x_shape, w_shape):
    """向量、点积和批次广播均应还原到各自输入的梯度形状。"""
    x_data = np.arange(np.prod(x_shape), dtype=np.float64).reshape(x_shape) / 7 + 1
    w_data = np.arange(np.prod(w_shape), dtype=np.float64).reshape(w_shape) / 11 + 1
    x = torch.tensor(x_data, requires_grad=True)
    w = torch.tensor(w_data, requires_grad=True)
    reference_x = pytorch.tensor(x_data, requires_grad=True)
    reference_w = pytorch.tensor(w_data, requires_grad=True)

    y = x @ w
    reference_y = reference_x @ reference_w
    (y * y).sum().backward()
    (reference_y * reference_y).sum().backward()

    np.testing.assert_allclose(y.numpy(), reference_y.detach().numpy())
    np.testing.assert_allclose(x.grad.numpy(), reference_x.grad.numpy())
    np.testing.assert_allclose(w.grad.numpy(), reference_w.grad.numpy())
    assert x.grad.shape == x_shape
    assert w.grad.shape == w_shape


def test_to_keeps_float_conversion_in_autograd_graph():
    """浮点转换和无操作的 CPU 转换都应把梯度传回原输入。"""
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    converted = (x * 2).float()
    assert converted.cpu() is converted
    converted.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 2.0])
    assert x.grad.dtype == np.float64


@pytest.mark.parametrize('optimizer_type,reference_type', [
    (optim.Adam, pytorch.optim.Adam),
    (optim.AdamW, pytorch.optim.AdamW),
])
def test_adam_uses_parameter_steps_and_restores_them(optimizer_type, reference_type):
    """参数首次获得梯度的时间不同，更新值及状态恢复仍应匹配 PyTorch。"""
    p = nn.Parameter(np.array([1.0]))
    q = nn.Parameter(np.array([1.0]))
    rp = pytorch.nn.Parameter(pytorch.tensor([1.0], dtype=pytorch.float64))
    rq = pytorch.nn.Parameter(pytorch.tensor([1.0], dtype=pytorch.float64))
    optimizer = optimizer_type([p, q], lr=0.1, weight_decay=0.0)
    reference = reference_type([rp, rq], lr=0.1, weight_decay=0.0)

    p.grad = torch.tensor([1.0])
    rp.grad = pytorch.tensor([1.0], dtype=pytorch.float64)
    optimizer.step()
    reference.step()
    p.grad = None
    rp.grad = None
    q.grad = torch.tensor([1.0])
    rq.grad = pytorch.tensor([1.0], dtype=pytorch.float64)
    optimizer.step()
    reference.step()
    np.testing.assert_allclose(q.numpy(), rq.detach().numpy(), atol=1e-7)

    state = optimizer.state_dict()
    assert state['state'][0]['step'] == 1
    assert state['state'][1]['step'] == 1
    restored_p = nn.Parameter(p.numpy().copy())
    restored_q = nn.Parameter(q.numpy().copy())
    restored = optimizer_type([restored_p, restored_q], lr=0.001,
                              weight_decay=0.0)
    restored.load_state_dict(state)
    assert restored.steps[id(restored_q)] == 1

    q.grad = torch.tensor([1.0])
    restored_q.grad = torch.tensor([1.0])
    optimizer.step()
    restored.step()
    np.testing.assert_allclose(restored_q.numpy(), q.numpy())


def test_repeated_advanced_index_accumulates_float32_gradients():
    """同一位置被取出多次时梯度相加，且切片不提升 dtype。"""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = torch.tensor(data, requires_grad=True)
    reference = pytorch.tensor(data, requires_grad=True)
    x[:, [1, 1]].sum().backward()
    reference[:, [1, 1]].sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), reference.grad.numpy())
    assert x.grad.dtype == np.float32


def test_pad_uses_value_and_input_dtype():
    """显式 Pad 的空位使用指定填充值，输出沿用输入 dtype。"""
    x = torch.tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
    y = pad(x, (3,), slice(1, 2), value=9)
    np.testing.assert_array_equal(y.numpy(), [9.0, 3.0, 9.0])
    assert y.dtype == np.float32
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.numpy(), [1.0])


@pytest.mark.parametrize('op,index', [
    ('index_select', torch.tensor([-1])),
    ('index_select', torch.tensor([2])),
    ('gather', torch.tensor([[-1]])),
    ('gather', torch.tensor([[2]])),
])
def test_index_ops_reject_out_of_bounds_indices(op, index):
    """越界索引不能按 NumPy 负索引或 take 的规则回绕。"""
    x = torch.tensor([[1.0, 2.0]])
    with pytest.raises(IndexError):
        getattr(torch, op)(x, 1, index)


@pytest.mark.parametrize('descending', [False, True])
def test_stable_sort_preserves_equal_value_order(descending):
    """升序与降序排序都应保持相同值的原始次序。"""
    data = np.array([3, 1, 3, 1, 2, 2], dtype=np.float64)
    x = torch.tensor(data)
    reference = pytorch.tensor(data)
    result = x.sort(descending=descending, stable=True)
    expected = reference.sort(descending=descending, stable=True)
    np.testing.assert_array_equal(result.values.numpy(), expected.values.numpy())
    np.testing.assert_array_equal(result.indices.numpy(), expected.indices.numpy())
    np.testing.assert_array_equal(
        x.argsort(descending=descending, stable=True).numpy(),
        pytorch.argsort(reference, descending=descending, stable=True).numpy(),
    )


def test_state_dict_rejects_shape_before_mutating_and_preserves_dtype():
    """错误检查点不能部分改写参数，合法输入也不能改变目标 dtype。"""
    model = nn.Linear(2, 2)
    before = model.weight.numpy().copy()
    bad = model.state_dict()
    bad['weight'] = torch.tensor(np.ones((4, 4), dtype=np.int64))
    with pytest.raises(ValueError, match='shape mismatch'):
        model.load_state_dict(bad)
    np.testing.assert_array_equal(model.weight.numpy(), before)

    state = model.state_dict()
    state['weight'] = torch.tensor(np.ones((2, 2), dtype=np.int64))
    model.load_state_dict(state)
    assert model.weight.shape == (2, 2)
    assert model.weight.dtype == before.dtype


def test_layernorm_rejects_broadcasted_trailing_shape():
    """归一化维度不匹配时必须报错，不能广播并改变梯度形状。"""
    with pytest.raises(ValueError, match='normalized_shape'):
        nn.LayerNorm((2, 1))(torch.ones(3, 1, 2, requires_grad=True))


def test_einsum_checks_each_operands_repeated_labels():
    """第一个操作数的对角维度不能误用第二个操作数的形状校验。"""
    a_data = np.eye(2, dtype=np.float64)
    b_data = np.arange(12, dtype=np.float64).reshape(3, 4)
    a = torch.tensor(a_data, requires_grad=True)
    b = torch.tensor(b_data, requires_grad=True)
    reference_a = pytorch.tensor(a_data, requires_grad=True)
    reference_b = pytorch.tensor(b_data, requires_grad=True)
    y = torch.einsum('ii,jk->jk', a, b)
    expected = pytorch.einsum('ii,jk->jk', reference_a, reference_b)
    y.sum().backward()
    expected.sum().backward()
    np.testing.assert_allclose(y.numpy(), expected.detach().numpy())
    np.testing.assert_allclose(a.grad.numpy(), reference_a.grad.numpy())
    np.testing.assert_allclose(b.grad.numpy(), reference_b.grad.numpy())


def test_shared_module_yields_unique_parameters_and_loads_alias_state():
    """共享模块只更新一次参数，状态文件仍可保留两个别名键。"""
    model = nn.Module()
    layer = nn.Linear(2, 2)
    model.a = layer
    model.b = layer
    names = [name for name, _ in model.named_parameters()]
    assert names == ['a.weight', 'a.bias']
    assert len(list(model.parameters())) == 2
    state = model.state_dict()
    assert set(state) == {'a.weight', 'a.bias', 'b.weight', 'b.bias'}
    model.load_state_dict(state)

    layer.weight.grad = torch.ones_like(layer.weight)
    before = layer.weight.numpy().copy()
    optim.SGD(model.parameters(), lr=0.1).step()
    np.testing.assert_allclose(layer.weight.numpy(), before - 0.1)


@pytest.mark.parametrize('samples', [
    [(1, 2), (3, 4, 5)],
    [[1, 2], [3]],
    [{'a': 1}, {'a': 2, 'b': 3}],
])
def test_collate_rejects_inconsistent_sample_structure(samples):
    """样本字段数量或键不同必须报错，避免静默丢数据。"""
    with pytest.raises(ValueError):
        default_collate(samples)


def test_cross_entropy_rejects_float_class_indices():
    """浮点分类标签不能被截断成另一个类别。"""
    with pytest.raises(TypeError, match='int64 or uint8'):
        nn.CrossEntropyLoss()(torch.tensor([[2.0, 1.0]]),
                              torch.tensor([0.9]))


@pytest.mark.parametrize('bad_index,exception', [
    ([1.9], TypeError),
    ([-1], IndexError),
    ([4], IndexError),
])
def test_embedding_rejects_invalid_indices(bad_index, exception):
    """无效词表索引不能被截断或从末尾回绕。"""
    embedding = nn.Embedding(4, 2)
    with pytest.raises(exception):
        embedding(torch.tensor(bad_index))


def test_module_call_passes_keyword_arguments_to_forward():
    """MultiheadAttention 的 need_weights 参数应可经模块调用入口传入。"""
    layer = nn.MultiheadAttention(4, 2)
    output, weights = layer(torch.ones(1, 2, 4), need_weights=True)
    assert output.shape == (1, 2, 4)
    assert weights.shape == (1, 2, 2)


def test_leaf_backward_sets_unit_gradient():
    """叶张量直接调用 backward 时应得到单位梯度。"""
    x = torch.tensor(2.0, requires_grad=True)
    x.backward()
    assert x.grad.item() == 1.0


class TwoOutputs(Function):
    """模拟自定义双输出算子，验证未使用的输出被释放后仍能求导。"""

    def forward(self, x):
        return x * 2, x * 3

    def backward(self, first_grad, second_grad):
        return first_grad * 2 if second_grad is None else (
            first_grad * 2 + second_grad * 3
        )


def test_multi_output_backward_accepts_collected_unused_output():
    """只保留第一个输出时，弱引用失效不应中断反向传播。"""
    x = torch.tensor(2.0, requires_grad=True)
    y = TwoOutputs()(x)[0]
    y.backward()
    assert x.grad.item() == 2.0


def test_retained_intermediate_gradient_is_not_propagated_twice():
    """第二次反向传播只传播本次梯度，叶节点累计两次真实贡献。"""
    x = torch.tensor(2.0, requires_grad=True)
    y = x * x
    y.backward(retain_grad=True)
    (y * y).backward()
    assert x.grad.item() == 36.0


def test_tensor_class_like_methods_keep_dtype():
    """类方法创建的常数张量沿用输入的 float32 dtype。"""
    x = torch.tensor(np.ones(2, dtype=np.float32))
    assert torch.Tensor.zeros_like(x).dtype == np.float32
    assert torch.Tensor.ones_like(x).dtype == np.float32


def test_scalar_random_and_dropout_paths():
    """零维输入也应能生成随机张量并执行 Dropout。"""
    x = torch.tensor(np.array(2.0, dtype=np.float32), requires_grad=True)
    assert torch.rand_like(x).shape == ()
    assert torch.randn_like(x).dtype == np.float32
    assert torch.rand((), dtype=torch.float32).shape == ()
    assert torch.randn((), dtype=torch.float32).shape == ()
    y = nn.functional.dropout(x, p=0.5, training=True)
    assert y.shape == ()
    y.backward()
    assert x.grad is not None

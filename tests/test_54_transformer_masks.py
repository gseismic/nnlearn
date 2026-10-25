"""验证掩码数值、完整反向传播、错误契约及 padding/未来信息隔离。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nnlearn
from nnlearn import nn


def _copy_attention(source, target, torch):
    with torch.no_grad():
        target.in_proj_weight.copy_(torch.tensor(np.concatenate([
            getattr(source, f'{name}_proj').weight.numpy().T for name in 'qkv'
        ])))
        target.in_proj_bias.copy_(torch.tensor(np.concatenate([
            getattr(source, f'{name}_proj').bias.numpy() for name in 'qkv'
        ])))
        target.out_proj.weight.copy_(torch.tensor(source.out_proj.weight.numpy().T))
        target.out_proj.bias.copy_(torch.tensor(source.out_proj.bias.numpy()))


def _assert_close(actual, expected, dtype):
    tolerance = 4e-5 if dtype == np.float32 else 5e-10
    np.testing.assert_allclose(actual.numpy(), expected.detach().cpu().numpy(),
                               rtol=tolerance, atol=tolerance)


def _assert_attention_grads(source, target, dtype):
    for index, name in enumerate('qkv'):
        projection = getattr(source, f'{name}_proj')
        section = slice(index * source.embed_dim, (index + 1) * source.embed_dim)
        _assert_close(projection.weight.grad, target.in_proj_weight.grad[section].T, dtype)
        _assert_close(projection.bias.grad, target.in_proj_bias.grad[section], dtype)
    _assert_close(source.out_proj.weight.grad, target.out_proj.weight.grad.T, dtype)
    _assert_close(source.out_proj.bias.grad, target.out_proj.bias.grad, dtype)


def _mask_arrays(kind, rng, dtype, batch=2, heads=2, length=3, source=5):
    mask = padding = None
    causal = kind.startswith('causal')
    if kind in ('bool2d', 'bool_combined', 'causal_attn'):
        mask = rng.random((length, source)) < 0.35
        mask[:, 0] = False
    if kind == 'bool3d':
        mask = rng.random((batch * heads, length, source)) < 0.35
        mask[:, :, 0] = False
    if kind in ('bool_padding', 'bool_combined', 'causal_padding'):
        padding = np.array([[False, False, False, True, True],
                            [False, False, True, False, True]])
    if kind in ('float2d', 'float_combined', 'causal_float_attn'):
        mask = rng.normal(scale=0.25, size=(length, source)).astype(dtype)
        mask[1:, -1] = -np.inf
    if kind == 'float3d':
        mask = rng.normal(scale=0.25, size=(batch * heads, length, source)).astype(dtype)
        mask[::2, :, -1] = -np.inf
    if kind in ('float_padding', 'float_combined', 'causal_float_padding'):
        padding = rng.normal(scale=0.15, size=(batch, source)).astype(dtype)
        padding[:, -1] = -np.inf
    return mask, padding, causal


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kind', [
    'none', 'bool2d', 'bool3d', 'bool_padding', 'bool_combined',
    'float2d', 'float3d', 'float_padding', 'float_combined',
    'causal', 'causal_padding', 'causal_attn', 'causal_float_attn', 'causal_float_padding',
])
def test_attention_masks_match_pytorch_outputs_and_all_gradients(batch_first, dtype, kind):
    # 交叉注意力使用不同的目标/源长度，逐一对照输入、八个投影参数和浮点掩码梯度。
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(540)
    model = nn.MultiheadAttention(4, 2, batch_first=batch_first).to(dtype=dtype)
    expected = torch.nn.MultiheadAttention(4, 2, dropout=0, batch_first=batch_first,
                                          dtype=getattr(torch, np.dtype(dtype).name))
    _copy_attention(model, expected, torch)
    arrays = [rng.normal(size=(2, length, 4)).astype(dtype) for length in (3, 5, 5)]
    if not batch_first:
        arrays = [array.transpose(1, 0, 2) for array in arrays]
    inputs = [nnlearn.tensor(array.copy(), requires_grad=True) for array in arrays]
    reference_inputs = [torch.tensor(array, requires_grad=True) for array in arrays]
    mask, padding, causal = _mask_arrays(kind, rng, dtype)
    masks = [None if array is None else nnlearn.tensor(array.copy(), requires_grad=array.dtype.kind == 'f')
             for array in (mask, padding)]
    reference_masks = [None if array is None else torch.tensor(array, requires_grad=array.dtype.kind == 'f')
                       for array in (mask, padding)]
    reference_mask = reference_masks[0]
    if causal:
        future = np.arange(5)[None, :] > np.arange(3)[:, None]
        floating = any(array is not None and array.dtype.kind == 'f' for array in (mask, padding))
        causal_mask = torch.tensor(np.where(future, -np.inf, 0.0).astype(dtype) if floating else future)
        if reference_mask is None:
            reference_mask = causal_mask
        elif floating:
            reference_mask = reference_mask + causal_mask
        else:
            reference_mask = reference_mask | causal_mask
    output, weights = model(*inputs, True, attn_mask=masks[0],
                            key_padding_mask=masks[1], is_causal=causal)
    reference_output, reference_weights = expected(
        *reference_inputs, need_weights=True, attn_mask=reference_mask,
        key_padding_mask=reference_masks[1],
    )
    _assert_close(output, reference_output, dtype)
    _assert_close(weights, reference_weights, dtype)
    upstream = rng.normal(size=output.shape).astype(dtype)
    weights_upstream = rng.normal(size=weights.shape).astype(dtype)
    ((output * nnlearn.tensor(upstream)).sum()
     + (weights * nnlearn.tensor(weights_upstream)).sum()).backward()
    ((reference_output * torch.tensor(upstream)).sum()
     + (reference_weights * torch.tensor(weights_upstream)).sum()).backward()
    for actual, reference in zip(inputs, reference_inputs):
        _assert_close(actual.grad, reference.grad, dtype)
    _assert_attention_grads(model, expected, dtype)
    for actual, reference, original in zip(masks, reference_masks, (mask, padding)):
        if actual is not None:
            np.testing.assert_array_equal(actual.numpy(), original)
            if actual.requires_grad:
                _assert_close(actual.grad, reference.grad, dtype)
    for actual, original in zip(inputs, arrays):
        np.testing.assert_array_equal(actual.numpy(), original)


@pytest.mark.parametrize('batch_first', [True, False])
@pytest.mark.parametrize('kind', ['bool', 'float'])
def test_encoder_forwards_masks_and_matches_all_pytorch_gradients(batch_first, kind):
    # 编码层必须转发两种显式掩码，并保持自注意力分支汇总后的输入及全部参数梯度。
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(541)
    model = nn.TransformerEncoderLayer(4, 2, dim_feedforward=7, batch_first=batch_first)
    expected = torch.nn.TransformerEncoderLayer(4, 2, dim_feedforward=7, dropout=0,
                                                batch_first=batch_first, dtype=torch.float64)
    _copy_attention(model.self_attn, expected.self_attn, torch)
    with torch.no_grad():
        for name in ('linear1', 'linear2'):
            source, target = getattr(model, name), getattr(expected, name)
            target.weight.copy_(torch.tensor(source.weight.numpy().T))
            target.bias.copy_(torch.tensor(source.bias.numpy()))
        for name in ('norm1', 'norm2'):
            source, target = getattr(model, name), getattr(expected, name)
            target.weight.copy_(torch.tensor(source.weight.numpy()))
            target.bias.copy_(torch.tensor(source.bias.numpy()))
    array = rng.normal(size=(2, 4, 4))
    if not batch_first:
        array = array.transpose(1, 0, 2)
    causal = np.triu(np.ones((4, 4), dtype=bool), k=1)
    padding = np.array([[False, False, False, True], [False, False, True, True]])
    if kind == 'float':
        causal = np.where(causal, -np.inf, 0.0)
        padding = np.where(padding, -np.inf, 0.0)
    x, tx = nnlearn.tensor(array.copy(), requires_grad=True), torch.tensor(array, requires_grad=True)
    output = model(x, src_mask=nnlearn.tensor(causal),
                   src_key_padding_mask=nnlearn.tensor(padding), is_causal=True)
    reference = expected(tx, src_mask=torch.tensor(causal),
                         src_key_padding_mask=torch.tensor(padding), is_causal=True)
    _assert_close(output, reference, np.float64)
    upstream = rng.normal(size=array.shape)
    (output * nnlearn.tensor(upstream)).sum().backward()
    (reference * torch.tensor(upstream)).sum().backward()
    _assert_close(x.grad, tx.grad, np.float64)
    _assert_attention_grads(model.self_attn, expected.self_attn, np.float64)
    for name in ('linear1', 'linear2', 'norm1', 'norm2'):
        source, target = getattr(model, name), getattr(expected, name)
        expected_weight_grad = target.weight.grad.T if name.startswith('linear') else target.weight.grad
        _assert_close(source.weight.grad, expected_weight_grad, np.float64)
        _assert_close(source.bias.grad, target.bias.grad, np.float64)


@pytest.mark.parametrize('batch_first', [True, False])
def test_default_and_positional_need_weights_contract(batch_first):
    # 旧调用保留单 Tensor 默认返回，以及第 4 个位置参数开启权重返回的约定。
    model = nn.MultiheadAttention(4, 2, batch_first=batch_first)
    array = np.random.default_rng(542).normal(size=(2, 3, 4))
    if not batch_first:
        array = array.transpose(1, 0, 2)
    x = nnlearn.tensor(array)
    plain = model(x)
    explicit = model(x, x, x, False)
    weighted, weights = model(x, None, None, True)
    assert isinstance(plain, nnlearn.Tensor)
    assert plain.shape == array.shape
    assert weights.shape == (2, 3, 3)
    np.testing.assert_array_equal(plain.numpy(), explicit.numpy())
    np.testing.assert_array_equal(plain.numpy(), weighted.numpy())
    np.testing.assert_allclose(weights.numpy().sum(axis=-1), 1.0)


@pytest.mark.parametrize('batch_first', [True, False])
def test_masked_keys_cannot_change_output_or_receive_gradients(batch_first):
    # padding 仅屏蔽 key；被屏蔽 key/value 即使大幅扰动也不影响输出，且梯度精确为零。
    rng = np.random.default_rng(543)
    model = nn.MultiheadAttention(4, 2, batch_first=batch_first)
    q, k, v = [rng.normal(size=(2, length, 4)) for length in (3, 5, 5)]
    padding = np.array([[False, False, True, True, True], [False, False, False, True, True]])
    changed_k, changed_v = k.copy(), v.copy()
    changed_k[padding] += 1000
    changed_v[padding] -= 1000
    arrays = [q, k, v, changed_k, changed_v]
    if not batch_first:
        arrays = [array.transpose(1, 0, 2) for array in arrays]
    tq, tk, tv, ck, cv = [nnlearn.tensor(array, requires_grad=True) for array in arrays]
    mask = nnlearn.tensor(padding)
    output, weights = model(tq, tk, tv, True, key_padding_mask=mask)
    changed = model(tq, ck, cv, key_padding_mask=mask)
    np.testing.assert_array_equal(output.numpy(), changed.numpy())
    output.sum().backward()
    for tensor in (tk, tv):
        grad = tensor.grad.numpy() if batch_first else tensor.grad.numpy().transpose(1, 0, 2)
        np.testing.assert_array_equal(grad[padding], 0)
    np.testing.assert_array_equal(weights.numpy()[np.broadcast_to(padding[:, None, :], (2, 3, 5))], 0)


@pytest.mark.parametrize('batch_first', [True, False])
def test_encoder_causal_and_padding_prevent_output_and_gradient_leakage(batch_first):
    # 扰动未来和 padding token 后，前缀输出保持相同；只监督前缀时后缀输入没有梯度。
    rng = np.random.default_rng(544)
    model = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, batch_first=batch_first)
    array = rng.normal(size=(2, 5, 4))
    changed = array.copy()
    changed[:, 2:, :] += rng.normal(scale=100, size=(2, 3, 4))
    padding = nnlearn.tensor(np.array([[False, False, False, True, True],
                                       [False, False, False, False, True]]))
    if not batch_first:
        array, changed = array.transpose(1, 0, 2), changed.transpose(1, 0, 2)
    x = nnlearn.tensor(array, requires_grad=True)
    first = model(x, src_key_padding_mask=padding, is_causal=True)
    second = model(nnlearn.tensor(changed), src_key_padding_mask=padding, is_causal=True)
    prefix = first[:, :2, :] if batch_first else first[:2, :, :]
    other_prefix = second[:, :2, :] if batch_first else second[:2, :, :]
    np.testing.assert_array_equal(prefix.numpy(), other_prefix.numpy())
    (prefix * nnlearn.tensor(rng.normal(size=prefix.shape))).sum().backward()
    suffix_grad = x.grad.numpy()[:, 2:, :] if batch_first else x.grad.numpy()[2:, :, :]
    np.testing.assert_array_equal(suffix_grad, 0)


@pytest.mark.parametrize('kind', ['bool', 'float', '3d', 'padding', 'combined', 'causal_padding'])
def test_fully_masked_rows_raise_before_softmax(kind):
    # 不允许 softmax 在全 -inf 行中产生 NaN，合并之后形成的空行也必须识别。
    model, x = nn.MultiheadAttention(4, 2), nnlearn.zeros(2, 3, 4)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, :] = True
    kwargs = {'attn_mask': nnlearn.tensor(mask)}
    if kind == 'float':
        kwargs = {'attn_mask': nnlearn.tensor(np.where(mask, -np.inf, 0.0))}
    elif kind == '3d':
        kwargs = {'attn_mask': nnlearn.tensor(np.broadcast_to(mask, (4, 3, 3)).copy())}
    elif kind == 'padding':
        kwargs = {'key_padding_mask': nnlearn.tensor(np.ones((2, 3), dtype=bool))}
    elif kind == 'combined':
        mask[1, 0] = False
        kwargs = {'attn_mask': nnlearn.tensor(mask),
                  'key_padding_mask': nnlearn.tensor([[True, False, False], [False, False, False]])}
    elif kind == 'causal_padding':
        kwargs = {'is_causal': True,
                  'key_padding_mask': nnlearn.tensor([[True, False, False], [False, False, False]])}
    with pytest.raises(ValueError, match='at least one visible key'):
        model(x, **kwargs)


@pytest.mark.parametrize('name', ['attn_mask', 'key_padding_mask'])
@pytest.mark.parametrize('dtype', [np.int64, np.uint8, np.complex128])
def test_masks_reject_non_bool_non_floating_dtype(name, dtype):
    # 整数不能被隐式理解成 True/False，复数也不是合法的注意力偏置。
    shape = (3, 3) if name == 'attn_mask' else (2, 3)
    with pytest.raises(TypeError, match='bool or floating'):
        nn.MultiheadAttention(4, 2)(nnlearn.zeros(2, 3, 4),
                                    **{name: nnlearn.tensor(np.zeros(shape, dtype=dtype))})


@pytest.mark.parametrize('name', ['attn_mask', 'key_padding_mask'])
@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_masks_reject_nan_and_positive_infinity(name, value):
    # 非有限正偏置和 NaN 没有可用的归一化语义，必须在污染训练结果前拒绝。
    shape = (3, 3) if name == 'attn_mask' else (2, 3)
    mask = np.zeros(shape)
    mask.flat[-1] = value
    with pytest.raises(ValueError, match='finite values or -inf'):
        nn.MultiheadAttention(4, 2)(nnlearn.zeros(2, 3, 4), **{name: nnlearn.tensor(mask)})


@pytest.mark.parametrize('name,shape', [
    ('attn_mask', (3,)), ('attn_mask', (2, 3, 3)), ('attn_mask', (2, 2, 3, 3)),
    ('attn_mask', (3, 4)), ('key_padding_mask', (3,)), ('key_padding_mask', (3, 2)),
])
def test_masks_validate_exact_shape(name, shape):
    # 禁止错误形状靠 NumPy 广播变成另一种语义，三维第一轴必须为 batch*heads。
    with pytest.raises(ValueError, match=f'{name} shape'):
        nn.MultiheadAttention(4, 2)(nnlearn.zeros(2, 3, 4),
                                    **{name: nnlearn.tensor(np.zeros(shape, dtype=bool))})


def test_mask_dtype_and_tensor_contracts():
    # 调用方需要显式转换浮点 dtype，不隐式搬设备或把原始数组包装为 Tensor。
    model, x = nn.MultiheadAttention(4, 2), nnlearn.zeros(2, 3, 4, dtype=np.float32)
    with pytest.raises(TypeError, match='floating dtype must match'):
        model(x, attn_mask=nnlearn.zeros(3, 3, dtype=np.float64))
    with pytest.raises(TypeError, match='must be an nnlearn Tensor'):
        model(x, attn_mask=np.zeros((3, 3), dtype=bool))
    with pytest.raises(TypeError, match='must have the same dtype'):
        model(x, attn_mask=nnlearn.tensor(np.zeros((3, 3), dtype=bool)),
              key_padding_mask=nnlearn.zeros(2, 3, dtype=np.float32))
    with pytest.raises(TypeError, match='is_causal must be a bool'):
        model(x, is_causal=1)


@pytest.mark.parametrize('case', ['rank', 'embedding', 'batch', 'source', 'empty', 'dtype', 'integer'])
def test_attention_validates_inputs(case):
    # 检查真正的输入契约，避免跨批次广播或不匹配的 key/value 被默认运算接受。
    model = nn.MultiheadAttention(4, 2)
    q, k, v = [nnlearn.zeros(2, length, 4) for length in (3, 5, 5)]
    expected_error, message = ValueError, ''
    if case == 'rank':
        q, message = nnlearn.zeros(3, 4), '3D batched'
    elif case == 'embedding':
        k, message = nnlearn.zeros(2, 5, 8), 'embed_dim'
    elif case == 'batch':
        k, message = nnlearn.zeros(1, 5, 4), 'same batch'
    elif case == 'source':
        v, message = nnlearn.zeros(2, 4, 4), 'same sequence length'
    elif case == 'empty':
        k, v, message = nnlearn.zeros(2, 0, 4), nnlearn.zeros(2, 0, 4), 'at least one position'
    elif case == 'dtype':
        k, expected_error, message = k.to(dtype=np.float32), TypeError, 'same dtype'
    elif case == 'integer':
        q, expected_error, message = q.long(), TypeError, 'floating dtype'
    with pytest.raises(expected_error, match=message):
        model(q, k, v)


@pytest.mark.parametrize('embed_dim,heads', [(0, 2), (4, 0), (5, 2), (True, 1), (4, 1.5)])
def test_attention_constructor_validates_dimensions(embed_dim, heads):
    # 构造失败应给可定位的 ValueError，而不是依赖 assert 或除零异常。
    with pytest.raises(ValueError):
        nn.MultiheadAttention(embed_dim, heads)


@pytest.mark.skipif(not nnlearn.cuda.is_available(), reason='需要可用 CuPy/CUDA 设备')
def test_mask_devices_must_match_inputs():
    # CUDA 环境中检查显式设备边界，禁止隐式把 CPU 掩码带入 GPU 计算。
    model = nn.MultiheadAttention(4, 2).to('cuda')
    x = nnlearn.zeros(2, 3, 4, device='cuda')
    with pytest.raises(ValueError, match='same device as query'):
        model(x, attn_mask=nnlearn.tensor(np.zeros((3, 3), dtype=bool)))
    with pytest.raises(ValueError, match='same device as query'):
        model(x, key_padding_mask=nnlearn.tensor(np.zeros((2, 3), dtype=bool)))
    with pytest.raises(ValueError, match='same device'):
        model(x, x.cpu(), x)
    output = model(x, key_padding_mask=nnlearn.tensor(np.zeros((2, 3), dtype=bool), device='cuda'),
                   is_causal=True)
    assert output.device == 'cuda'
    assert np.isfinite(output.cpu().numpy()).all()

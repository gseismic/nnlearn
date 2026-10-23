"""不依赖 PyTorch，便于在 NumPy 1.x/2.x 环境复现完整的精度训练契约。"""

import numpy as np
import pytest

import nnlearn as torch


def test_integer_optimizer_keeps_existing_promotion():
    """浮点系数规则不能截断整数参数的学习率，保留既有更新行为。"""
    parameter = torch.nn.Parameter(np.array([1, 2], dtype=np.int64))
    parameter.grad = torch.tensor([1., 1.])
    torch.optim.SGD([parameter], lr=0.1).step()
    assert parameter.dtype == np.float64
    np.testing.assert_allclose(parameter.numpy(), [0.9, 1.9])


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
@pytest.mark.parametrize('shape', [(), (2, 3)])
@pytest.mark.parametrize('name', ['SGD', 'MomentumSGD', 'Adam', 'AdamW'])
def test_backend_multistep_dtype_and_analytic_gradient(dtype, shape, name):
    """组合常见路径并检查解析梯度，八步训练中途恢复状态后继续检查精度。"""
    parameter = torch.nn.Parameter(np.full(shape, 0.5, dtype=dtype))
    options = {'lr': 0.003}
    if name == 'MomentumSGD':
        options['momentum'] = 0.7
    elif name in ('Adam', 'AdamW'):
        # 半精度可表示的 epsilon，避免优化器分母本身下溢干扰精度传递验证。
        options.update(eps=1e-4, betas=(0.8, 0.9), weight_decay=0.01)
    optimizer_class = getattr(torch.optim, name)
    optimizer = optimizer_class([parameter], **options)
    for step in range(8):
        optimizer.zero_grad()
        original = parameter.numpy().astype(np.float64)
        sigmoid = 1 / (1 + np.exp(-original))
        expected_gradient = (1 / parameter.numel() + 2 + 2 * original
                             + sigmoid * (1 - sigmoid)
                             + (original > 0) + ((original >= 0) & (original <= 1)))
        outputs = [parameter.mean(), (parameter * 2).sum(),
                   (parameter ** 2).sum(), parameter.sigmoid().sum(),
                   parameter.relu().sum(), parameter.clamp(min=0., max=1.).sum()]
        assert all(output.dtype == dtype for output in outputs)
        loss = sum(outputs)
        assert loss.dtype == dtype
        loss.backward()
        assert parameter.grad.dtype == dtype
        np.testing.assert_allclose(
            parameter.grad.numpy(), expected_gradient,
            rtol=0.003 if dtype == np.float16 else 1e-6,
        )
        optimizer.step()
        assert parameter.dtype == dtype
        assert np.all(np.isfinite(parameter.numpy()))
        for attribute in ['velocity', 'm', 'v']:
            assert all(value.dtype == dtype
                       for value in getattr(optimizer, attribute, {}).values())
        state_dict = optimizer.state_dict()
        for state in state_dict['state'].values():
            assert all(value.dtype == dtype for value in state.values()
                       if isinstance(value, torch.Tensor))
        if step == 3:
            parameter = torch.nn.Parameter(parameter.numpy().copy())
            optimizer = optimizer_class([parameter], **options)
            optimizer.load_state_dict(state_dict)

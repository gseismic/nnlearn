"""在真实 GPU 上验收设备迁移、训练、优化器恢复及 PyTorch 数值一致性。"""

import os

import numpy as np
import pytest

import nnlearn as torch
from nnlearn import backend, nn, optim
from nnlearn.nn.conv import conv2d


def _cuda_unavailable(reason):
    """普通 CPU 作业显式跳过，GPU 专用作业把环境缺失作为失败。"""
    if os.environ.get('NNLEARN_REQUIRE_CUDA', '').lower() in ('1', 'true', 'yes'):
        pytest.fail(f'NNLEARN_REQUIRE_CUDA 已启用：{reason}', pytrace=False)
    pytest.skip(reason)


@pytest.fixture(scope='module')
def cuda_backend():
    """验证实际分配、编译、矩阵乘法及同步，不能只相信设备探测结果。"""
    if backend.cp is None:
        _cuda_unavailable('未安装可导入的 CuPy，CUDA 实机测试未执行')
    try:
        count = backend.cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        _cuda_unavailable(f'CUDA 设备探测失败：{error}')
    if count < 1:
        _cuda_unavailable('没有可见的 CUDA 设备，CUDA 实机测试未执行')
    # 已发现 GPU 后的内核/动态库错误必须暴露，不能降级为跳过。
    sample = backend.cp.arange(6, dtype=backend.cp.float32).reshape(2, 3)
    np.testing.assert_allclose(backend.cp.asnumpy(sample @ sample.T), [[5, 14], [14, 50]])
    backend.cp.cuda.Stream.null.synchronize()
    return backend.cp


@pytest.fixture(scope='module')
def pytorch_cuda(cuda_backend):
    try:
        import torch as pytorch
    except ImportError:
        _cuda_unavailable('未安装 PyTorch，CUDA 对照未执行')
    if not pytorch.cuda.is_available():
        _cuda_unavailable('PyTorch CUDA 不可用，CUDA 对照未执行')
    # 固定使用独立的原生 CUDA 卷积，避免把可选 cuDNN 作为数值对照的前提。
    with pytorch.backends.cudnn.flags(enabled=False):
        yield pytorch


class _MLP(nn.Module):
    """小型分类网络，覆盖线性层、激活和参数所有权。"""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 6)
        self.output = nn.Linear(6, 2)

    def forward(self, x):
        return self.output(self.hidden(x).tanh())


class _CNN(nn.Module):
    """包含卷积和池化的分类网络，验证空间算子的 CUDA 梯度。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.output = nn.Linear(8, 2)

    def forward(self, x):
        return self.output(self.pool(self.conv(x).relu()).flatten(1))


class _Transformer(nn.Module):
    """重复 token 的编码器分类网络，验证嵌入梯度散射与注意力。"""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.encoder = nn.TransformerEncoderLayer(4, 2, 8, batch_first=True)
        self.output = nn.Linear(4, 2)

    def forward(self, x):
        return self.output(self.encoder(self.embedding(x))[:, 0, :])


MODELS = [('mlp', _MLP), ('cnn', _CNN), ('transformer', _Transformer)]
OPTIMIZERS = [optim.SGD, optim.MomentumSGD, optim.Adam, optim.AdamW]


def _data(kind, dtype=np.float32):
    rng = np.random.default_rng(19)
    if kind == 'mlp':
        x = rng.normal(size=(6, 3)).astype(dtype)
        y = (x[:, 0] > 0).astype(np.int64)
    elif kind == 'cnn':
        x = rng.normal(0, 0.03, size=(6, 1, 4, 4)).astype(dtype)
        x[:3] += 1
        x[3:] -= 1
        y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    else:
        x = np.array([[1, 2, 1], [1, 3, 1], [2, 1, 2], [3, 2, 3],
                      [4, 5, 4], [5, 4, 5]], dtype=np.int64)
        y = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    return x, y


def _assert_tensor(actual, expected, device='cuda', dtype=None,
                   rtol=None, atol=None):
    """值、精度和物理后端同时验证，避免 CPU 回退看起来也通过。"""
    assert actual.device == device
    assert backend.is_cuda_array(actual.data) == (device == 'cuda')
    if hasattr(expected, 'detach'):
        expected = expected.detach().cpu().numpy()
    elif isinstance(expected, torch.Tensor):
        expected = expected.numpy()
    expected = np.asarray(expected)
    assert actual.dtype == (expected.dtype if dtype is None else np.dtype(dtype))
    assert np.isfinite(actual.numpy()).all()
    if rtol is None:
        rtol = 1e-9 if actual.dtype == np.float64 else 8e-4
    if atol is None:
        atol = 1e-10 if actual.dtype == np.float64 else 2e-5
    np.testing.assert_allclose(actual.numpy(), expected, rtol=rtol, atol=atol)


def _assert_gradients(model, device, dtype):
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.device == device
        assert parameter.dtype == np.dtype(dtype)
        assert parameter.grad.dtype == parameter.dtype
        assert backend.is_cuda_array(parameter.data) == (device == 'cuda')
        assert backend.is_cuda_array(parameter.grad.data) == (device == 'cuda')
        assert np.isfinite(parameter.grad.numpy()).all()


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_cuda_copy_preserves_graph_and_source_dtype(cuda_backend, dtype):
    """CPU→CUDA→CPU 的可微复制应把一、二阶梯度送回原设备与精度。"""
    data = np.array([0.5, 1.25, 2.0], dtype=dtype)
    source = torch.tensor(data, requires_grad=True)
    moved = source.to(device='cuda', dtype=np.float64)
    returned = moved.cpu()
    _assert_tensor(moved, data, dtype=np.float64)
    assert moved is not source and returned is not moved
    (returned ** 3).sum().backward(create_graph=True)
    _assert_tensor(source.grad, 3 * data ** 2, device='cpu')
    first = source.grad
    source.zero_grad()
    first.sum().backward()
    _assert_tensor(source.grad, 6 * data, device='cpu')


def test_cuda_module_migration_keeps_parameter_references_and_buffers(cuda_backend):
    """迁移已有梯度、子模块和 buffer 时保持优化器引用、整数与布尔类型。"""
    np.random.seed(17)
    model = _MLP().to(dtype=np.float32)
    model.register_buffer('indices', torch.tensor([1, 2], dtype=np.int64))
    model.register_buffer('mask', torch.tensor([True, False]))
    model.register_buffer('running', torch.tensor([0.5], dtype=np.float32))
    optimizer = optim.Adam(model.parameters())
    parameters = list(model.parameters())
    expected = model.state_dict()
    x = torch.tensor(_data('mlp')[0])
    model(x).sum().backward()
    expected_gradients = [parameter.grad.numpy().copy() for parameter in parameters]
    assert model.to(device='cuda', dtype=np.float64) is model
    assert all(left is right for left, right in zip(parameters, model.parameters()))
    assert all(left is right for left, right in zip(parameters, optimizer.parameters))
    for parameter, gradient in zip(parameters, expected_gradients):
        _assert_tensor(parameter.grad, gradient, dtype=np.float64)
    for name, value in model.state_dict().items():
        target_dtype = np.float64 if expected[name].dtype.kind == 'f' else expected[name].dtype
        _assert_tensor(value, expected[name], dtype=target_dtype)
    assert model.indices.dtype == np.int64
    assert model.mask.dtype == np.bool_
    model.to(device='cpu', dtype=np.float32)
    for name, value in model.state_dict().items():
        _assert_tensor(value, expected[name], device='cpu')
    for parameter, gradient in zip(parameters, expected_gradients):
        _assert_tensor(parameter.grad, gradient, device='cpu')


@pytest.mark.parametrize('kind,model_class', MODELS, ids=[item[0] for item in MODELS])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_models_match_cpu_during_multistep_training(
        cuda_backend, kind, model_class, dtype):
    """三类模型逐步比较输出、损失、输入/参数梯度及更新，且损失实际下降。"""
    np.random.seed(23)
    cpu = model_class().to(dtype=dtype)
    gpu = model_class().to(device='cuda', dtype=dtype)
    gpu.load_state_dict(cpu.state_dict())
    data, targets = _data(kind, dtype)
    floating_input = data.dtype.kind == 'f'
    x_cpu = torch.tensor(data, requires_grad=floating_input)
    x_gpu = torch.tensor(data, device='cuda', requires_grad=floating_input)
    y_cpu = torch.tensor(targets, requires_grad=False)
    y_gpu = y_cpu.cuda()
    # key 投影偏置的数学梯度为零；显式 eps 防止 Adam 放大后端规约舍入。
    cpu_optimizer = optim.Adam(cpu.parameters(), lr=0.03, eps=1e-4)
    gpu_optimizer = optim.Adam(gpu.parameters(), lr=0.03, eps=1e-4)
    losses = []
    for _ in range(8):
        cpu_optimizer.zero_grad(set_to_none=True)
        gpu_optimizer.zero_grad(set_to_none=True)
        x_cpu.zero_grad()
        x_gpu.zero_grad()
        actual = gpu(x_gpu)
        expected = cpu(x_cpu)
        _assert_tensor(actual, expected)
        loss = nn.CrossEntropyLoss()(actual, y_gpu)
        reference_loss = nn.CrossEntropyLoss()(expected, y_cpu)
        _assert_tensor(loss, reference_loss)
        losses.append(loss.item())
        loss.backward()
        reference_loss.backward()
        _assert_gradients(gpu, 'cuda', dtype)
        if floating_input:
            _assert_tensor(x_gpu.grad, x_cpu.grad)
        for actual_parameter, reference_parameter in zip(gpu.parameters(), cpu.parameters()):
            _assert_tensor(actual_parameter.grad, reference_parameter.grad)
        cpu_optimizer.step()
        gpu_optimizer.step()
        for actual_parameter, reference_parameter in zip(gpu.parameters(), cpu.parameters()):
            _assert_tensor(actual_parameter, reference_parameter)
    assert losses[-1] < losses[0] * 0.95


def _pytorch_model_and_parameters(kind, model, pytorch, dtype):
    """按参数布局映射到 PyTorch，线性层转置，注意力合并 Q/K/V 权重。"""
    pnn = pytorch.nn
    pairs = []

    def linear(source, target):
        pairs.append((source.weight, target.weight, True))
        pairs.append((source.bias, target.bias, False))

    if kind == 'mlp':
        reference = pnn.Sequential(pnn.Linear(3, 6), pnn.Tanh(), pnn.Linear(6, 2))
        linear(model.hidden, reference[0])
        linear(model.output, reference[2])
    elif kind == 'cnn':
        reference = pnn.Sequential(pnn.Conv2d(1, 2, 3, padding=1), pnn.ReLU(),
                                   pnn.MaxPool2d(2), pnn.Flatten(1), pnn.Linear(8, 2))
        pairs.extend([(model.conv.weight, reference[0].weight, False),
                      (model.conv.bias, reference[0].bias, False)])
        linear(model.output, reference[4])
    else:
        class ReferenceTransformer(pnn.Module):
            """保持两个实现的前向结构一致，以便逐参数比较 CUDA 梯度。"""

            def __init__(self):
                super().__init__()
                self.embedding = pnn.Embedding(8, 4)
                self.encoder = pnn.TransformerEncoderLayer(4, 2, 8, dropout=0,
                                                           batch_first=True)
                self.output = pnn.Linear(4, 2)

            def forward(self, x):
                return self.output(self.encoder(self.embedding(x))[:, 0, :])

        reference = ReferenceTransformer()
        pairs.append((model.embedding.weight, reference.embedding.weight, False))
        linear(model.output, reference.output)
        linear(model.encoder.linear1, reference.encoder.linear1)
        linear(model.encoder.linear2, reference.encoder.linear2)
        linear(model.encoder.self_attn.out_proj, reference.encoder.self_attn.out_proj)
        for name in ('norm1', 'norm2'):
            source, target = getattr(model.encoder, name), getattr(reference.encoder, name)
            pairs.extend([(source.weight, target.weight, False),
                          (source.bias, target.bias, False)])
    reference.to(device='cuda', dtype=getattr(pytorch, np.dtype(dtype).name))
    with pytorch.no_grad():
        for source, target, transpose in pairs:
            target.copy_(pytorch.tensor(source.numpy().T if transpose else source.numpy(),
                                        device='cuda', dtype=target.dtype))
        if kind == 'transformer':
            attention = reference.encoder.self_attn
            for index, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
                projection = getattr(model.encoder.self_attn, name)
                for source, target, transpose in (
                        (projection.weight, attention.in_proj_weight[index * 4:(index + 1) * 4], True),
                        (projection.bias, attention.in_proj_bias[index * 4:(index + 1) * 4], False)):
                    target.copy_(pytorch.tensor(source.numpy().T if transpose else source.numpy(),
                                                device='cuda', dtype=target.dtype))
    return reference, pairs


@pytest.mark.parametrize('kind,model_class', MODELS, ids=[item[0] for item in MODELS])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_models_match_pytorch_forward_and_gradients(
        pytorch_cuda, kind, model_class, dtype):
    """用 PyTorch CUDA 独立核对前向与全参数梯度，避免两后端共享同一错误。"""
    pytorch = pytorch_cuda
    np.random.seed(29)
    model = model_class().to(device='cuda', dtype=dtype)
    reference, pairs = _pytorch_model_and_parameters(kind, model, pytorch, dtype)
    data, targets = _data(kind, dtype)
    floating_input = data.dtype.kind == 'f'
    x = torch.tensor(data, device='cuda', requires_grad=floating_input)
    ref_x = pytorch.tensor(data, device='cuda', requires_grad=floating_input)
    output, expected = model(x), reference(ref_x)
    _assert_tensor(output, expected)
    loss = nn.CrossEntropyLoss()(output, torch.tensor(targets, device='cuda', requires_grad=False))
    ref_loss = pytorch.nn.functional.cross_entropy(expected, pytorch.tensor(targets, device='cuda'))
    _assert_tensor(loss, ref_loss)
    loss.backward()
    ref_loss.backward()
    if floating_input:
        _assert_tensor(x.grad, ref_x.grad)
    for source, target, transpose in pairs:
        expected_gradient = target.grad.T if transpose else target.grad
        _assert_tensor(source.grad, expected_gradient)
    if kind == 'transformer':
        attention = reference.encoder.self_attn
        for index, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
            projection = getattr(model.encoder.self_attn, name)
            _assert_tensor(projection.weight.grad,
                           attention.in_proj_weight.grad[index * 4:(index + 1) * 4].T)
            _assert_tensor(projection.bias.grad,
                           attention.in_proj_bias.grad[index * 4:(index + 1) * 4])


@pytest.mark.parametrize('mask_kind', ['bool', 'float'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_transformer_masks_match_pytorch_and_block_future_inputs(
        pytorch_cuda, mask_kind, dtype):
    """CUDA 上的 causal/padding 掩码隔离未来输入，并保留浮点掩码梯度。"""
    pytorch = pytorch_cuda
    np.random.seed(41)
    model = _Transformer().to(device='cuda', dtype=dtype)
    reference, pairs = _pytorch_model_and_parameters('transformer', model, pytorch, dtype)
    rng = np.random.default_rng(43)
    data = rng.normal(size=(2, 4, 4)).astype(dtype)
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    padding = np.array([[False, False, False, True], [False, False, True, True]])
    if mask_kind == 'float':
        mask = np.where(mask, -np.inf, 0).astype(dtype)
        padding = np.where(padding, -np.inf, 0).astype(dtype)
    trainable_mask = mask_kind == 'float'
    x = torch.tensor(data, device='cuda', requires_grad=True)
    ref_x = pytorch.tensor(data, device='cuda', requires_grad=True)
    masks = [torch.tensor(value, device='cuda', requires_grad=trainable_mask)
             for value in (mask, padding)]
    ref_masks = [pytorch.tensor(value, device='cuda', requires_grad=trainable_mask)
                 for value in (mask, padding)]
    result = model.encoder(x, src_mask=masks[0], src_key_padding_mask=masks[1], is_causal=True)
    expected = reference.encoder(ref_x, src_mask=ref_masks[0],
                                 src_key_padding_mask=ref_masks[1], is_causal=True)
    _assert_tensor(result, expected)
    changed = data.copy()
    changed[:, 2:, :] += 20
    perturbed = model.encoder(torch.tensor(changed, device='cuda'), src_mask=masks[0],
                              src_key_padding_mask=masks[1], is_causal=True)
    _assert_tensor(result[:, :2], perturbed[:, :2], rtol=1e-6, atol=1e-6)
    upstream = rng.normal(size=(2, 2, 4)).astype(dtype)
    (result[:, :2] * torch.tensor(upstream, device='cuda', requires_grad=False)).sum().backward()
    (expected[:, :2] * pytorch.tensor(upstream, device='cuda')).sum().backward()
    _assert_tensor(x.grad, ref_x.grad)
    np.testing.assert_array_equal(x.grad.numpy()[:, 2:], 0)
    for source, target, transpose in pairs:
        if source.grad is not None:
            _assert_tensor(source.grad, target.grad.T if transpose else target.grad)
    attention = reference.encoder.self_attn
    for index, name in enumerate(('q_proj', 'k_proj', 'v_proj')):
        projection = getattr(model.encoder.self_attn, name)
        _assert_tensor(projection.weight.grad,
                       attention.in_proj_weight.grad[index * 4:(index + 1) * 4].T)
        _assert_tensor(projection.bias.grad,
                       attention.in_proj_bias.grad[index * 4:(index + 1) * 4])
    if trainable_mask:
        for actual_mask, expected_mask in zip(masks, ref_masks):
            _assert_tensor(actual_mask.grad, expected_mask.grad)


def _training_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    optimizer.step()
    return loss


@pytest.mark.parametrize('optimizer_class', OPTIMIZERS, ids=[item.__name__ for item in OPTIMIZERS])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_optimizers_match_cpu_and_pytorch_multistep(
        pytorch_cuda, optimizer_class, dtype):
    """四种优化器逐步核对向量/零维参数与状态的数值、设备及精度。"""
    pytorch = pytorch_cuda
    arrays = [np.array([0.5, -1.0], dtype=dtype), np.array(0.25, dtype=dtype)]
    cpu = [nn.Parameter(value.copy()) for value in arrays]
    gpu = [nn.Parameter(value.copy(), device='cuda') for value in arrays]
    reference = [pytorch.tensor(value, device='cuda', requires_grad=True) for value in arrays]
    options = {'lr': 0.025}
    if optimizer_class is optim.MomentumSGD:
        options['momentum'] = 0.8
        reference_class = pytorch.optim.SGD
    else:
        reference_class = getattr(pytorch.optim, optimizer_class.__name__)
    if optimizer_class in (optim.Adam, optim.AdamW):
        options['weight_decay'] = 0.03
    cpu_optimizer = optimizer_class(cpu, **options)
    gpu_optimizer = optimizer_class(gpu, **options)
    reference_optimizer = reference_class(reference, **options)
    for step in range(5):
        for index, (cpu_parameter, gpu_parameter, reference_parameter) in enumerate(
                zip(cpu, gpu, reference)):
            gradient = np.full(cpu_parameter.shape, (step + 1) * 0.1 + index * 0.05, dtype=dtype)
            cpu_parameter.grad = torch.tensor(gradient, requires_grad=False)
            gpu_parameter.grad = torch.tensor(gradient, device='cuda', requires_grad=False)
            reference_parameter.grad = pytorch.tensor(gradient, device='cuda')
        cpu_optimizer.step()
        gpu_optimizer.step()
        reference_optimizer.step()
        for cpu_parameter, gpu_parameter, reference_parameter in zip(cpu, gpu, reference):
            _assert_tensor(gpu_parameter, cpu_parameter, rtol=2e-5, atol=2e-7)
            _assert_tensor(gpu_parameter, reference_parameter, rtol=2e-5, atol=2e-7)
        cpu_state = cpu_optimizer.state_dict()['state']
        for index, values in gpu_optimizer.state_dict()['state'].items():
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    _assert_tensor(value, cpu_state[index][name], rtol=2e-5, atol=2e-7)
                    assert value.dtype == np.dtype(dtype)


@pytest.mark.parametrize('optimizer_class', OPTIMIZERS, ids=[item.__name__ for item in OPTIMIZERS])
@pytest.mark.parametrize('checkpoint_device', ['cpu', 'cuda'])
def test_cuda_optimizer_checkpoint_resume_matches_uninterrupted(
        cuda_backend, tmp_path, optimizer_class, checkpoint_device):
    """四种优化器从 CPU/CUDA 检查点恢复到 CUDA 后，与不中断训练完全一致。"""
    np.random.seed(31)
    model = _MLP().to(device='cuda', dtype=np.float32)
    optimizer = optimizer_class(model.parameters(), lr=0.025)
    data, targets = _data('mlp')
    x = torch.tensor(data, device='cuda')
    y = torch.tensor(targets, device='cuda', requires_grad=False)
    for _ in range(3):
        _training_step(model, optimizer, x, y)
    checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(checkpoint_device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        return value

    checkpoint = move(checkpoint)
    path = tmp_path / 'cuda-checkpoint.pkl'
    torch.save(checkpoint, path)
    for _ in range(3):
        _training_step(model, optimizer, x, y)
    restored = _MLP().to(device='cuda', dtype=np.float32)
    restored_optimizer = optimizer_class(restored.parameters(), lr=0.001)
    parameter_ids = [id(parameter) for parameter in restored.parameters()]
    checkpoint = torch.load(path)
    restored.load_state_dict(checkpoint['model'])
    restored_optimizer.load_state_dict(checkpoint['optimizer'])
    assert restored_optimizer.lr == optimizer.lr
    assert parameter_ids == [id(parameter) for parameter in restored.parameters()]
    assert parameter_ids == [id(parameter) for parameter in restored_optimizer.parameters]
    for _ in range(3):
        _training_step(restored, restored_optimizer, x, y)
        _assert_gradients(restored, 'cuda', np.float32)
    for name, parameter in restored.named_parameters():
        _assert_tensor(parameter, dict(model.named_parameters())[name], rtol=1e-6, atol=1e-7)
    expected_state = optimizer.state_dict()
    actual_state = restored_optimizer.state_dict()
    assert actual_state['param_groups'] == expected_state['param_groups']
    assert actual_state.get('t') == expected_state.get('t')
    for index, values in actual_state['state'].items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                _assert_tensor(value, expected_state['state'][index][name],
                               rtol=1e-6, atol=1e-7)
            else:
                assert value == expected_state['state'][index][name]
    _assert_tensor(restored(x), model(x), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('operation', ['conv2d', 'layer_norm', 'cross_entropy'])
def test_cuda_second_derivatives_match_pytorch(pytorch_cuda, operation):
    """关键算子在 CUDA 上的 Hessian 向量积应匹配独立 PyTorch 计算图。"""
    pytorch = pytorch_cuda
    rng = np.random.default_rng(37)
    shape = (1, 1, 3, 3) if operation == 'conv2d' else (2, 4)
    data = rng.normal(size=shape).astype(np.float64)
    x = torch.tensor(data, device='cuda', requires_grad=True)
    ref_x = pytorch.tensor(data, device='cuda', requires_grad=True)
    if operation == 'conv2d':
        weights = rng.normal(size=(2, 1, 2, 2)).astype(np.float64)
        result = conv2d(x, torch.tensor(weights, device='cuda'), padding=1)
        expected = pytorch.nn.functional.conv2d(ref_x, pytorch.tensor(weights, device='cuda'), padding=1)
    elif operation == 'layer_norm':
        model = nn.LayerNorm(4).to(device='cuda', dtype=np.float64)
        result = model(x)
        expected = pytorch.nn.functional.layer_norm(ref_x, (4,))
    else:
        target = np.array([1, 3], dtype=np.int64)
        result = nn.CrossEntropyLoss(label_smoothing=0.1)(
            x, torch.tensor(target, device='cuda', requires_grad=False))
        expected = pytorch.nn.functional.cross_entropy(
            ref_x, pytorch.tensor(target, device='cuda'), label_smoothing=0.1)
    _assert_tensor(result, expected)
    (result ** 2).sum().backward(create_graph=True)
    ref_first, = pytorch.autograd.grad((expected ** 2).sum(), ref_x, create_graph=True)
    _assert_tensor(x.grad, ref_first, rtol=1e-8, atol=1e-9)
    vector = rng.normal(size=shape).astype(np.float64)
    first = x.grad
    x.zero_grad()
    (first * torch.tensor(vector, device='cuda', requires_grad=False)).sum().backward()
    ref_second, = pytorch.autograd.grad((ref_first * pytorch.tensor(vector, device='cuda')).sum(), ref_x)
    _assert_tensor(x.grad, ref_second, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('operation', ['max', 'min', 'topk', 'sort'])
@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_selected_indices_scatter_gradients_match_pytorch(
        pytorch_cuda, operation, axis, dtype):
    """选中索引的三维梯度散射应兼容 CuPy，覆盖首轴、中间轴和负轴。"""
    pytorch = pytorch_cuda
    data = np.random.default_rng(47).normal(size=(2, 3, 4)).astype(dtype)
    x = torch.tensor(data, device='cuda', requires_grad=True)
    reference = pytorch.tensor(data, device='cuda', requires_grad=True)
    if operation in ('max', 'min'):
        keepdim = operation == 'max'
        result = getattr(torch, operation)(x, dim=axis, keepdim=keepdim)
        expected = getattr(pytorch, operation)(reference, dim=axis, keepdim=keepdim)
    elif operation == 'topk':
        result, expected = torch.topk(x, 2, dim=axis), pytorch.topk(reference, 2, dim=axis)
    else:
        result = torch.sort(x, dim=axis, descending=True, stable=True)
        expected = pytorch.sort(reference, dim=axis, descending=True, stable=True)
    _assert_tensor(result.values, expected.values)
    _assert_tensor(result.indices, expected.indices)
    upstream = np.linspace(0.25, 1.25, result.values.numel(), dtype=dtype).reshape(result.values.shape)
    (result.values * torch.tensor(upstream, device='cuda', requires_grad=False)).sum().backward()
    (expected.values * pytorch.tensor(upstream, device='cuda')).sum().backward()
    _assert_tensor(x.grad, reference.grad)


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cuda_empty_topk_keeps_empty_output_and_zero_gradient(pytorch_cuda, axis, dtype):
    """空索引的 CUDA 散射仍应产生完整零梯度，不能进入缺失后端 API。"""
    pytorch = pytorch_cuda
    data = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    x = torch.tensor(data, device='cuda', requires_grad=True)
    reference = pytorch.tensor(data, device='cuda', requires_grad=True)
    result = torch.topk(x, 0, dim=axis)
    expected = pytorch.topk(reference, 0, dim=axis)
    _assert_tensor(result.values, expected.values)
    _assert_tensor(result.indices, expected.indices)
    result.values.sum().backward()
    expected.values.sum().backward()
    _assert_tensor(x.grad, reference.grad)
    np.testing.assert_array_equal(x.grad.numpy(), np.zeros_like(data))


@pytest.mark.parametrize('operation', ['softmax', 'layer_norm', 'cross_entropy'])
def test_cuda_half_precision_large_reductions_match_pytorch(pytorch_cuda, operation):
    """半精度大类别和大方差场景在 GPU 上保持有限值、归一化及梯度。"""
    pytorch = pytorch_cuda
    if operation == 'layer_norm':
        data = np.array([[-1000, -500, 500, 1000]], dtype=np.float16)
    else:
        data = np.zeros((2, 70000), dtype=np.float16)
    x = torch.tensor(data, device='cuda', requires_grad=True)
    ref_x = pytorch.tensor(data, device='cuda', requires_grad=True)
    if operation == 'softmax':
        result, expected = x.softmax(dim=-1), ref_x.softmax(dim=-1)
        upstream = np.linspace(-1, 1, data.shape[-1], dtype=np.float32).astype(np.float16)
        loss = (result * torch.tensor(upstream, device='cuda', requires_grad=False)).sum()
        ref_loss = (expected * pytorch.tensor(upstream, device='cuda')).sum()
        np.testing.assert_allclose(result.numpy().sum(axis=-1, dtype=np.float32), 1, atol=0.003)
    elif operation == 'layer_norm':
        result = nn.LayerNorm(4).to(device='cuda', dtype=np.float16)(x)
        expected = pytorch.nn.functional.layer_norm(ref_x, (4,))
        upstream = np.array([[0.5, 1.5, -0.75, 0.25]], dtype=np.float16)
        loss = (result * torch.tensor(upstream, device='cuda', requires_grad=False)).sum()
        ref_loss = (expected * pytorch.tensor(upstream, device='cuda')).sum()
        assert np.count_nonzero(result.numpy()) == data.size
    else:
        targets = np.array([0, 69999], dtype=np.int64)
        result = nn.CrossEntropyLoss()(x, torch.tensor(targets, device='cuda', requires_grad=False))
        expected = pytorch.nn.functional.cross_entropy(ref_x, pytorch.tensor(targets, device='cuda'))
        loss, ref_loss = result, expected
    _assert_tensor(result, expected, rtol=0.02, atol=0.003)
    loss.backward()
    ref_loss.backward()
    _assert_tensor(x.grad, ref_x.grad, rtol=0.03,
                   atol=1e-6 if operation == 'layer_norm' else 1e-7)
    if operation == 'cross_entropy':
        assert np.count_nonzero(x.grad.numpy()) == data.size

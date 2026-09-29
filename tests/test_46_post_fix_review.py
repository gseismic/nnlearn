"""验证提交后复审发现的状态转换与输入边界。"""

import numpy as np
import pytest
import torch as pytorch

import torch_1k as torch
import torch_1k.nn as nn
from torch_1k.nn.normalization import LayerNormFunction


def test_state_load_casts_numeric_buffer_like_pytorch():
    """合法数值状态按目标缓冲区 dtype 转换，并保留原形状。"""
    model = nn.Module()
    model.register_buffer('count', torch.tensor([0], dtype=torch.int64))
    reference = pytorch.nn.Module()
    reference.register_buffer('count', pytorch.tensor([0], dtype=pytorch.int64))

    model.load_state_dict({'count': torch.tensor([2.9])})
    reference.load_state_dict({'count': pytorch.tensor([2.9])})
    assert model.count.dtype == np.int64
    np.testing.assert_array_equal(model.count.numpy(), reference.count.numpy())


def test_state_load_rejects_nonnumeric_without_partial_mutation():
    """状态转换失败前不得修改已验证的其他缓冲区。"""
    model = nn.Module()
    model.register_buffer('first', torch.tensor([0], dtype=torch.int64))
    model.register_buffer('second', torch.tensor([0.0]))
    with pytest.raises(TypeError, match='dtype mismatch'):
        model.load_state_dict({
            'first': torch.tensor([2.9]),
            'second': np.array(['invalid']),
        })
    assert model.first.numpy()[0] == 0
    assert model.second.numpy()[0] == 0.0


def test_cross_entropy_matches_pytorch_integer_target_types():
    """支持 PyTorch 接受的 int64、uint8，并拒绝 int32 类别索引。"""
    logits = torch.tensor([[2.0, 1.0]])
    for dtype in (np.int64, np.uint8):
        target = torch.tensor([0], dtype=dtype)
        assert nn.CrossEntropyLoss()(logits, target).item() == pytest.approx(
            pytorch.nn.functional.cross_entropy(
                pytorch.tensor([[2.0, 1.0]]),
                pytorch.tensor([0], dtype={
                    np.int64: pytorch.int64,
                    np.uint8: pytorch.uint8,
                }[dtype]),
            ).item(),
            abs=1e-6,
        )
    with pytest.raises(TypeError, match='int64 or uint8'):
        nn.CrossEntropyLoss()(logits, torch.tensor([0], dtype='int32'))


def test_layernorm_rejects_empty_normalized_shape_at_construction():
    """空形状应在构造时暴露，包含直接创建函数对象的路径。"""
    with pytest.raises(ValueError, match='normalized_shape'):
        nn.LayerNorm(())
    with pytest.raises(ValueError, match='normalized_shape'):
        LayerNormFunction(())

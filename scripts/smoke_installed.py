"""在安装后的干净环境中检查导入来源、无 PyTorch 训练与状态恢复。"""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def assert_installed(module, source_root):
    """拒绝从源码目录导入，以免掩盖发行包漏文件等问题。"""
    origin = Path(module.__file__).resolve()
    if source_root is not None:
        root = source_root.resolve()
        assert root != origin and root not in origin.parents, origin
    return str(origin)


def check_protocol(source_root):
    """独立协议只依赖标准库，导入不能触发任何训练框架。"""
    import torch_protocol

    origin = assert_installed(torch_protocol, source_root)
    assert torch_protocol.PROTOCOL_VERSION == "1.0"
    assert not importlib.metadata.requires("torch-protocol")
    for name in ("nnlearn", "numpy", "loguru", "torch"):
        assert importlib.util.find_spec(name) is None, name
        assert name not in sys.modules, name
    try:
        torch_protocol.validate_backend(object())
    except torch_protocol.BackendProtocolError as error:
        assert "nn.Linear" in error.missing_members
    else:
        raise AssertionError("不满足协议的后端应当被拒绝")
    return {"package": "torch-protocol", "origin": origin}


def check_nnlearn(source_root):
    """验证真实优化、梯度以及恢复状态后下一步更新的一致性。"""
    assert importlib.util.find_spec("torch") is None, "本烟测必须在无 PyTorch 环境运行"
    import numpy as np
    import nnlearn as backend
    import torch_protocol

    origin = assert_installed(backend, source_root)
    assert_installed(torch_protocol, source_root)
    assert torch_protocol.validate_backend(backend) is backend
    backend.manual_seed(7)
    x = backend.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=backend.float32)
    target = backend.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=backend.float32)
    model = backend.nn.Linear(1, 1).to(dtype=backend.float32)
    optimizer = backend.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = backend.nn.MSELoss()
    initial_loss = loss_fn(model(x), target).item()
    for _ in range(120):
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert np.isfinite(parameter.grad.numpy()).all()
        optimizer.step()
    final_loss = loss_fn(model(x), target).item()
    assert np.isfinite(final_loss) and final_loss < initial_loss * 0.01

    restored = backend.nn.Linear(1, 1).to(dtype=backend.float32)
    restored.load_state_dict(model.state_dict())
    restored_optimizer = backend.optim.Adam(restored.parameters(), lr=0.1)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    assert restored_optimizer.state_dict()["state"]
    for current, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        current_optimizer.zero_grad()
        loss_fn(current(x), target).backward()
        current_optimizer.step()
    np.testing.assert_allclose(restored(x).numpy(), model(x).numpy(), rtol=1e-6)
    assert "torch" not in sys.modules
    return {
        "package": "nnlearn", "origin": origin, "numpy": np.__version__,
        "initial_loss": initial_loss, "final_loss": final_loss, "torch_installed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=("nnlearn", "torch-protocol"), default="nnlearn")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    result = (check_nnlearn(args.source_root) if args.package == "nnlearn"
              else check_protocol(args.source_root))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

import importlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "examples"))

import nnlearn
import example40_torch_protocol as protocol_example
from torch_protocol import (
    BackendProtocolError,
    DataLoaderProtocol,
    DatasetProtocol,
    LossProtocol,
    ModuleProtocol,
    OptimizerProtocol,
    TensorProtocol,
    TorchBackendProtocol,
    validate_backend,
)


def _assert_common_object_contract(backend):
    value = backend.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=backend.float32,
        requires_grad=True,
    )
    assert tuple(value.shape) == (2, 2)
    assert value.ndim == 2
    assert value.dtype == backend.float32
    assert value.requires_grad
    assert value.grad is None
    assert (value + 1).shape == value.shape
    assert (1 + value).shape == value.shape
    assert (value - 1).shape == value.shape
    assert (1 - value).shape == value.shape
    assert (value * 2).shape == value.shape
    assert (2 * value).shape == value.shape
    assert (value / 2).shape == value.shape
    assert (2 / value).shape == value.shape
    assert (value**2).shape == value.shape
    assert value[0].tolist() == [1.0, 2.0]
    assert value.reshape(4).shape == (4,)
    assert value.sum().item() == 10.0
    assert value.mean(dim=0, keepdim=True).shape == (1, 2)
    assert value.clone().tolist() == value.tolist()
    assert value.detach().requires_grad is False
    assert value.to(dtype=backend.float32).shape == value.shape
    assert value.cpu().shape == value.shape
    assert value.detach().cpu().numpy().shape == (2, 2)
    assert value.size(0) == 2
    (value * value).sum().backward()
    assert value.grad is not None

    model = backend.nn.Linear(2, 1)
    assert model.training
    assert model.eval() is model
    assert not model.training
    assert model.train() is model
    assert model.training
    assert model.to(dtype=backend.float32) is model
    model.load_state_dict(model.state_dict())


def test_protocol_package_import_does_not_import_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, torch_protocol; "
                "assert 'nnlearn' not in sys.modules; "
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_nnlearn_satisfies_torch_protocol_v1():
    assert validate_backend(nnlearn) is nnlearn
    assert isinstance(nnlearn, TorchBackendProtocol)
    _assert_common_object_contract(nnlearn)

    value = nnlearn.tensor([1.0, 2.0], requires_grad=True)
    model = nnlearn.nn.Linear(1, 1)
    loss_fn = nnlearn.nn.MSELoss()
    optimizer = nnlearn.optim.SGD(model.parameters(), lr=0.1)
    dataset = nnlearn.utils.data.TensorDataset(
        nnlearn.tensor([[1.0]]), nnlearn.tensor([[2.0]])
    )
    loader = nnlearn.utils.data.DataLoader(dataset, batch_size=1)

    assert isinstance(value, TensorProtocol)
    assert isinstance(model, ModuleProtocol)
    assert isinstance(loss_fn, LossProtocol)
    assert isinstance(optimizer, OptimizerProtocol)
    assert isinstance(dataset, DatasetProtocol)
    assert isinstance(loader, DataLoaderProtocol)


def test_invalid_backend_lists_missing_members():
    with pytest.raises(BackendProtocolError) as error:
        validate_backend(object())
    assert "nn.Linear" in error.value.missing_members
    assert "utils.data.DataLoader" in error.value.missing_members


def test_shared_training_program_runs_with_nnlearn():
    result = protocol_example.run_training(nnlearn, epochs=80)
    assert math.isfinite(result["initial_loss"])
    assert math.isfinite(result["final_loss"])
    assert result["final_loss"] < result["initial_loss"]
    assert result["predictions"].shape == (4, 1)


def test_pytorch_backend_satisfies_protocol_and_runs_shared_program():
    torch = pytest.importorskip("torch")
    importlib.import_module("torch.nn")
    importlib.import_module("torch.optim")
    importlib.import_module("torch.utils.data")

    assert validate_backend(torch) is torch
    assert isinstance(torch, TorchBackendProtocol)
    _assert_common_object_contract(torch)
    assert isinstance(torch.tensor([1.0]), TensorProtocol)
    assert isinstance(torch.nn.Linear(1, 1), ModuleProtocol)
    assert isinstance(torch.nn.MSELoss(), LossProtocol)
    parameter = torch.tensor([1.0], requires_grad=True)
    assert isinstance(
        torch.optim.SGD([parameter], lr=0.1), OptimizerProtocol
    )
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1.0]]), torch.tensor([[2.0]])
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    assert isinstance(dataset, DatasetProtocol)
    assert isinstance(loader, DataLoaderProtocol)

    result = protocol_example.run_training(torch, epochs=80)
    assert math.isfinite(result["initial_loss"])
    assert math.isfinite(result["final_loss"])
    assert result["final_loss"] < result["initial_loss"]
    assert result["predictions"].shape == (4, 1)

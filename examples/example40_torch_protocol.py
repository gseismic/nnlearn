"""同一训练流程通过 Torch Protocol v1 在 nnlearn 或 PyTorch 上运行。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_protocol import TorchBackendProtocol, validate_backend


def run_training(backend: TorchBackendProtocol, epochs: int = 80):
    """只使用 Torch Protocol v1 中声明的后端入口和对象。"""

    backend = validate_backend(backend)
    backend.manual_seed(0)

    features = backend.tensor(
        [[0.0], [1.0], [2.0], [3.0]], dtype=backend.float32
    )
    targets = backend.tensor(
        [[1.0], [3.0], [5.0], [7.0]], dtype=backend.float32
    )
    dataset = backend.utils.data.TensorDataset(features, targets)
    loader = backend.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False
    )

    model = backend.nn.Linear(1, 1)
    loss_fn = backend.nn.MSELoss()
    optimizer = backend.optim.SGD(model.parameters(), lr=0.05)

    model.eval()
    with backend.no_grad():
        initial_loss = loss_fn(model(features), targets).item()

    model.train()
    for _ in range(epochs):
        for batch_features, batch_targets in loader:
            loss = loss_fn(model(batch_features), batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with backend.no_grad():
        predictions = model(features)
        final_loss = loss_fn(predictions, targets).item()

    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "predictions": predictions.detach().cpu().numpy(),
    }


def _load_backend(name: str):
    if name == "nnlearn":
        import nnlearn as backend

        return backend
    if name == "torch":
        try:
            import torch as backend
            import torch.nn
            import torch.optim
            import torch.utils.data
        except ImportError as error:
            raise SystemExit(
                "The torch backend requires the official PyTorch package."
            ) from error
        return backend
    raise SystemExit("TORCH_BACKEND must be either 'nnlearn' or 'torch'.")


def main():
    name = os.environ.get("TORCH_BACKEND", "nnlearn").strip().lower()
    backend = validate_backend(_load_backend(name))
    result = run_training(backend)
    print(f"backend={name}")
    print(f"protocol_version=1.0")
    print(f"initial_loss={result['initial_loss']:.6f}")
    print(f"final_loss={result['final_loss']:.6f}")
    print(f"predictions={result['predictions']}")
    assert result["final_loss"] < result["initial_loss"]


if __name__ == "__main__":
    main()

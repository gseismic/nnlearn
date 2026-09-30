# nnlearn

[English](README.md) | [简体中文](README.zh-CN.md)

## Install

Clone the repository, then install the package and its dependencies:

```bash
git clone https://github.com/pai-studio/nnlearn.git
cd nnlearn
python -m pip install .
```

The examples below use the package from this checkout. They do not require PyTorch.

## Create tensors and run operations

```python
import nnlearn as nl

x = nl.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=nl.float32)
y = nl.tensor([[5.0], [6.0]], dtype=nl.float32)

print((x @ y).tolist())
```

`nnlearn.tensor` creates a tensor. Tensor operations such as matrix multiplication build a result you can use in later calculations.

## Compute gradients

Set `requires_grad=True` on values you want gradients for, calculate a scalar loss, then call `backward()`:

```python
import nnlearn as nl

x = nl.tensor([1.0, 2.0, 3.0], requires_grad=True)
weights = nl.tensor([2.0, -1.0, 0.5])

loss = (x * weights).sum()
loss.backward()

print(x.grad.tolist())
```

The gradient is available from `x.grad` after backpropagation.

## Train a small model

Use `nnlearn.nn` to define a model and `nnlearn.optim` to update its parameters:

```python
import nnlearn as nl
import nnlearn.nn as nn
import nnlearn.optim as optim

features = nl.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=nl.float32)
targets = nl.tensor([[1.0], [3.0], [5.0], [7.0]], dtype=nl.float32)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

model.train()
for _ in range(100):
    predictions = model(features)
    loss = criterion(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

model.eval()
with nl.no_grad():
    predictions = model(features)
print(predictions.tolist())
```

## Load data in batches

Wrap tensors in `TensorDataset` and iterate over them with `DataLoader`. This example uses the `features`, `targets`, `model`, `criterion`, and `optimizer` from the previous section:

```python
from nnlearn.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(features, targets)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model.train()
for _ in range(20):
    for batch_features, batch_targets in loader:
        predictions = model(batch_features)
        loss = criterion(predictions, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Run the bundled tutorial

From the repository root, run:

```bash
python examples/example38_beginner_tutorial.py
```

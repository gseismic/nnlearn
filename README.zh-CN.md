# nnlearn 使用教程

[English](README.md) | [简体中文](README.zh-CN.md)

## 安装

克隆仓库并安装包及其依赖：

```bash
git clone https://github.com/pai-studio/nnlearn.git
cd nnlearn
python -m pip install .
```

下面的示例使用当前检出目录中的包，不需要安装 PyTorch。

## 创建张量并进行运算

```python
import nnlearn as nl

x = nl.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=nl.float32)
y = nl.tensor([[5.0], [6.0]], dtype=nl.float32)

print((x @ y).tolist())
```

`nnlearn.tensor` 用于创建张量。矩阵乘法等张量运算会生成结果，可继续用于后续计算。

## 计算梯度

对需要计算梯度的值设置 `requires_grad=True`，计算标量损失后调用 `backward()`：

```python
import nnlearn as nl

x = nl.tensor([1.0, 2.0, 3.0], requires_grad=True)
weights = nl.tensor([2.0, -1.0, 0.5])

loss = (x * weights).sum()
loss.backward()

print(x.grad.tolist())
```

反向传播后，可以从 `x.grad` 读取梯度。

## 训练一个小模型

使用 `nnlearn.nn` 定义模型，并使用 `nnlearn.optim` 更新模型参数：

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

## 按批次加载数据

将张量放入 `TensorDataset`，再用 `DataLoader` 按批次遍历。示例使用上一节定义的 `features`、`targets`、`model`、`criterion` 和 `optimizer`：

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

## 运行仓库教程

在仓库根目录运行：

```bash
python examples/example38_beginner_tutorial.py
```

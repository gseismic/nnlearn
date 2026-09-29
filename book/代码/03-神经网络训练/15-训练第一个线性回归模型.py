import numpy as np

from mini_nn import Linear, MSELoss, SGD, Tensor


if __name__ == "__main__":
    np.random.seed(14)
    rng = np.random.default_rng(12)
    features = rng.uniform(-1.0, 1.0, size=(100, 1))
    noise = rng.normal(0.0, 0.02, size=(100, 1))
    targets = 3.0 * features - 1.5 + noise

    model = Linear(1, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    inputs = Tensor(features)
    target_tensor = Tensor(targets)
    initial_loss = loss_fn(model(inputs), target_tensor).item()

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), target_tensor)
        loss.backward()
        optimizer.step()

    final_loss = loss_fn(model(inputs), target_tensor).item()
    print(f"初始均方误差: {initial_loss:.6f}")
    print(f"最终均方误差: {final_loss:.6f}")
    print(f"学到的 weight: {model.weight.item():.4f}")
    print(f"学到的 bias: {model.bias.item():.4f}")

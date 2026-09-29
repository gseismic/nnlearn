from mini_tensor import Tensor


if __name__ == "__main__":
    inputs = Tensor([[1.0, 2.0], [2.0, 1.0]], requires_grad=True)
    weight = Tensor([[0.1], [0.2]], requires_grad=True)
    bias = Tensor([0.0], requires_grad=True)
    target = Tensor([[1.0], [1.0]])

    prediction = inputs @ weight + bias
    loss = ((prediction - target) ** 2).mean()
    loss.backward()

    print(f"预测形状: {prediction.shape}")
    print(f"预测值: {prediction.tolist()}")
    print(f"均方误差: {loss.item():.6f}")
    print(f"dL/dinputs: {inputs.grad.tolist()}")
    print(f"dL/dweight: {weight.grad.tolist()}")
    print(f"dL/dbias: {bias.grad.tolist()}")

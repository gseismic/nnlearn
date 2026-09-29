from mini_nn import CrossEntropyLoss, MSELoss, Tensor


if __name__ == "__main__":
    prediction = Tensor([[0.8], [0.2]], requires_grad=True)
    regression_target = Tensor([[1.0], [0.0]])
    mse = MSELoss()(prediction, regression_target)
    mse.backward()
    print(f"均方误差: {mse.item():.6f}")
    print(f"预测值梯度: {prediction.grad.tolist()}")

    logits = Tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        requires_grad=True,
    )
    class_ids = Tensor([0, 1])
    cross_entropy = CrossEntropyLoss()(logits, class_ids)
    cross_entropy.backward()
    print(f"交叉熵: {cross_entropy.item():.6f}")
    print(f"logits 梯度: {logits.grad.tolist()}")

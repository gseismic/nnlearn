from mini_nn import MSELoss, Parameter, SGD, Tensor


if __name__ == "__main__":
    weight = Parameter([1.0])
    target = Tensor([5.0])
    optimizer = SGD([weight], lr=0.1)
    loss_fn = MSELoss()

    optimizer.zero_grad()
    loss = loss_fn(weight, target)
    loss.backward()
    print(f"更新前: weight={weight.item():.1f}, loss={loss.item():.2f}")
    print(f"梯度: {weight.grad.tolist()}")
    optimizer.step()

    new_loss = loss_fn(weight, target)
    print(f"更新后: weight={weight.item():.1f}, loss={new_loss.item():.2f}")

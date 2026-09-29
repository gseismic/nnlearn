from scalar_autograd import Value


def mean_squared_error(weight, bias, inputs, targets):
    total_loss = Value(0.0)
    for x, target in zip(inputs, targets):
        prediction = weight * x + bias
        error = prediction - target
        total_loss = total_loss + error * error
    return total_loss / len(inputs)


if __name__ == "__main__":
    inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    targets = [2.0 * x + 1.0 for x in inputs]
    weight = Value(0.0)
    bias = Value(0.0)
    learning_rate = 0.05

    for step in range(101):
        loss = mean_squared_error(weight, bias, inputs, targets)
        if step in (0, 10, 50, 100):
            print(
                f"step={step:3d} loss={loss.data:.8f} "
                f"weight={weight.data:.6f} bias={bias.data:.6f}"
            )
        if step == 100:
            break

        loss.backward()
        weight.data -= learning_rate * weight.grad
        bias.data -= learning_rate * bias.grad

    print("\n目标函数: y = 2x + 1")

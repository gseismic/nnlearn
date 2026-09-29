from scalar_autograd import Value


if __name__ == "__main__":
    x = Value(2)
    result = x * x + 3 * x
    result.backward()
    print(f"x^2 + 3x = {result.data:g}")
    print(f"d(x^2 + 3x)/dx = {x.grad:g}")

    x = Value(2)
    weight = Value(1)
    bias = Value(0)
    target = Value(5)
    prediction = weight * x + bias
    loss = (prediction - target) ** 2
    loss.backward()

    print(f"\nloss = {loss.data:g}")
    print(f"dL/dw = {weight.grad:g}")
    print(f"dL/db = {bias.grad:g}")

from mini_tensor import Tensor


if __name__ == "__main__":
    left = Tensor([1.0, 2.0], requires_grad=True)
    right = Tensor([3.0, 4.0], requires_grad=True)
    product = left * right
    product.backward()

    print(f"逐元素乘积: {product.tolist()}")
    print(f"dL/dleft: {left.grad.tolist()}")
    print(f"dL/dright: {right.grad.tolist()}")

    matrix = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    weight = Tensor([[2.0, 0.0], [0.0, 3.0]], requires_grad=True)
    output = matrix @ weight
    output.backward()

    print(f"\n矩阵乘法结果: {output.tolist()}")
    print(f"dL/dmatrix: {matrix.grad.tolist()}")
    print(f"dL/dweight: {weight.grad.tolist()}")

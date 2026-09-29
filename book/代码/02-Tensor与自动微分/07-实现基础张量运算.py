from mini_tensor import Tensor


if __name__ == "__main__":
    left = Tensor([[1.0, 2.0], [3.0, 4.0]])
    right = Tensor([[2.0, 0.0], [1.0, 2.0]])
    row_offset = Tensor([10.0, 20.0])

    print(f"逐元素加法: {(left + right).tolist()}")
    print(f"逐元素乘法: {(left * right).tolist()}")
    print(f"矩阵乘法: {(left @ right).tolist()}")
    print(f"广播加法: {(left + row_offset).tolist()}")
    print(f"重塑为向量: {(left @ right).reshape(4).tolist()}")

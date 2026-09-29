import numpy as np

from mini_tensor import Tensor


if __name__ == "__main__":
    scalar = Tensor(3.0)
    vector = Tensor([1.0, 2.0, 3.0])
    matrix = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    float_matrix = Tensor([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    print(
        f"标量: shape={scalar.shape}, ndim={scalar.ndim}, "
        f"item={scalar.item()}"
    )
    print(
        f"向量: shape={vector.shape}, ndim={vector.ndim}, "
        f"dtype={vector.dtype}, values={vector.tolist()}"
    )
    print(
        f"矩阵: shape={matrix.shape}, ndim={matrix.ndim}, "
        f"dtype={matrix.dtype}"
    )
    print(f"显式类型: {float_matrix.dtype}")

    original = np.array([1.0, 2.0, 3.0])
    tensor = Tensor(original)
    original[0] = 99.0
    exported = tensor.numpy()
    exported[1] = 88.0
    print(f"原数组和导出副本修改后，Tensor 仍是: {tensor.tolist()}")

import numpy as np

from mini_tensor import Tensor


if __name__ == "__main__":
    matrix = Tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        requires_grad=True,
    )
    bias = Tensor([0.1, 0.2, 0.3], requires_grad=True)
    broadcast_loss = (matrix + bias).sum()
    broadcast_loss.backward()
    print(f"广播加法损失: {broadcast_loss.item():.1f}")
    print(f"偏置梯度: {bias.grad.tolist()}")

    values = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    row_scale = Tensor([[2.0], [3.0]], requires_grad=True)
    scaled_loss = (values * row_scale).sum()
    scaled_loss.backward()
    print(f"逐行缩放参数梯度: {row_scale.grad.tolist()}")

    mean_input = Tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        requires_grad=True,
    )
    column_means = mean_input.mean(axis=0, keepdims=True)
    column_means.sum().backward()
    print(f"按列均值: {column_means.tolist()}")
    print(f"均值反向梯度: {mean_input.grad.tolist()}")

    shared = Tensor([2.0], requires_grad=True)
    (shared * shared).backward()
    (shared * 3.0).backward()
    print(f"两次反向后的累积梯度: {shared.grad.tolist()}")
    shared.zero_grad()
    print(f"zero_grad 后: {shared.grad}")

    cube = Tensor(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        requires_grad=True,
    )
    reduced = cube.sum(axis=(0, 2), keepdims=True)
    reduced.backward()
    print(f"多轴归约形状: {reduced.shape}")
    print(f"多轴归约结果: {reduced.tolist()}")
    print(f"多轴求和反向梯度形状: {cube.grad.shape}")

    cube_for_mean = Tensor(
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        requires_grad=True,
    )
    reduced_mean = cube_for_mean.mean(axis=(0, 2), keepdims=True)
    reduced_mean.backward()
    print(f"多轴均值反向梯度首项: {cube_for_mean.grad[0, 0, 0]:g}")

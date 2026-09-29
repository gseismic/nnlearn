"""对照 PyTorch 的线性回归单步训练，并核对新加的 sigmoid 算子。"""

from pathlib import Path
import sys

import numpy as np


TENSOR_CODE = Path(__file__).resolve().parents[1] / "02-Tensor与自动微分"
NN_CODE = Path(__file__).resolve().parents[1] / "03-神经网络训练"
for code_path in (str(TENSOR_CODE), str(NN_CODE)):
    if code_path not in sys.path:
        sys.path.insert(0, code_path)

from mini_tensor import Tensor  # noqa: E402
from mini_nn import SGD  # noqa: E402


INPUTS = np.array([[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]])
TARGETS = np.array([[0.5], [0.5], [-2.0]])
INITIAL_WEIGHT = np.array([[0.2], [-0.4]])
INITIAL_BIAS = np.array([0.1])
LEARNING_RATE = 0.05


def main():
    inputs = Tensor(INPUTS)
    targets = Tensor(TARGETS)
    weight = Tensor(INITIAL_WEIGHT, requires_grad=True)
    bias = Tensor(INITIAL_BIAS, requires_grad=True)
    prediction = inputs @ weight + bias
    loss = ((prediction - targets) ** 2).mean()
    loss.backward()

    small = {
        "prediction": prediction.data.copy(),
        "loss": loss.data.copy(),
        "weight_grad": weight.grad.copy(),
        "bias_grad": bias.grad.copy(),
    }
    optimizer = SGD([weight, bias], lr=LEARNING_RATE)
    optimizer.step()
    small["weight_after_step"] = weight.data.copy()
    small["bias_after_step"] = bias.data.copy()

    sigmoid_input = np.array([-1000.0, -3.0, 0.0, 2.0, 1000.0])
    sigmoid_tensor = Tensor(sigmoid_input, requires_grad=True)
    sigmoid_output = sigmoid_tensor.sigmoid()
    sigmoid_output.sum().backward()
    small["sigmoid_output"] = sigmoid_output.data.copy()
    small["sigmoid_grad"] = sigmoid_tensor.grad.copy()

    try:
        import torch
    except ImportError:
        print("当前环境未安装 PyTorch，跳过跨框架数值对照。")
        print(f"本书 Tensor 的线性回归损失: {small['loss'].item():.6f}")
        print(f"sigmoid 输出: {small['sigmoid_output'].tolist()}")
        print(f"sigmoid 梯度: {small['sigmoid_grad'].tolist()}")
        return

    torch_inputs = torch.tensor(INPUTS, dtype=torch.float64)
    torch_targets = torch.tensor(TARGETS, dtype=torch.float64)
    torch_weight = torch.tensor(
        INITIAL_WEIGHT, dtype=torch.float64, requires_grad=True
    )
    torch_bias = torch.tensor(
        INITIAL_BIAS, dtype=torch.float64, requires_grad=True
    )
    torch_prediction = torch_inputs @ torch_weight + torch_bias
    torch_loss = ((torch_prediction - torch_targets) ** 2).mean()
    torch_optimizer = torch.optim.SGD(
        [torch_weight, torch_bias], lr=LEARNING_RATE
    )
    torch_optimizer.zero_grad()
    torch_loss.backward()
    reference = {
        "prediction": torch_prediction.detach().numpy().copy(),
        "loss": torch_loss.detach().numpy().copy(),
        "weight_grad": torch_weight.grad.detach().numpy().copy(),
        "bias_grad": torch_bias.grad.detach().numpy().copy(),
    }
    torch_optimizer.step()
    reference["weight_after_step"] = torch_weight.detach().numpy().copy()
    reference["bias_after_step"] = torch_bias.detach().numpy().copy()

    torch_sigmoid_input = torch.tensor(
        sigmoid_input, dtype=torch.float64, requires_grad=True
    )
    torch_sigmoid_output = torch.sigmoid(torch_sigmoid_input)
    torch_sigmoid_output.sum().backward()
    reference["sigmoid_output"] = torch_sigmoid_output.detach().numpy().copy()
    reference["sigmoid_grad"] = torch_sigmoid_input.grad.detach().numpy().copy()

    print("同一输入、参数和学习率下的结果对照")
    max_error = 0.0
    for name, result in small.items():
        error = float(np.max(np.abs(result - reference[name])))
        max_error = max(max_error, error)
        print(f"{name} 最大绝对误差: {error:.3e}")
    print(f"全部对照项最大绝对误差: {max_error:.3e}")
    if max_error > 1e-10:
        raise SystemExit("对照误差超过 1e-10")
    print("对照通过：前向、梯度和单步 SGD 更新一致。")


if __name__ == "__main__":
    main()

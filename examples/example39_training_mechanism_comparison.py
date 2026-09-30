"""用相同输入和参数比较 nlearn 与 PyTorch 的单步训练过程。"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


INPUTS = np.array(
    [[1.0, 2.0], [-1.0, 3.0], [2.0, -2.0]], dtype=np.float32
)
TARGETS = np.array([[0.5], [0.5], [-2.0]], dtype=np.float32)
INITIAL_WEIGHT = np.array([[0.2], [-0.4]], dtype=np.float32)
INITIAL_BIAS = np.array([0.1], dtype=np.float32)
LEARNING_RATE = 0.05


def _to_numpy(value):
    """把框架张量转成独立 NumPy 数组，供结果比较和显示使用。"""
    return np.asarray(value.detach().cpu().numpy()).copy()


def _run_once(torch_api, optimizer_api):
    """用同一组常量完成线性回归前向、反向和一次 SGD 更新。"""
    inputs = torch_api.tensor(INPUTS, dtype=torch_api.float32)
    targets = torch_api.tensor(TARGETS, dtype=torch_api.float32)
    weight = torch_api.tensor(
        INITIAL_WEIGHT, dtype=torch_api.float32, requires_grad=True
    )
    bias = torch_api.tensor(
        INITIAL_BIAS, dtype=torch_api.float32, requires_grad=True
    )

    prediction = inputs @ weight + bias
    loss = ((prediction - targets) ** 2).mean()

    optimizer = optimizer_api.SGD([weight, bias], lr=LEARNING_RATE)
    optimizer.zero_grad()
    loss.backward()

    result = {
        "prediction": _to_numpy(prediction),
        "loss": _to_numpy(loss),
        "weight_grad": _to_numpy(weight.grad),
        "bias_grad": _to_numpy(bias.grad),
    }

    optimizer.step()
    result["weight_after_step"] = _to_numpy(weight)
    result["bias_after_step"] = _to_numpy(bias)
    return result


def main():
    try:
        import torch
        import torch.optim as torch_optim
    except ImportError as error:
        raise SystemExit(
            "此示例需要 PyTorch 作为对照实现。请先按 "
            "https://pytorch.org/get-started/locally/ 安装 PyTorch。"
        ) from error

    import nlearn
    import nlearn.optim as nlearn_optim

    small = _run_once(nlearn, nlearn_optim)
    reference = _run_once(torch, torch_optim)

    print("相同初始条件下的一步线性回归训练")
    print(f"学习率：{LEARNING_RATE}")
    print()

    max_error = 0.0
    for key in small:
        error = float(np.max(np.abs(small[key] - reference[key])))
        max_error = max(max_error, error)
        print(f"{key}:")
        print(f"  nlearn = {small[key]}")
        print(f"  PyTorch  = {reference[key]}")
        print(f"  最大绝对误差 = {error:.8g}")

    print()
    print(f"所有对照项的最大绝对误差：{max_error:.8g}")
    if max_error > 1e-6:
        raise SystemExit("对照结果超过 1e-6，请检查当前实现和运行环境。")
    print("对照通过：前向、梯度和单步参数更新在容差内一致。")


if __name__ == "__main__":
    main()

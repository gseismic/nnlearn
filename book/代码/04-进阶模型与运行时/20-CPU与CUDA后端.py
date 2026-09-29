"""运行 CPU 数组后端，并在可用时对照可选 CuPy CUDA 路径。"""

from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from mini_device import DeviceArray, cuda_available  # noqa: E402


def main():
    inputs = DeviceArray([[1.0, 2.0], [3.0, 4.0]], device="cpu")
    weights = DeviceArray([[2.0], [-1.0]], device="cpu")
    cpu_result = inputs.matmul(weights)
    print(f"CPU 输入设备: {inputs.device}")
    print(f"CPU 矩阵乘法结果: {cpu_result.numpy().ravel().tolist()}")

    if not cuda_available():
        print("CUDA/CuPy 不可用，跳过可选设备路径。")
        return

    cuda_inputs = inputs.to("cuda")
    cuda_weights = weights.to("cuda")
    cuda_result = cuda_inputs.matmul(cuda_weights)
    returned = cuda_result.to("cpu")
    difference = np.max(np.abs(returned.numpy() - cpu_result.numpy()))
    print(f"CUDA 输出设备: {cuda_result.device}")
    print(f"搬回 CPU 后最大差值: {difference:.3e}")
    if difference > 1e-12:
        raise SystemExit("CPU 与 CUDA 结果超过容差")


if __name__ == "__main__":
    main()

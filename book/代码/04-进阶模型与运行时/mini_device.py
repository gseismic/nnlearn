"""展示 NumPy/CuPy 数组分发和显式设备搬运的最小封装。"""

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None


def cuda_available():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _array_module(device):
    if device == "cpu":
        return np
    if device == "cuda":
        if not cuda_available():
            raise RuntimeError(
                "CUDA 路径不可用；需要安装与 CUDA 环境匹配的 CuPy"
            )
        return cp
    raise ValueError("device 只支持 'cpu' 或 'cuda'")


class DeviceArray:
    """只展示设备存储与矩阵乘法，不包含自动微分。"""

    def __init__(self, data, device="cpu"):
        self.device = device
        xp = _array_module(device)
        self.data = xp.asarray(data)

    @property
    def shape(self):
        return self.data.shape

    def to(self, device):
        if device == self.device:
            xp = _array_module(device)
            return DeviceArray(xp.asarray(self.data).copy(), device)
        if device == "cpu":
            return DeviceArray(cp.asnumpy(self.data), "cpu")
        if device == "cuda":
            xp = _array_module(device)
            return DeviceArray(xp.asarray(self.data), "cuda")
        raise ValueError("device 只支持 'cpu' 或 'cuda'")

    def matmul(self, other):
        if not isinstance(other, DeviceArray):
            raise TypeError("矩阵乘法的两侧都必须是 DeviceArray")
        if self.device != other.device:
            raise ValueError("两个数组位于不同设备，请先显式调用 to() 搬运")
        xp = _array_module(self.device)
        return DeviceArray(xp.matmul(self.data, other.data), self.device)

    def numpy(self):
        if self.device == "cuda":
            return cp.asnumpy(self.data)
        return np.array(self.data, copy=True)

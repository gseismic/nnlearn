from nlearn import backend
from nlearn.function import Function


class CopyTo(Function):
    """在设备或浮点 dtype 转换时保留计算图，使梯度能回到原张量。"""

    def __init__(self, device, dtype):
        super().__init__()
        self.target_device = device
        self.target_dtype = dtype

    def forward(self, x):
        self.source_device = backend.device_of(x)
        self.source_dtype = x.dtype
        return backend.to_device(
            x, self.target_device, dtype=self.target_dtype,
        )

    def backward(self, gy):
        return copy_to(gy, self.source_device, self.source_dtype)


def copy_to(x, device, dtype):
    return CopyTo(device, dtype)(x)

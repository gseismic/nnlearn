import numpy as np


try:
    import cupy as cp
except Exception:
    cp = None


class device:
    def __init__(self, type, index=None):
        if isinstance(type, device):
            self.type = type.type
            self.index = type.index if index is None else index
            return

        value = str(type)
        parsed_index = None
        if ':' in value:
            value, raw_index = value.split(':', 1)
            if raw_index:
                parsed_index = int(raw_index)
        if value not in ('cpu', 'cuda'):
            raise ValueError(f'unsupported device: {type}')

        self.type = value
        self.index = parsed_index if index is None else index

    def __str__(self):
        if self.index is None:
            return self.type
        return f'{self.type}:{self.index}'

    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        try:
            other = device(other)
        except Exception:
            return False
        return self.type == other.type and self.index == other.index


def normalize_device(device):
    if device is None:
        return None
    device = str(device)
    if device.startswith("cuda"):
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"unsupported device: {device}")


def cuda_available():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def is_cuda_array(data):
    return cp is not None and isinstance(data, cp.ndarray)


def get_array_module(*arrays):
    for array in arrays:
        if is_cuda_array(array):
            return cp
    return np


def _put_along_axis(array, indices, values, axis):
    """写回每个切片中互不重复的索引，兼容没有 put_along_axis 的 CuPy。"""
    xp = get_array_module(array, indices)
    if xp is np:
        return np.put_along_axis(array, indices, values, axis=axis)
    if axis < 0:
        axis += array.ndim
    # 每条非目标轴只创建一维坐标，广播后完成 GPU 高级索引写入。
    selector = []
    for dimension, size in enumerate(array.shape):
        if dimension == axis:
            selector.append(indices)
        else:
            shape = [1] * array.ndim
            shape[dimension] = size
            selector.append(xp.arange(size, dtype=xp.int64).reshape(shape))
    array[tuple(selector)] = values


def device_of(data):
    return "cuda" if is_cuda_array(data) else "cpu"


def as_numpy(data):
    if is_cuda_array(data):
        return cp.asnumpy(data)
    return np.asarray(data)


def ensure_array(data, device=None, dtype=None):
    target_device = normalize_device(device)
    if is_cuda_array(data) or isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data, dtype=dtype)

    if target_device is None:
        if dtype is not None and getattr(array, "dtype", None) != dtype:
            array = array.astype(dtype)
        return array
    return to_device(array, target_device, dtype=dtype)


def to_device(data, device, dtype=None):
    target_device = normalize_device(device)
    if target_device is None:
        return ensure_array(data, dtype=dtype)

    if target_device == "cuda":
        if not cuda_available():
            raise RuntimeError("cuda is not available: install CuPy with CUDA support")
        return cp.asarray(data, dtype=dtype)

    array = as_numpy(data)
    if dtype is not None:
        array = array.astype(dtype)
    return array


def array_module_for_device(device):
    target_device = normalize_device(device)
    if target_device == "cuda":
        if not cuda_available():
            raise RuntimeError("cuda is not available: install CuPy with CUDA support")
        return cp
    return np


def seed(seed_value):
    np.random.seed(seed_value)
    if cuda_available():
        cp.random.seed(seed_value)

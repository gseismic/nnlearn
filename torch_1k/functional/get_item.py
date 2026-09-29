from torch_1k.function import Function
from torch_1k import backend


class GetItemGrad(Function):
    """将索引后的梯度散射回输入，并累加重复索引的贡献。"""

    def __init__(self, shape, index_or_slices):
        super().__init__()
        self.shape = shape
        self.index_or_slices = index_or_slices

    def forward(self, gy):
        xp = backend.get_array_module(gy)
        gx = xp.zeros(self.shape, dtype=gy.dtype)
        xp.add.at(gx, self.index_or_slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.index_or_slices)


#GetItem
class GetItem(Function):
    def __init__(self, index_or_slices):
        self.index_or_slices = index_or_slices
        self.x_shape = None

    def forward(self, x):
        # XXX 这样支持多次forward的吗？
        self.x_shape = x.shape
        # 不能forward(self, x, index_or_slices) 设计?
        # get = GetItem()
        # y = get(x, 1), y = get(y, 2)
        # 导致get不知道第一次的index_or_slices(被覆盖了)
        y = x[self.index_or_slices]
        self.y_shape = y.shape
        return y

    def backward(self, gy):
        return GetItemGrad(self.x_shape, self.index_or_slices)(gy)

def get_item(x, index_or_slices):
    return GetItem(index_or_slices)(x)

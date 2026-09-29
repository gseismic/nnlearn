class Value:
    """只包装一个 Python 浮点数，暂时不记录计算图。"""

    def __init__(self, data):
        self.data = float(data)

    @staticmethod
    def _as_value(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._as_value(other)
        return Value(self.data + other.data)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        return Value(self.data * other.data)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return Value(-self.data)

    def __sub__(self, other):
        return self + (-self._as_value(other))

    def __rsub__(self, other):
        return self._as_value(other) - self

    def __repr__(self):
        return f"Value(data={self.data})"


if __name__ == "__main__":
    x = Value(2)
    target = Value(5)
    weight = Value(1)
    bias = Value(0)

    prediction = weight * x + bias
    error = prediction - target
    loss = error * error

    print(f"预测值: {prediction.data:g}")
    print(f"误差: {error.data:g}")
    print(f"平方损失: {loss.data:g}")

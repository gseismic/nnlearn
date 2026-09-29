"""本书标量自动微分篇的最终教学实现。"""


class Value:
    def __init__(self, data, parents=(), operation=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.operation = operation
        self._backward = lambda: None

    @staticmethod
    def _as_value(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._as_value(other)
        result = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = backward
        return result

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        result = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = backward
        return result

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._as_value(other))

    def __rsub__(self, other):
        return self._as_value(other) - self

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("标量 Value 的指数必须是 Python 数值")
        result = Value(self.data**exponent, (self,), f"**{exponent:g}")

        def backward():
            self.grad += exponent * self.data ** (exponent - 1) * result.grad

        result._backward = backward
        return result

    def __truediv__(self, other):
        other = self._as_value(other)
        return self * (other**-1)

    def __rtruediv__(self, other):
        return self._as_value(other) / self

    def backward(self):
        topological_order = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            topological_order.append(node)

        visit(self)

        for node in topological_order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topological_order):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data:g}, grad={self.grad:g})"

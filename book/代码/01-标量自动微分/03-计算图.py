class Value:
    """记录标量数值、产生它的父节点和局部导数。"""

    def __init__(self, data, parents=(), operation=""):
        self.data = float(data)
        self.parents = tuple(parents)
        self.operation = operation
        self.local_grads = ()

    @staticmethod
    def _as_value(other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._as_value(other)
        result = Value(self.data + other.data, (self, other), "+")
        result.local_grads = (1.0, 1.0)
        return result

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        result = Value(self.data * other.data, (self, other), "*")
        result.local_grads = (other.data, self.data)
        return result

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._as_value(other))

    def __rsub__(self, other):
        return self._as_value(other) - self


def print_graph(root):
    visited = set()

    def visit(node, depth):
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            visit(parent, depth + 1)

        if node.parents:
            edges = ", ".join(
                f"{parent.data:g} (局部导数 {local_grad:g})"
                for parent, local_grad in zip(node.parents, node.local_grads)
            )
            description = f"{node.operation} 的输入: {edges}"
        else:
            description = "叶子节点"
        print(f"{'  ' * depth}{node.data:g}: {description}")

    visit(root, 0)


if __name__ == "__main__":
    x = Value(2)
    weight = Value(1)
    bias = Value(0)
    target = Value(5)

    prediction = weight * x + bias
    error = prediction - target
    loss = error * error

    print(f"loss = {loss.data:g}\n计算图（从输入到结果）：")
    print_graph(loss)

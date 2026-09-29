"""最小随机梯度下降优化器。"""


class SGD:
    def __init__(self, parameters, lr=0.01):
        if lr <= 0:
            raise ValueError("学习率必须大于 0")
        unique = {}
        for parameter in parameters:
            unique[id(parameter)] = parameter
        self.parameters = tuple(unique.values())
        self.lr = float(lr)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.data = parameter.data - self.lr * parameter.grad

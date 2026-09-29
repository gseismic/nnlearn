import numpy as np

from mini_nn import Linear, Module, Parameter, Tensor


class TwoLayerModel(Module):
    def __init__(self):
        super().__init__()
        self.hidden = Linear(2, 3)
        self.output = Linear(3, 1)

    def forward(self, inputs):
        return self.output(self.hidden(inputs))


if __name__ == "__main__":
    np.random.seed(4)
    model = TwoLayerModel()
    standalone = Parameter([0.5, -0.5])

    print("模型参数：")
    for name, parameter in model.named_parameters():
        print(f"  {name}: shape={parameter.shape}, requires_grad={parameter.requires_grad}")
    print(f"参数总数: {sum(parameter.size for parameter in model.parameters())}")
    print(f"独立 Parameter: {standalone.tolist()}")
    print(f"状态键: {list(model.state_dict())}")

    prediction = model(Tensor([[1.0, 2.0]]))
    print(f"一次前向输出形状: {prediction.shape}")

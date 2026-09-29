import numpy as np

from mini_nn import Linear, ReLU, Sigmoid, Tanh, Tensor


if __name__ == "__main__":
    np.random.seed(0)
    inputs = Tensor([[1.0, -2.0], [0.5, 1.0]], requires_grad=True)
    layer = Linear(2, 3)
    logits = layer(inputs)
    activated = ReLU()(logits)
    activated.sum().backward()

    print(f"输入形状: {inputs.shape}")
    print(f"权重形状: {layer.weight.shape}")
    print(f"线性输出形状: {logits.shape}")
    print(f"ReLU 输出: {activated.tolist()}")
    print(f"权重梯度形状: {layer.weight.grad.shape}")

    sample = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
    tanh_output = Tanh()(sample)
    sigmoid_output = Sigmoid()(sample)
    (tanh_output.sum() + sigmoid_output.sum()).backward()
    print(f"Tanh 输出: {tanh_output.tolist()}")
    print(f"Sigmoid 输出: {sigmoid_output.tolist()}")
    print(f"激活输入梯度: {sample.grad.tolist()}")

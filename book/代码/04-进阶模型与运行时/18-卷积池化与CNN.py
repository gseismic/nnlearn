"""在带有不同位置亮块的合成图像上训练一个小型 CNN。"""

from pathlib import Path
import sys

import numpy as np


PART3_CODE = Path(__file__).resolve().parents[1] / "03-神经网络训练"
if str(PART3_CODE) not in sys.path:
    sys.path.insert(0, str(PART3_CODE))

from mini_nn import (  # noqa: E402
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    SGD,
    Sequential,
    Tensor,
)


def make_dataset(seed=18, samples_per_class=12):
    rng = np.random.default_rng(seed)
    positions = [(1, 1), (1, 8), (8, 4)]
    images = []
    labels = []
    for label, (row, col) in enumerate(positions):
        for _ in range(samples_per_class):
            image = rng.normal(0.0, 0.04, size=(12, 12))
            image[row:row + 3, col:col + 3] += 1.0
            images.append(image)
            labels.append(label)
    return (
        Tensor(np.asarray(images)[:, None, :, :]),
        Tensor(np.asarray(labels, dtype=np.int64)),
    )


def main():
    np.random.seed(18)
    images, labels = make_dataset()
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(kernel_size=2),
        Flatten(),
        Linear(4 * 6 * 6, 3),
    )
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.08)

    with_logits = model(images)
    initial_loss = loss_fn(with_logits, labels).item()
    for _ in range(70):
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    logits = model(images)
    final_loss = loss_fn(logits, labels).item()
    accuracy = np.mean(np.argmax(logits.data, axis=1) == labels.data)
    print(f"输入形状: {images.shape}")
    print(f"卷积输出形状: {model.layer_0(images).shape}")
    print(f"初始交叉熵: {initial_loss:.6f}")
    print(f"最终交叉熵: {final_loss:.6f}")
    print(f"合成图像准确率: {accuracy:.3f}")

    toy_input = Tensor(
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]]), requires_grad=True
    )
    toy_conv = Conv2d(1, 1, kernel_size=2, bias=False)
    toy_conv.weight.data = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    toy_output = toy_conv(toy_input)
    toy_output.sum().backward()
    print(f"单窗口卷积输出: {toy_output.item():.1f}")
    print(f"单窗口输入梯度: {toy_input.grad.tolist()}")
    print(f"单窗口卷积核梯度: {toy_conv.weight.grad.tolist()}")

    pool_input = Tensor(
        np.array([[[[1.0, 4.0], [3.0, 2.0]]]]), requires_grad=True
    )
    pool_output = MaxPool2d(2)(pool_input)
    pool_output.sum().backward()
    print(f"最大池化输出: {pool_output.item():.1f}")
    print(f"最大池化输入梯度: {pool_input.grad.tolist()}")


if __name__ == "__main__":
    main()

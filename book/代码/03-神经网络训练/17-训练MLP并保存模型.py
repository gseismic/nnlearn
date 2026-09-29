import tempfile

import numpy as np

from mini_nn import (
    ArrayDataset,
    CrossEntropyLoss,
    DataLoader,
    Linear,
    Module,
    ReLU,
    SGD,
    Sequential,
    Tensor,
    load_state_dict,
    save_state_dict,
)


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Linear(2, 12),
            ReLU(),
            Linear(12, 3),
        )

    def forward(self, inputs):
        return self.network(inputs)


def make_blobs(samples_per_class, rng):
    centers = np.array([[-1.0, -0.8], [1.0, -0.8], [0.0, 1.0]])
    features = []
    labels = []
    for class_id, center in enumerate(centers):
        features.append(
            rng.normal(center, 0.35, size=(samples_per_class, 2))
        )
        labels.append(
            np.full(samples_per_class, class_id, dtype=np.int64)
        )
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    order = rng.permutation(len(features))
    return features[order], labels[order]


if __name__ == "__main__":
    np.random.seed(5)
    data_rng = np.random.default_rng(21)
    train_x, train_y = make_blobs(60, data_rng)
    test_x, test_y = make_blobs(30, data_rng)
    train_loader = DataLoader(
        ArrayDataset(train_x, train_y),
        batch_size=24,
        shuffle=True,
        seed=9,
    )

    model = MLP()
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.08)

    for epoch in range(30):
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.shape[0]
            seen += batch_x.shape[0]

        if epoch in (0, 9, 29):
            print(f"epoch={epoch:2d} loss={total_loss / seen:.6f}")

    model.eval()
    test_logits = model(Tensor(test_x))
    accuracy = np.mean(np.argmax(test_logits.data, axis=1) == test_y)
    print(f"测试准确率: {accuracy:.3f}")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = f"{directory}/mlp_state.npz"
        save_state_dict(model, checkpoint)
        restored = MLP()
        load_state_dict(restored, checkpoint)
        restored_logits = restored(Tensor(test_x))
        difference = np.max(np.abs(test_logits.data - restored_logits.data))
        print(f"载入后 logits 最大差值: {difference:.1e}")

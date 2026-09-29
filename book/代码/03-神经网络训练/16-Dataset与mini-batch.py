import numpy as np

from mini_nn import ArrayDataset, DataLoader


if __name__ == "__main__":
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    targets = np.arange(10, dtype=np.int64) % 2
    dataset = ArrayDataset(features, targets)

    loader = DataLoader(
        dataset, batch_size=4, shuffle=True, drop_last=False, seed=7
    )
    print(f"样本数: {len(dataset)}")
    print(f"批次数（保留末批）: {len(loader)}")
    for batch_index, (batch_features, batch_targets) in enumerate(loader):
        print(
            f"batch {batch_index}: x={batch_features.shape}, "
            f"y={batch_targets.shape}, labels={batch_targets.tolist()}"
        )

    drop_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, drop_last=True
    )
    print(f"批次数（丢弃末批）: {len(drop_loader)}")

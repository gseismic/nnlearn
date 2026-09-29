"""内存数组数据集与 mini-batch 加载器。"""

import numpy as np

from mini_tensor import Tensor


class ArrayDataset:
    def __init__(self, features, targets):
        self.features = np.array(features, copy=True)
        self.targets = np.array(targets, copy=True)
        if self.features.ndim < 2:
            raise ValueError("features 至少需要样本轴和特征轴")
        if self.targets.ndim == 0:
            raise ValueError("targets 必须包含样本轴")
        if len(self.features) != len(self.targets):
            raise ValueError("features 和 targets 的样本数不一致")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 drop_last=False, seed=None):
        if (isinstance(batch_size, (bool, np.bool_))
                or not isinstance(batch_size, (int, np.integer))
                or batch_size <= 0):
            raise ValueError("batch_size 必须是正整数")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        count = len(self.dataset)
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            samples = [self.dataset[index] for index in batch_indices]
            features, targets = zip(*samples)
            yield (
                Tensor(np.stack(features, axis=0)),
                Tensor(np.stack(targets, axis=0)),
            )

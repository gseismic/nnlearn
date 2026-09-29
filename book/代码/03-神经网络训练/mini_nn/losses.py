"""训练目标与分类损失。"""

import numpy as np

from mini_tensor import Function

from .core import Module


class MSELoss(Module):
    def forward(self, prediction, target):
        error = prediction - target
        return (error * error).mean()


class _CrossEntropy(Function):
    def forward(self, logits, targets):
        if logits.ndim != 2:
            raise ValueError("交叉熵输入必须是 (batch, classes) 二维数组")
        if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
            raise ValueError("类别标签形状必须是 (batch,)")
        if not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("交叉熵标签必须是整数类别编号")
        if logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("交叉熵不接受空批次或零个类别")

        labels = targets.astype(np.int64, copy=False)
        if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("类别编号超出 [0, 类别数) 范围")

        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        log_partition = np.log(
            np.sum(exp_logits, axis=1, keepdims=True)
        )
        log_probabilities = shifted - log_partition
        rows = np.arange(logits.shape[0])

        self.probabilities = exp_logits / np.sum(
            exp_logits, axis=1, keepdims=True
        )
        self.labels = labels
        self.batch_size = logits.shape[0]
        return -np.mean(log_probabilities[rows, labels])

    def backward(self, output_grad):
        gradient = self.probabilities.copy()
        gradient[np.arange(self.batch_size), self.labels] -= 1.0
        gradient /= self.batch_size
        return gradient * output_grad, None


class CrossEntropyLoss(Module):
    def forward(self, logits, targets):
        return _CrossEntropy()(logits, targets)

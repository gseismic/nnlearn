"""变长传感器序列训练：仅根据当前前缀预测累计均值，不下载数据。

USE_NNLEARN=1 使用 nnlearn，USE_NNLEARN=0 使用 PyTorch。
padding 掩码隔离批次补齐值，causal 掩码隔离未来，损失仅统计有效位置。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_NNLEARN = os.getenv('USE_NNLEARN', '1') != '0'

if USE_NNLEARN:
    import nnlearn as torch
    import nnlearn.nn as nn
    import nnlearn.optim as optim
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim


class PrefixMeanRegressor(nn.Module):
    """学习传感器观测的累计均值，用前缀监督检验序列注意力可真实训练。"""

    def __init__(self, d_model=8):
        super().__init__()
        # 把标量观测和采样时刻映射到注意力使用的特征空间。
        self.input_projection = nn.Linear(2, d_model)
        encoder_options = dict(d_model=d_model, nhead=2, dim_feedforward=16, batch_first=True)
        if not USE_NNLEARN:
            encoder_options['dropout'] = 0.0
        # 同时隔离未来观测和变长序列的补齐值。
        self.encoder = nn.TransformerEncoderLayer(**encoder_options)
        # 每个有效时刻独立输出一个累计均值预测。
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, observations, causal_mask, padding_mask):
        encoded = self.encoder(self.input_projection(observations), src_mask=causal_mask,
                               src_key_padding_mask=padding_mask, is_causal=True)
        return self.output_projection(encoded)


def make_dataset(seed=106, batch_size=30, max_length=6):
    """生成不同长度的有噪观测；第 2 个特征是固定采样时刻，不含未来观测。"""
    rng = np.random.default_rng(seed)
    lengths = 2 + np.arange(batch_size) % (max_length - 1)
    signal = rng.uniform(-1.0, 1.0, size=(batch_size, max_length)).astype('float32')
    valid = np.arange(max_length)[None, :] < lengths[:, None]
    positions = np.broadcast_to(np.arange(max_length, dtype='float32') / max_length, signal.shape)
    observations = np.stack([signal, positions], axis=-1)
    observations[~valid] = 0
    targets = (np.cumsum(signal, axis=1) / np.arange(1, max_length + 1)).astype('float32')
    targets[~valid] = 0
    return observations, targets[..., None], valid


def run(epochs=220, lr=0.02, device='cpu'):
    """完成连续训练并验证屏蔽区域扰动不泄漏，返回可打印的验收指标。"""
    torch.manual_seed(106)
    if not USE_NNLEARN:
        torch.set_num_threads(1)
    observations, targets, valid = make_dataset()
    sequence_length = observations.shape[1]
    x = torch.tensor(observations, device=device)
    y = torch.tensor(targets, device=device)
    valid_weights = torch.tensor(valid[..., None].astype('float32'), device=device)
    padding_mask = torch.tensor(~valid, device=device)
    causal_mask = torch.tensor(np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1),
                               device=device)
    model = PrefixMeanRegressor().to(device=device, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def loss_for(prediction):
        return (((prediction - y) ** 2) * valid_weights).sum() / valid_weights.sum()

    with torch.no_grad():
        initial_loss = loss_for(model(x, causal_mask, padding_mask)).item()
    model.train()
    for _ in range(epochs):
        loss = loss_for(model(x, causal_mask, padding_mask))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        prediction = model(x, causal_mask, padding_mask)
        final_loss = loss_for(prediction).item()
        validation_x, validation_y, validation_valid = make_dataset(seed=107)
        validation_output = model(torch.tensor(validation_x, device=device), causal_mask,
                                  torch.tensor(~validation_valid, device=device))
        validation_error = validation_output.detach().cpu().numpy() - validation_y
        validation_loss = float(np.mean(validation_error[validation_valid] ** 2))
        baseline = prediction.detach().cpu().numpy()
        changed_padding = observations.copy()
        changed_padding[~valid] = 100.0
        padded = model(torch.tensor(changed_padding, device=device), causal_mask, padding_mask)
        padding_error = float(np.max(np.abs(padded.detach().cpu().numpy()[valid] - baseline[valid])))
        changed_future = observations.copy()
        changed_future[:, 2:, 0] += 100.0
        future = model(torch.tensor(changed_future, device=device), causal_mask, padding_mask)
        causal_error = float(np.max(np.abs(future.detach().cpu().numpy()[:, :2] - baseline[:, :2])))

    assert np.isfinite(final_loss), '训练损失必须有限'
    assert final_loss < 0.008 and final_loss < initial_loss * 0.1, '前缀均值预测未达到收敛要求'
    assert validation_loss < 0.02, '独立验证序列的前缀均值误差过大'
    assert padding_error < 1e-5, 'padding 观测泄漏到有效位置'
    assert causal_error < 1e-5, '未来观测泄漏到前缀预测'
    return dict(backend='nnlearn' if USE_NNLEARN else 'pytorch', device=device,
                effective_tokens=int(valid.sum()), initial_loss=initial_loss,
                final_loss=final_loss, validation_loss=validation_loss,
                padding_error=padding_error, causal_error=causal_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    metrics = run(epochs=args.epochs, lr=args.lr, device=args.device)
    for name, value in metrics.items():
        print(f'{name}={value:.8f}' if isinstance(value, float) else f'{name}={value}')

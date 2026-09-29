# 第四篇配套代码

本篇继续使用第三篇的 `mini_nn`，并新增 `spatial.py` 中的卷积、最大池化和展平，以及 `attention.py` 中的 Embedding、多头注意力、LayerNorm 和最小编码器层。章节示例位于本目录。

从仓库根目录运行 CNN 示例：

```bash
python 'book/代码/04-进阶模型与运行时/18-卷积池化与CNN.py'
```

运行 Transformer 示例：

```bash
python 'book/代码/04-进阶模型与运行时/19-注意力与Transformer.py'
```

| 章节 | 脚本 | 关注点 |
| --- | --- | --- |
| 18 | `18-卷积池化与CNN.py` | NCHW 卷积、卷积梯度、最大池化、展平和合成图像分类 |
| 19 | `19-注意力与Transformer.py` | Embedding 梯度累加、多头注意力、LayerNorm 和编码器训练 |
| 20 | `20-CPU与CUDA后端.py` | NumPy CPU 路径和可选 CuPy/CUDA 设备搬运 |
| 21 | `21-与PyTorch对照并扩展算子.py` | 前向、梯度、SGD 更新与 sigmoid 的 PyTorch 对照 |

示例只使用本地合成数据，不下载数据集。卷积使用 NumPy 和 Python 循环；Transformer 不支持 mask、dropout、位置编码和因果注意力。第 20 章的 CuPy 路径要求本机有可用 CUDA；第 21 章的跨框架对照需要另行安装 PyTorch。两者均不是项目的必需依赖。

`mini_device.py` 只演示数组后端与设备搬运，不包含自动微分。第四篇的 `mini_nn` 仍是 NumPy CPU 教学实现。

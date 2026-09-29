# 第三篇配套代码

本篇在第二篇的 `mini_tensor.py` 上继续实现参数、模块、神经网络层、损失函数、SGD、数据批次和模型状态保存。可运行的教学实现位于 `mini_nn/`，各章脚本只展示当前章节的重点。

先在仓库根目录安装依赖：

```bash
python -m pip install -e .
```

从仓库根目录运行单章示例：

```bash
python 'book/代码/03-神经网络训练/17-训练MLP并保存模型.py'
```

| 章节 | 脚本 | 关注点 |
| --- | --- | --- |
| 11 | `11-Parameter与Module.py` | 参数注册、模块递归枚举、梯度清理与状态字典 |
| 12 | `12-Linear与激活函数.py` | 线性层、ReLU、Tanh、Sigmoid 与激活梯度 |
| 13 | `13-损失函数.py` | MSE、数值稳定的交叉熵及 logits 梯度 |
| 14 | `14-SGD与参数更新.py` | 清理梯度和一次 SGD 参数更新 |
| 15 | `15-训练第一个线性回归模型.py` | 全批次训练的完整步骤 |
| 16 | `16-Dataset与mini-batch.py` | 数组数据集、打乱、分批和丢弃末批次 |
| 17 | `17-训练MLP并保存模型.py` | mini-batch 分类训练、评估和 NPZ 状态恢复 |

`mini_nn` 由 `core.py`、`layers.py`、`losses.py`、`optim.py`、`data.py` 和 `checkpoint.py` 组成。脚本会从相邻的第二篇代码目录导入 `mini_tensor`，因此按上面的命令直接运行即可。

这是教学用的最小实现：`Linear` 只接受二维批次输入；数据加载器处理 NumPy 数组；优化器仅实现无动量 SGD；检查点只保存模型参数，不保存优化器状态或训练进度。它不兼容完整 PyTorch API。

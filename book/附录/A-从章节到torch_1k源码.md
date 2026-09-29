# A 从章节到 `torch_1k` 源码

本附录给出阅读路线：先读书中从零实现的教学代码，再沿下表定位仓库实现。前五章使用纯 Python 标量程序搭建直觉；`torch_1k` 是从 Tensor 开始的框架，所以这几章的概念没有逐类一一对应。

## 章节映射

| 书稿 | 主要源码或示例 | 阅读提示 |
| --- | --- | --- |
| 00 阅读指南 | [`docs/tutorial/torch-training-mechanism-20260929.md`](../../docs/tutorial/torch-training-mechanism-20260929.md) | 项目训练流程总览；不是框架 API 手册 |
| 01–05 标量自动微分 | [`torch_1k/function.py`](../../torch_1k/function.py)、[`torch_1k/tensor.py`](../../torch_1k/tensor.py) | 用框架计算图、反向执行和梯度累积回看标量阶段概念；仓库没有同构的标量 `Value` 类 |
| 06–10 Tensor 与自动微分 | [`torch_1k/tensor.py`](../../torch_1k/tensor.py)、[`torch_1k/functional/numeric.py`](../../torch_1k/functional/numeric.py)、[`torch_1k/functional/matrix.py`](../../torch_1k/functional/matrix.py)、[`torch_1k/functional/get_item.py`](../../torch_1k/functional/get_item.py) | 从 Tensor 的运算入口继续到数值、矩阵和索引算子；广播梯度归约可看 `sum_to` |
| 11–12 参数、模块和层 | [`torch_1k/nn/parameter.py`](../../torch_1k/nn/parameter.py)、[`torch_1k/nn/module.py`](../../torch_1k/nn/module.py)、[`torch_1k/nn/linear.py`](../../torch_1k/nn/linear.py)、[`torch_1k/nn/activation.py`](../../torch_1k/nn/activation.py)、[`torch_1k/nn/sequential.py`](../../torch_1k/nn/sequential.py) | 沿参数注册和 `forward` 调用追踪一次层组合 |
| 13 损失函数 | [`torch_1k/nn/loss.py`](../../torch_1k/nn/loss.py) | 比较 MSE 与交叉熵的前向/反向；重点留意 logits、标签和 reduction |
| 14–15 优化与训练 | [`torch_1k/optim/optimizer.py`](../../torch_1k/optim/optimizer.py)、[`torch_1k/optim/sgd.py`](../../torch_1k/optim/sgd.py)、[`examples/example1_linear_reg_simple.py`](../../examples/example1_linear_reg_simple.py) | 对照参数梯度清理、SGD 更新和线性回归训练循环 |
| 16 数据与 mini-batch | [`torch_1k/utils/data/dataset.py`](../../torch_1k/utils/data/dataset.py)、[`torch_1k/utils/data/data_loader.py`](../../torch_1k/utils/data/data_loader.py)、[`torch_1k/utils/data/sampler.py`](../../torch_1k/utils/data/sampler.py) | 查看数据集索引、批次拼接和采样职责 |
| 17 模型状态与分类训练 | [`torch_1k/nn/module.py`](../../torch_1k/nn/module.py)、[`examples/example7_mnist_dataloader_train_compare.py`](../../examples/example7_mnist_dataloader_train_compare.py) | 模块状态和数据加载训练的组合示例 |
| 18 CNN | [`torch_1k/nn/conv.py`](../../torch_1k/nn/conv.py)、[`torch_1k/nn/pool.py`](../../torch_1k/nn/pool.py)、[`torch_1k/nn/flatten.py`](../../torch_1k/nn/flatten.py)、[`examples/example5_mnist_cnn_train_compare.py`](../../examples/example5_mnist_cnn_train_compare.py) | 从 NCHW 窗口、梯度累加和池化索引开始读 |
| 19 Transformer | [`torch_1k/nn/transformer.py`](../../torch_1k/nn/transformer.py)、[`torch_1k/nn/sparse.py`](../../torch_1k/nn/sparse.py)、[`torch_1k/nn/normalization.py`](../../torch_1k/nn/normalization.py)、[`examples/example6_transformer_train_compare.py`](../../examples/example6_transformer_train_compare.py) | 对照注意力、Embedding、LayerNorm 和序列训练 |
| 20 CPU/CUDA 后端 | [`torch_1k/backend.py`](../../torch_1k/backend.py)、[`torch_1k/cuda.py`](../../torch_1k/cuda.py)、[`torch_1k/functional/device.py`](../../torch_1k/functional/device.py)、[`torch_1k/tensor.py`](../../torch_1k/tensor.py) | 阅读数组模块选择、设备搬运和 `Tensor.to()` 的梯度回传 |
| 21 PyTorch 对照与算子扩展 | [`examples/example39_training_mechanism_comparison.py`](../../examples/example39_training_mechanism_comparison.py)、[`torch_1k/functional/numeric.py`](../../torch_1k/functional/numeric.py) | 用同一条件逐项比较，并参照现有 `Function` 算子的组织方式 |

## 推荐的源码阅读顺序

1. 从 [`torch_1k/tensor.py`](../../torch_1k/tensor.py) 看公开对象保存什么状态，以及运算如何转到函数实现。
2. 再看 [`torch_1k/function.py`](../../torch_1k/function.py) 的输入记录、输出创建和反向调度。
3. 任选一个简单算子，例如 [`torch_1k/functional/numeric.py`](../../torch_1k/functional/numeric.py) 中的加法或乘法，前向和反向对照本书公式。
4. 最后沿 `Linear → CrossEntropyLoss → SGD → 示例训练循环` 读完整的一次更新。

## 两套实现的范围

书中 `book/代码/` 的程序刻意保持短小：`mini_tensor` 只支持 NumPy，矩阵乘法限二维；本书的 `mini_nn` 是教学用组合层。仓库 `torch_1k/` 包含更多算子、数据类型和设备路径，但仍是独立项目实现，并非完整 PyTorch。读源码时应区分“概念对应”与“接口完全相同”。

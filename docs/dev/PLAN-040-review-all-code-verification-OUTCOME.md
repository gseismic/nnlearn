# PLAN-040 全量代码复核与 PLAN-039 结果校验结果

日期：2026-09-29

计划文件：`docs/dev/PLAN-040-review-all-code-verification.md`

## 结论与范围

复核了 `torch_1k` 的张量与自动微分、数值及矩阵算子、神经网络层、优化器、数据管道、设备后端和对外入口，并检查了相关测试、示例及安装说明。`PLAN-039-review-all-code-OUTCOME.md` 的 **12 项实现问题均成立**，优先级划分总体合理，因此保留该历史审查记录，不修改其正文。其“当次未运行测试”的历史过程无法仅凭仓库独立证明；提交 `555cc18` 确实只包含 PLAN-039 的两份文档，没有运行时代码变更。

本次是审查任务；没有修改运行时代码。以下复现均在 CPU 上运行。当前 Python 环境未安装 CuPy，且 PyTorch 对照测试遇到 cuDNN 初始化失败，因此未完成 `torch_1k` 的 CUDA 路径验证。

## PLAN-039 逐项核验

| 编号 | 原结论及位置 | 复核证据 |
| --- | --- | --- |
| 1 | P1 `functional/matrix.py:609`，`matmul` 反向 | `x=[1,2]` 与 2×2 矩阵相乘后求和，矩阵梯度是标量 `3`，应为 `[[1,1],[2,2]]`；广播输入形状 `(1,4,5)` 的梯度却为 `(2,4,5)`。 |
| 2 | P1 `tensor.py:259`，`to()` 断图 | `(x*2).float().sum().backward()` 后，原 `x.grad` 为 `None`。 |
| 3 | P1 `optim/adam.py:22,55`，全局步数 | 两参数中第二个参数首步才有梯度时，学习率 `0.1` 令其从 `1` 变为约 `0.925586`；PyTorch 为约 `0.9`。 |
| 4 | P2 `functional/get_item.py:22`、`functional/pad.py:37`，重复索引 | `x[[0,0]].sum().backward()` 得到梯度 `[1,0]`，应为 `[2,0]`。 |
| 5 | P2 `functional/pad.py:20,37`，填充值及 dtype | 对 `float32` 的单元素输入调用 `pad(..., value=9)`，输出是 `[0,3,0]`、`float64`，填充值丢失。 |
| 6 | P2 `functional/numeric.py:881,921`，负索引 | `index_select` 和 `gather` 用 `-1` 均取到末项；PyTorch 均报越界。 |
| 7 | P2 `functional/numeric.py:1065,1112`，稳定排序 | 对六个相等元素执行 `argsort(descending=True, stable=True)` 得到 `[5,4,3,2,1,0]`；PyTorch 保持 `[0,1,2,3,4,5]`。 |
| 8 | P2 `nn/module.py:157`，状态加载校验 | 将 4×4 `int64` 数组加载到 `Linear(2,2).weight` 后，权重静默变成 4×4 `int64`。 |
| 9 | P2 `nn/normalization.py:16`，LayerNorm 形状 | `LayerNorm((2,1))` 接收形状 `(3,1,2)` 后输出及输入梯度均变为 `(3,2,2)`；PyTorch 拒绝该输入。 |
| 10 | P2 `functional/matrix.py:315,322`，Einsum 校验 | 合法的 `einsum('ii,jk->jk', eye(2), ones(3,4))` 报“重复标签维度不匹配”；第二个循环误用了前一循环最后一个 `shape`。 |
| 11 | P2 `nn/module.py:117`，共享模块参数 | 同一 `Linear` 挂在 `a` 和 `b` 后，`named_parameters()` 返回四项而非两项；直接传给优化器会重复更新。 |
| 12 | P2 `utils/data/data_loader.py:24`，样本结构 | `default_collate([(1,2),(3,4,5)])` 静默丢弃第二个样本的第三个字段；PyTorch 报错。 |

上述位置与 PLAN-039 文档一致；第 1、7、9 项现有文字虽较宽泛，但结论正确。第 7 项尤其包含降序时把同值元素顺序反转的确定性错误。

## 本次新增发现

### P2

1. **交叉熵静默截断浮点类标签。** `torch_1k/nn/loss.py:90` 在校验类型前将 `target` 强制转为 `int64`。例如标签 `[0.9]` 被当作类别 `0` 计算损失 `0.313261...`；PyTorch 对该标签报 dtype 错误。错误标签会悄悄参与训练。
2. **Embedding 静默转换或回绕无效索引。** `torch_1k/nn/sparse.py:11-13` 把 `1.9` 转成索引 `1`，把 `-1` 当作最后一行；PyTorch 分别报类型与越界错误。错误 token 可产生合法形状但错误内容的向量。
3. **模块调用无法传递关键字参数。** `torch_1k/nn/module.py:74` 只接受 `*inputs`。例如 `MultiheadAttention(x, need_weights=True)` 在其 `forward()` 支持该参数的情况下仍抛出 `TypeError`，导致公开参数无法通过正常模块调用使用。
4. **叶张量直接反向传播不产生梯度。** `torch_1k/tensor.py:54-56` 在 `creator is None` 时直接返回。`tensor(2., requires_grad=True).backward()` 后 `.grad` 为 `None`，而 PyTorch 为 `1`。
5. **多输出函数的未使用输出被释放后，反向传播崩溃。** `torch_1k/tensor.py:75` 直接读取弱引用结果的 `.grad`。自定义双输出 `Function` 的 `y = fn(x)[0]; y.backward()` 抛 `AttributeError: 'NoneType' object has no attribute 'grad'`。这使公开的 `Function` 扩展机制无法可靠处理多输出算子。
6. **保留中间梯度后再次反向传播会重复传递旧梯度。** `torch_1k/tensor.py:93-96` 把新梯度累加到已有中间节点 `.grad`，再用累计值继续传播。令 `x=2, y=x*x`，先 `y.backward(retain_grad=True)`，再 `(y*y).backward()`，得到 `x.grad=40`；两次真实贡献之和应为 `4+32=36`。此前文档误写为 `20`，本轮已修正。

### P3

7. **`Tensor.zeros_like()` 和 `Tensor.ones_like()` 类方法丢失输入 dtype。** `torch_1k/tensor.py:319-328` 未给 `xp.zeros/ones` 传入输入 dtype；`float32` 输入得到 `float64`。模块级 `torch_1k.zeros_like/ones_like` 有正确处理，此问题限于类方法。
8. **标量输入的随机类函数报错。** `torch_1k/nn/functional.py:19` 的 `dropout`、`torch_1k/tensor.py:544,554` 的 `rand_like/randn_like` 对零维随机结果直接调用 `.astype()`；NumPy 在该情形返回 Python 标量，因而抛 `AttributeError`。PyTorch 的标量 Dropout 可以正常运行。
9. **README 安装路径错误。** `README.md:22-23` 指示先 `cd torch_1k` 再执行 `pip install .`，但 `setup.py` 位于仓库根目录，照此命令不能安装本包。

## 测试与边界

- 当前 `pytest -q` 入口使用的 Python 3.10 环境未安装 PyTorch，收集阶段报 10 个 `ModuleNotFoundError`；这不是代码测试失败。
- 当前 `python -m pytest -q` 使用已安装 PyTorch 的 Python 3.12：**252 通过、1 失败**。唯一失败发生在 PyTorch 对照示例的 CUDA 卷积，报 `CUDNN_STATUS_NOT_INITIALIZED`；栈中未进入 `torch_1k` 卷积实现。
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q`：**253 通过**。本次新发现来自临时最小复现；现有测试通过不代表这些边界已被覆盖。
- 未做可用 GPU 上的 CUDA 梯度、数值或性能验证；也未对所有可能的输入组合做穷举或形式化证明。

## 建议修复顺序

先处理会静默产生错误训练结果的 `matmul`、`Tensor.to`、Adam、重复索引、状态加载、交叉熵及 Embedding 索引；再修复自动微分的叶节点、多输出与重复反向传播，以及公开 API 的参数传递与边界校验。每项修复应添加能复现本文件具体输入的回归测试。

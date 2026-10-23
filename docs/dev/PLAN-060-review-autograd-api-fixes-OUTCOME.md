# PLAN-060 实施结果：修复完整仓库 Review 的自动微分与接口问题

日期：2026-10-24

## 核实结论

用户给出的 13 条 Review 均成立，已全部修复。修复前 CPU 既有测试为 **297 passed**；首版新增回归为 **84 failed, 5 passed**，证明原有测试未覆盖这些问题。复核时补充 float32 的 MSE 高阶梯度检查，最终新增回归共 92 项。

## 逐项变更

| Review | 修复与验证 |
| --- | --- |
| P1：损失实例重复调用丢梯度 | MSELoss、CrossEntropyLoss 改为可复用 Module，每次调用创建私有 Function；覆盖三种 reduction、不同批次形状，以及训练前向后插入 no_grad 调用的情况。 |
| P1：首次保存梯度共享或只读 | 首次累积使用可微 clone，获得独立可写存储；覆盖规约梯度清零、两个参数和保留输出之间的梯度隔离，以及 create_graph 两种模式。 |
| P2：Gather/Scatter 索引形状 | Gather 前向、Scatter/ScatterAdd 的 src 反向按 index.shape 的坐标选择，避免 take_along_axis 广播；与 PyTorch 对照部分维度、负维度、三维和空索引的值与梯度。 |
| P2：MSE 二阶导断图 | 反向使用 self.inputs 的 Tensor 表达式，并将均值系数保留为同精度常数；覆盖 float32/float64、三种 reduction、广播目标及预测和目标的二阶梯度依赖。 |
| P2：模块转换破坏整数/布尔 buffer | Module.to 对浮点和复数应用 dtype，整数及布尔 Tensor 只迁移设备；验证递归子模块、Embedding 索引、mask、计数器及浮点参数/梯度。 |
| P2：优化器状态精度不匹配 | _state_array 同时对齐 parameter.device 和 parameter.dtype，并保留独立拷贝；覆盖 Adam、AdamW、MomentumSGD 从 float64 恢复到 float32 及后续两次更新。 |
| P2：创建接口形状未展开 | Tensor.zeros/ones/randn 调用 _get_shape 时展开参数；覆盖顶层及类方法的元组、列表、空形状、展开参数、dtype 与 requires_grad。 |
| P2：规约位置参数错误 | Tensor.sum/mean 签名调整为 dim、keepdim，axis/keepdims 为关键字别名；验证输出形状、数值与反向梯度。 |
| P2：item 拒绝非零维单元素 | 按 numel()==1 判断，空或多元素张量抛出 ValueError；验证标量、向量、矩阵单元素的 item 与 bool。 |
| P2：零次幂产生 NaN 梯度 | 在计算负次幂前处理零指数；用严格 NumPy 浮点异常检查零点、正负输入和含常数项的多项式梯度。 |
| P2：强引用模式不能反向 | 新增 Function.get_outputs，按实际引用类型解析，并统一反向引擎及 8 类算子的读取；覆盖强引用和前后向之间切换配置的路径。 |
| P2：全局 argmax 忽略 keepdim | 全局规约后恢复所有长度为 1 的原始轴；覆盖 0 至 3 维输入、顶层/方法及别名入口、索引 dtype 与无梯度属性。 |
| P2：性能基准模型精度不同 | 两边模型显式 to(device=..., dtype=float32)，输出显示精度；回归验证建模及一次真实训练更新后的参数 dtype，历史 PLAN-057 记录增加更正。 |

## 验证结果

- `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest tests/test_48_review_autograd_api.py -q`：**92 passed**。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest -q`：**389 passed**，包含原有 297 项及新增 92 项。
- `CUDA_VISIBLE_DEVICES='' uv run --no-project python examples/704_compat_training_performance_benchmark.py --device cpu --steps 50 --warmup 2 --repeats 1`：退出码 0，两边 loss 均为 `0.000751`，accuracy 均为 `1.0000`；loss 差值 `1.74623e-10`，accuracy 差值为 0。该次运行用于验证训练闭环，不作为稳定性能测量。
- 环境：NumPy 2.2.6、PyTorch 2.6.0+cu124，CPU。
- CUDA 未实测：当前环境 `nnlearn.cuda.is_available()` 为 False。
- `git diff --check`：通过。

## 代码复核

- 检查了每次损失调用的节点生命周期、输入与缓存所有权，模块元数据不再承担反向节点状态。
- 检查了梯度克隆处于 create_graph 控制范围内，能够同时隔离存储和保留高阶依赖。
- 检查了所有直接调用 outputs 弱引用的算子均已改为统一解析，反向不依赖当前全局引用配置。
- 检查了索引坐标在非索引维度按 index.shape 生成，Gather 的重复索引仍使用 add.at 累积。
- 检查了优化器状态与源检查点保持存储隔离，模块整数/布尔 buffer 在精度转换中保留类型。
- 检查了 Tensor 接口的关键字别名兼容性和基准两端实际训练的参数精度。

## 交付

- 计划：`docs/dev/PLAN-060-review-autograd-api-fixes.md`。
- 结果：`docs/dev/PLAN-060-review-autograd-api-fixes-OUTCOME.md`。
- 按仓库要求纳入中文提交，提交后立即推送。

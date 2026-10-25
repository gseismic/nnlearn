# PLAN-064 实施结果：神经网络算子的高阶导数

日期：2026-10-25。

## 实际变更

1. LayerNorm 在 `create_graph=True` 的反向中重新用 Tensor 计算统计量和解析梯度，保留输入、weight、bias 和非线性上游依赖；普通一阶路径继续使用数组缓存。
2. 交叉熵在高阶反向中重算可微概率，以常量标签分布应用类别权重、忽略掩码与平滑；支持三种 reduction 和空间输入的轴恢复。
3. Conv2d 的高阶反向用可微窗口切片、pad、stack 和矩阵乘法表达输入与权重梯度；支持现有 stride、padding、bias 子集及交叉二阶导，普通一阶实现保留。
4. 新增 `docs/autograd.md`，明确已验证范围、常量标签/权重、非标量反向种子、梯度累积、内存开销和未覆盖的模块边界。

## 复现及验证

- 修复前首版 23 项为 **22 failed, 1 passed**：一阶数值正确，但二阶反向无法回到原输入；独立差分用例同样证明断图。
- 首轮新增 **29 项**，包含 PyTorch 全部输入/参数一阶及联合方向二阶对照、三个算子的独立中央差分、float16/32/64 以及重复反向；独立复查后补充 4 项，最终共 **33 项**。
- 新高阶回归加 PLAN-063 数值稳定性回归：**67 passed**。
- CPU 既有及 PLAN-063/064 回归（排除并行开发中的 test_54/55）：**747 passed, 22 skipped**。
- `ruff check tests/test_53_higher_order_nn.py nnlearn/nn/conv.py nnlearn/nn/loss.py nnlearn/nn/normalization.py` 与 `git diff --check` 通过。
- CUDA 二阶联合验证随 PLAN-067 记录，不用 CPU 证据替代设备验收。

## 复查

- 三个高阶分支都读取原 Tensor，未将 ndarray 重新包装成无依赖的叶子来代替可微中间量；标签和类别权重才作为常量。
- 半精度转换使用 Tensor.to，梯度在引擎回传边恢复输入 dtype；连续一阶训练路径与数值稳定性测试均通过。
- 卷积窗口重叠通过可微 pad 加法累积，输入、权重和偏置的 Hessian 交叉项均有对照。

### 独立复查补修

独立复查复现了两项边界缺陷：半精度输入搭配 float64 仿射参数时，高阶分支把原本有限的 `1e40` 上游降为 float32，导致无穷与 NaN；合法空批次卷积因 `(0,-1)` 形状推断报错。修复前新增 4 项为 **3 failed, 1 passed**。

修复后仅提升半精度上游，保留显式更高精度；卷积窗口和上游 reshape 使用明确的窗口大小与空间位置数。新增检查覆盖 float32/float64 参数与大幅值上游、两种 stride/padding 下的空批次一阶及二阶 PyTorch 对照。高阶与半精度联合 **71 passed**，ruff 通过；完整联合回归和 CUDA 结果在 PLAN-067 汇总。

## 交付

- 设计：`docs/design/autograd-20261025-higher-order.md`。
- 计划：`docs/dev/PLAN-064-higher-order-autograd.md`。
- 结果：`docs/dev/PLAN-064-higher-order-autograd-OUTCOME.md`。
- 使用与边界：`docs/autograd.md`。

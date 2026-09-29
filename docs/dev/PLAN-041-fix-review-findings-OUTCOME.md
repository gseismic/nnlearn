# PLAN-041 修复全量审查发现结果

日期：2026-09-29

计划文件：`docs/dev/PLAN-041-fix-review-findings.md`

## 完成情况

已修复 PLAN-039 的 12 项实现问题和 PLAN-040 新增的 9 项问题，共 21 项。新增 `tests/test_45_review_fix_regressions.py`，以原审查中的失败输入为主要回归场景；原有测试中把“叶张量直接反向后梯度为 None”当作预期的一项断言已更正。

### 自动微分与张量

- `matmul` 反向将一维输入临时提升为矩阵，计算后按广播规则归约，并还原到原始形状；覆盖向量乘矩阵、矩阵乘向量、点积及批次广播。
- 浮点 dtype 或设备转换现在由 `CopyTo` 记录在计算图中；转换的梯度返回原设备和 dtype。同设备同 dtype 时返回原张量。
- `GetItemGrad` 对重复高级索引使用散射累加，保持梯度 dtype，并提供对应的反向索引路径。
- `Tensor.backward()` 可为叶张量产生单位梯度；反向传播使用本次调用的局部梯度，避免保留的旧中间梯度再次向下传播；已释放的多输出弱引用按未使用输出处理。
- `Tensor.zeros_like/ones_like` 类方法沿用输入 dtype；标量 `rand/randn/rand_like/randn_like` 和 Dropout 可处理零维输入。

### 算子、模块和数据管道

- Pad 按指定值及输入 dtype 创建输出；`index_select/gather` 拒绝负索引和上界越界索引；稳定排序在升序与降序时都保持同值元素的原始顺序；Einsum 使用当前操作数形状校验重复标签。
- Adam/AdamW 分别记录每个参数的实际更新次数，状态文件保存每参数 `step`，同时保留全局 `t` 以兼容现有使用；旧状态缺少 `step` 时以 `t` 作为恢复值。
- `load_state_dict()` 先核对键与全部形状，再复制数据；可转换 dtype 被转换成目标张量原有 dtype，不能按同类规则转换的 dtype 报错。`named_parameters/named_buffers` 对共享对象去重，同时状态文件保留共享模块的别名键。
- LayerNorm 在计算前校验输入尾部形状；交叉熵拒绝非整数类别标签；Embedding 拒绝非整数、负数及超出词表的索引；默认批处理校验 tuple/list 长度和字典键；`Module.__call__` 将关键字参数交给 `forward()`。
- README 的安装命令改为在仓库根目录执行 `pip install .`。

## 文档复核修正

PLAN-040 结果中重复反向传播例子的正确梯度曾误写成 `20`。对 `x=2, y=x*x`，第一次 `y.backward()` 贡献 `4`，第二次 `(y*y).backward()` 贡献 `32`，正确累计值为 `36`；旧实现得到 `40`。已同步修正该结果文档和回归测试。

## 代码审查与测试

- 检查了修改后的梯度形状、dtype、设备转换、状态文件兼容、共享对象与错误输入路径，修正了回归测试中的错误数学期望。
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q`：**288 passed**。
- `CUDA_VISIBLE_DEVICES='' python examples/example38_beginner_tutorial.py`：准确率 `1.0`，回归损失约 `0.000013`。
- `CUDA_VISIBLE_DEVICES='' USE_TORCH_1K=1 python examples/example36_pytorch_training_baseline.py`：MLP、CNN、Transformer 准确率均为 `1.0`，MLP 检查点往返误差 `0`。
- 当前环境没有 CuPy，无法验证 `torch_1k` 的 CUDA 运行路径；CPU 回归测试和示例均已通过。提交后的独立复审按用户要求继续进行，若有新问题，将先写入文档再修复。

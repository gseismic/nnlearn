# PLAN-039 全量代码静态审查结果

日期：2026-09-29

计划文件：`docs/dev/PLAN-039-review-all-code.md`

## 审查范围与方式

静态审查了 `torch_1k` 运行库中的张量与自动微分、矩阵和数值算子、神经网络模块、优化器、数据加载及后端实现，并参考相关示例、测试和设计文档核对行为。没有修改实现，也没有运行测试。

## 发现

### P1

1. **`matmul` 反向传播不支持常见的一维和批次广播情况。** `torch_1k/functional/matrix.py:609` 使用矩阵乘法转置公式计算梯度。向量乘矩阵、矩阵乘向量需要外积梯度，当前路径可能报错或产生不匹配的形状；批次维广播时也没有将梯度归约回原输入形状。
2. **`Tensor.to()` 会断开计算图。** `torch_1k/tensor.py:259` 创建了新的 Tensor，但没有连接原 Tensor 的 creator。需要梯度的输入经过 `.to()`、`.cpu()`、`.cuda()` 或 `.float()` 后，反向传播不会回到原 Tensor。
3. **Adam 对所有参数共用全局步数。** `torch_1k/optim/adam.py:22` 每次 `step()` 都递增 `self.t`，而 `torch_1k/optim/adam.py:55` 用它计算所有参数的偏差修正。某参数无梯度期间，全局步数仍前进；该参数后续获得梯度时会使用错误的修正步数。

### P2

4. **重复高级索引的梯度会被覆盖。** `torch_1k/functional/get_item.py:22` 将反向梯度交给 Pad；`torch_1k/functional/pad.py:37` 用索引赋值回填。`x[[i, i]]` 这类重复索引应累加梯度，但当前赋值会丢失重复项。
5. **Pad 忽略填充值并可能提升 dtype。** `torch_1k/functional/pad.py:20` 将传入的 `value` 固定为 `0`；`torch_1k/functional/pad.py:37` 未指定填充数组 dtype，CPU NumPy 默认创建 `float64`。直接调用 Pad 会得到错误填充值，切片反向也可能生成与输入 dtype 不同的梯度。
6. **索引算子接受负索引并按 NumPy 规则回绕。** `torch_1k/functional/numeric.py:881` 的 `index_select` 和 `torch_1k/functional/numeric.py:921` 的 `gather` 没有检查索引下界；负索引会选择末尾元素，而不是作为越界索引报错。
7. **排序接口的 `stable=True` 没有生效。** `torch_1k/functional/numeric.py:1065` 和 `torch_1k/functional/numeric.py:1112` 调用底层排序时未应用 `stable` 参数，因此重复值的索引顺序不受稳定排序保证。
8. **`load_state_dict()` 不校验形状和 dtype。** `torch_1k/nn/module.py:157` 主要检查键名，随后直接替换目标参数的 `.data`。不匹配的检查点可能静默改变参数形状或 dtype，错误延迟到后续计算才暴露。
9. **LayerNorm 未验证输入尾部形状。** `torch_1k/nn/normalization.py:16` 没有检查输入尾部维度是否与 `normalized_shape` 相同。形状不匹配时可能发生广播，返回意外形状并使输入梯度形状不正确。
10. **Einsum 重复标签校验使用了最后一个操作数的形状。** `torch_1k/functional/matrix.py:315` 的循环未绑定当前操作数的 `shape`，因此 `torch_1k/functional/matrix.py:322` 使用前一个循环残留的形状校验每个操作数。合法的对角缩约式可能因此被误判为维度不匹配。
11. **共享子模块会让参数重复进入优化器。** `torch_1k/nn/module.py:117` 的 `named_parameters()` 递归时没有去重。同一个子模块挂载到两个属性后，`parameters()` 会重复返回其中的参数，优化器可能在一次 `step()` 中更新同一参数多次。
12. **默认批处理会静默截断结构不一致的样本。** `torch_1k/utils/data/data_loader.py:24` 对 tuple/list 使用 `zip(*batch)`；样本字段数不一致时，结果会按最短样本截断，额外字段不会报错。

## 结果

- 共记录 3 项 P1 和 9 项 P2 发现。
- 本次仅新增审查文档，未修改运行时代码。
- 未运行测试；上述结论来自静态阅读和代码路径分析。

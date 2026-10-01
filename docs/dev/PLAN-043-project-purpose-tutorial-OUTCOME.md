# PLAN-043 项目定位与训练机制对照教程结果

日期：2026-09-29

计划文件：`docs/dev/PLAN-043-project-purpose-tutorial.md`

实施基线 Git：`e7633ee`

结果提交 Git：`1b838dd`（明确项目教学定位并新增训练机制对照教程）。

## 实施内容

1. 新增 `docs/design/nlearn-20260930-project-purpose.md`，明确项目面向想理解自动微分与训练机制的 PyTorch 学习者，说明项目价值、范围、边界和后续工作判断标准。
2. 新增 `examples/702_compat_training_mechanism_comparison.py`，用同一输入、目标、初始权重、偏置和学习率，分别执行 `nlearn` 和 PyTorch 的线性前向、MSE、反向传播和 SGD 更新，并逐项打印最大绝对误差。
3. 新增 `docs/tutorial/torch-training-mechanism-20260929.md`，解释训练公式、代码步骤、输出含义、运行依赖和兼容边界。
4. 重写 `README.md` 为中文项目入口，移除“1000 行”和宽泛替换承诺，增加教程阅读路径、安装方法和支持范围。
5. 更新 `docs/design/nlearn-20260624-core-cuda-roadmap.md`，把 3000 行建议标记为历史目标并链接到当前定位。
6. 更新 `docs/design/nlearn-20260624-principles.md`，明确该文记录的是历史源码快照，旧目标和边界不代表当前项目承诺。

## 复核与运行结果

- 示例命令：`python examples/702_compat_training_mechanism_comparison.py`
- 结果：退出码 `0`；前向、损失、权重与偏置梯度、更新后参数均在 `1e-6` 容差内一致。
- 所有对照项的最大绝对误差：`2.3841858e-07`。
- 偏置梯度非零；偏置和权重在 SGD 更新后均发生变化。
- `git diff --check`：通过。
- 本轮未新增或运行测试套件；新增对照示例已直接执行。

## 边界

1. 数值对照只覆盖 CPU 上的一层线性模型和一次 SGD 更新，不证明其他模型或任意 PyTorch 脚本兼容。
2. 本轮未在 CUDA 设备上运行示例；CUDA 路径仍取决于 CuPy、驱动和设备环境。
3. PyTorch 仅为此对照示例的可选参考依赖，不由项目安装配置自动安装。

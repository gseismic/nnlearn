# PLAN-049 编写设备后端与 PyTorch 对照章节

日期：2026-09-29

## 背景

第四篇的 CNN 与 Transformer 章节已完成。接下来解释 CPU/CUDA 的数组分发边界，并用同一计算对照教学 Tensor 与 PyTorch；再通过新增 sigmoid 算子演示如何扩展自动微分引擎。

当前环境可导入 PyTorch，但未安装 CuPy。因此 CPU 路径和数值对照可直接运行；CUDA 示例需要在装有 CUDA 与匹配 CuPy 的环境中运行，当前环境应清楚报告跳过原因。

## 目标

1. 完成第 20、21 章以及对应脚本和代码说明。
2. 实现一个小型 NumPy/CuPy 设备数组示例，展示设备选择、显式搬运、同设备运算和可选 CUDA 路径。
3. 给基础 Tensor 增加稳定 sigmoid 运算及反向梯度。
4. 在相同输入、参数和学习率下，对照本书 Tensor 与 PyTorch 的预测、损失、梯度和一步 SGD 更新。
5. 更新全书导航与阶段结果；提交前复核实现限制、相对链接和运行输出。

## 实施范围

- 新增 `20-CPU与CUDA后端.md`、`21-与PyTorch对照并扩展算子.md`。
- 新增 `book/代码/04-进阶模型与运行时/mini_device.py`、两个章节脚本。
- 在 `book/代码/02-Tensor与自动微分/mini_tensor.py` 增加 `Tensor.sigmoid()` 及稳定的反向实现。
- PyTorch 仅作为可选数值对照依赖；CUDA 路径不下载依赖、不假设当前机器有 GPU。
- 设备数组示例只讲存储、分发与数据搬运，不将第四篇的教学 `mini_nn` 宣称为完整 CUDA 自动微分框架。
- 不新增测试套件；运行章节脚本，记录 CuPy 不可用时的跳过行为和 PyTorch 对照误差。

## 验收标准

1. CPU 路径只依赖 NumPy；数组跨设备需显式搬运；不同设备的张量不能直接参与同一运算。
2. 当 CuPy 与 CUDA 可用时，CUDA 路径能执行矩阵乘法并显式回到 CPU；不可用时给出可理解的提示。
3. Sigmoid 前向对大正数/负数保持数值稳定，反向为 `g * y * (1-y)`，并接入 Tensor 计算图。
4. 教学 Tensor 与 PyTorch 在预测、损失、参数梯度和单步更新上误差处于 `1e-10` 容差内；Sigmoid 前向和梯度也完成对照。
5. 书稿 Markdown 相对链接和 `git diff --cached --check` 通过，OUTCOME 记录实测环境与结果。

## 实施步骤

1. 建立本计划，检查 `nlearn.backend` 与现有设备搬运接口。
2. 编写 NumPy/CuPy 设备数组抽象和第 20 章脚本。
3. 在基础 Tensor 中新增 sigmoid 前向/反向，编写 PyTorch 对照脚本与第 21 章。
4. 运行两个示例、检查误差和相对链接，更新目录及代码 README。
5. 完成代码复核和 OUTCOME，使用中文提交并推送。

## 非目标

1. 不将 `mini_tensor` 或 `mini_nn` 重构为任意后端通用框架。
2. 不在无 CUDA 的环境安装驱动或 CuPy，也不承诺当前运行机能验证 GPU 内核。
3. 不要求 PyTorch 与教学框架 API 一致；只比较本章选定的有限计算。

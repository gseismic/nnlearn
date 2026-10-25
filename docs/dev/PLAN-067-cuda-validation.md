# PLAN-067：CUDA 实机验收

## 目标

在真实 NVIDIA GPU 上验证张量迁移、MLP/CNN/Transformer 前后向与连续训练、四种优化器和检查点恢复，并建立可复用的硬件测试门禁。

## 范围与实施顺序

1. 核对驱动、CUDA Toolkit、Python 和 CuPy 官方兼容表；使用 `uv` 管理隔离测试环境，复用已有 PyTorch，不修改系统驱动或系统 Python 依赖。
2. 新增 `tests/test_55_cuda_training.py`：无 GPU 时明确跳过；设置 `NNLEARN_REQUIRE_CUDA=1` 时缺失 CuPy、GPU 或必要硬件能力必须失败。
3. 比较 CPU/CUDA 前向、输入与参数梯度、连续训练结果及 dtype/device；覆盖 MLP、包含卷积与池化的 CNN、Transformer。
   - 三模型逐步参数对照显式使用 `Adam(eps=1e-4)`：实机已观察到 Transformer key 投影偏置的数学零梯度在 CPU/CUDA 规约后出现约 `1e-9` 舍入，默认 `eps=1e-8` 会把它放大到约 `0.003` 的参数差。保留原有数值容差，用明确的训练配置隔离零梯度舍入放大；四优化器固定非零梯度对照及同设备恢复继续使用默认 eps。
4. 对适用路径增加 PyTorch CUDA 对照；验证 CPU→CUDA→CPU 的 Module 迁移，保留参数对象及优化器引用、整数与布尔 buffer 的 dtype。
   - PyTorch 参考固定使用原生 CUDA 卷积分支（`torch.backends.cudnn.flags(enabled=False)`），保持独立 GPU 数值对照。现有环境的 cuDNN 包元数据为 9.1.0.70，实际加载版本为 92000，独立卷积报 `CUDNN_STATUS_NOT_INITIALIZED`；原生 CUDA 卷积前后向已独立验证成功。nnlearn/CuPy 本身不依赖 cuDNN，测试不需要额外安装或修改该可选加速库。
5. 验证仓库已有的 SGD、MomentumSGD、Adam、AdamW 的多步更新、状态保存与跨设备恢复后继续训练。
   - 增加 CPU/CUDA/PyTorch 四种优化器逐步对照。
   - 补充 Conv2d、LayerNorm、CrossEntropy 的 CUDA Hessian 向量积对照，以及半精度大类别/大方差回归。
6. 如验收揭示 CUDA 设备或底层算子缺陷，先更新本计划，再实施修复；正在并行修改的卷积、数值规约、归一化、损失与 Transformer 文件由对应负责人修复。
   - 全量实机首次运行发现 CuPy 13.6 缺少 `put_along_axis`，影响 Max/Min/TopK/Sort 反向；本任务在 backend 增加内部兼容散射，由数值算子负责人接入四处反向调用，并增加三维输入、不同轴、两种精度及空 TopK 的独立 PyTorch CUDA 回归。
   - 修正既有 `test_cuda_shape_api_if_available` 对 `flatten(start_dim=0)` 的错误预期，正确输出为 `(6,)`。
   - PyTorch 兼容训练子进程也要显式选择原生 CUDA 卷积，避免已确认损坏的现有 cuDNN 环境影响测试参考；该测试继续运行真实 CUDA 训练。
7. 复核新增代码，执行真实设备测试及必要的 CPU 回归，记录完整命令、硬件与依赖版本、通过/跳过数。
8. 编写 `docs/cuda.md` 安装与验收说明及 `docs/dev/PLAN-067-cuda-validation-OUTCOME.md`，交由主代理集成提交和推送。
9. 联合集成验收：在 CUDA 运行变长掩码 Transformer 的完整训练示例；主代理在中文 README 增加测试、高阶导数、掩码训练和 CUDA 文档入口，更新开发索引并核对最终提交的远程 CI。

## 验收标准

- 真实 GPU 门禁运行全部通过，不能以跳过代替 CUDA 验证。
- CPU/CUDA/PyTorch 对照使用明确数值容差，并检查梯度、参数、优化器状态的 dtype/device。
- 三类模型执行多步训练且损失下降；恢复的模型和优化器与不中断训练结果一致。
- 无 GPU 环境明确显示跳过原因；强制门禁对同一条件明确失败。
- 文档包含已经执行的环境、命令、结果及实际支持边界。

## 计划进度

- [x] 核对仓库约定与官方 CuPy 13.6 安装兼容表。
- [x] 准备隔离环境并完成 CUDA kernel 冒烟（CuPy 13.6.0 / NumPy 2.2.6，RTX 3060，CUDA 11.5）。
- [x] 实施并复核硬件测试与必要修复。
- [x] 执行真实 GPU 和 CPU 回归验证：72 项专项及全量 947 项通过，无跳过；无 GPU 时的显式跳过和强制失败也已验证。
- [x] 编写安装说明与结果文件。
- [ ] 主代理完成 README/索引入口、集成提交推送及最终远程 CI 核对。

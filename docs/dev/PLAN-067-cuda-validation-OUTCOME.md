# PLAN-067：CUDA 实机验收结果

## 已实现的变更

- `tests/test_55_cuda_training.py`：新增真实 CUDA 训练、迁移、优化器、检查点、掩码、半精度和高阶导数对照；缺少 GPU 时明确跳过，`NNLEARN_REQUIRE_CUDA=1` 时明确失败。
- `nnlearn/backend.py`：增加内部 `_put_along_axis`，使用原生 GPU 高级索引兼容 CuPy 缺少的写回 API；调用限定每个切片中的选中索引互不重复，符合 Max/Min/TopK/Sort 的使用场景。
- `tests/test_24_shape_api.py`：修正既有 GPU 测试中 `flatten(start_dim=0)` 的输出预期为 `(6,)`。
- `tests/test_42_pytorch_training_baseline.py`：PyTorch 参考子进程明确使用原生 CUDA 卷积，继续验收三类模型的 GPU 训练和恢复。
- `docs/cuda.md`：补充兼容性选择、隔离安装、硬件强制门禁、测试覆盖及实机记录。
- `README.zh-CN.md`：增加高阶导数、掩码训练、固定测试环境、CUDA 验收及计划结果索引入口。

`nnlearn/functional/numeric.py` 的四个写回调用已统一接入兼容后端，并通过全部新增散射对照。

## 实机环境

| 项目 | 实际版本 |
| --- | --- |
| 验收日期 | 2026-10-25 |
| 系统 | Linux 6.8.0-52-generic / x86_64 / glibc 2.35 |
| GPU | NVIDIA GeForce RTX 3060，12 GiB，Compute Capability 8.6 |
| NVIDIA 驱动 | 550.144.03，驱动 CUDA 上限 12.4 |
| 系统 Toolkit / NVRTC | 11.5 / 11.5 |
| Python | 3.12.8 |
| NumPy / CuPy | 2.2.6 / 13.6.0（cupy-cuda11x） |
| fastrlock / loguru / pytest | 0.8.4 / 0.7.3 / 9.0.2 |
| PyTorch | 2.6.0+cu124，原生 CUDA 卷积参考 |
| 隔离环境 | `/tmp/nnlearn-cuda-validation-venv` |

使用 `uv venv --system-site-packages` 复用既有 PyTorch；所有新增包均放在隔离环境，系统驱动和原 Python 环境没有修改。官方源下载两次明确超时后改从镜像获取同一 wheel，SHA256 为 `0418f788908985cb615d4c8fe1dda46bf0c09462626e643eda694b33002ae296`，已核对一致。

## 已确认的测试证据

1. CuPy kernel 编译、实际 GPU 数组分配、cuBLAS 矩阵乘法和同步成功；`nnlearn.cuda.is_available()` 为 True。
2. 完整硬件门禁 **72 passed，0 skipped**，命令：

   ```bash
   NNLEARN_REQUIRE_CUDA=1 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python -m pytest tests/test_55_cuda_training.py -q -rs
   ```

3. 同一 CuPy 环境中的半精度 CPU 回归 **38 passed**，神经网络高阶导数回归 **33 passed**。
4. 设置 `CUDA_VISIBLE_DEVICES=''` 后，最终普通运行 **72 skipped**，逐项说明 `cudaErrorNoDevice`；相同条件增加 `NNLEARN_REQUIRE_CUDA=1` 后立即非零退出。无 CuPy 的原解释器同样验证了显式跳过和强制失败。
5. 兼容散射函数独立对照 NumPy，覆盖不同形状、各轴、标量/数组值和空索引，**42 个互不重复索引场景通过**。
6. 修正旧 flatten 预期与参考子进程后，两个定向真实 GPU 测试 **2 passed**。

7. 独立执行带 causal/padding 掩码的 Transformer CUDA 示例，220 步训练损失从 `0.63198376` 降至 `0.00011298`，验证损失 `0.00027135`，padding/未来扰动误差均为 0：

   ```bash
   uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python examples/106_training_masked_transformer_compare.py --device cuda
   ```

8. 最终完整回归 **947 passed，0 skipped**，所有新增测试和既有 CUDA 路径都实际执行：

   ```bash
   NNLEARN_REQUIRE_CUDA=1 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python -m pytest tests -q -rs
   ```

全部验证依据是实际功能和数值结果；并行运行期间的耗时不作为性能基准。

9. 无 CuPy 的 CPU 环境最终运行 `CUDA_VISIBLE_DEVICES='' uv run --no-project python -m pytest tests -q -rs`：**852 passed, 95 skipped**。95 项全部是明确报告的 CUDA 路径；相同用例已在上述真实设备全量验收中全部执行。

## 计划与交付

- 本次覆盖 PLAN-063 半精度、PLAN-064 高阶导数、PLAN-065 掩码、PLAN-066 持续集成及 PLAN-067 实机验收的联合工作树。
- 计划：`docs/dev/PLAN-067-cuda-validation.md`。
- 结果：`docs/dev/PLAN-067-cuda-validation-OUTCOME.md`。
- 安装与复现：`docs/cuda.md`；开发索引：`docs/dev/INDEX.md`。
- 主代理已提交并推送核心代码 `01cbcce`；[最终集成 CI 40169550170](https://github.com/gseismic/nnlearn/actions/runs/40169550170) 五个作业全部成功，覆盖两套 Python/NumPy、PyTorch CPU 及八个发行产物安装。两套无 PyTorch 环境均实际报告 **233 passed, 184 skipped**；参考框架与 CUDA 的跳过均明确显示原因。
- 中文 README 和索引已关联五份计划及结果；文档入口和所有本地链接已检查有效。最终文档归档提交继续自动触发同套 CI。

## 验收中修正的问题与测试配置

- 完整 CuPy 环境首次运行得到 `912 passed / 5 failed`，揭示此前 CPU 跳过所隐藏的问题：CuPy 13.6 没有 `put_along_axis`，导致 Min/TopK/Sort 反向报错；同一 API 还用于 Max 反向，需要一并接入兼容散射。新增三维、多轴、float32/float64、空 TopK 的 PyTorch CUDA 对照。
- 旧 flatten GPU 测试的 `(6, 1)` 预期不符合默认展平至最后一轴的语义，已改为 `(6,)`。
- 现有 cuDNN 包元数据是 9.1.0.70，但实际头文件及运行时为 9.20，独立 PyTorch CUDA 卷积报 `CUDNN_STATUS_NOT_INITIALIZED`。明确选择原生 CUDA 卷积后独立前后向成功，参考测试保持 GPU 执行；未修改或替换现有 cuDNN。
- Transformer 的 key 投影偏置具有数学零梯度，CPU/CUDA 约 `1e-9` 的规约舍入被默认 Adam `eps=1e-8` 放大到约 `0.003` 参数差。三模型逐步 CPU/CUDA 对照显式使用 `eps=1e-4`，保留全部参数、梯度和数值容差检查；四优化器固定非零梯度对照与同设备恢复保持默认 eps。

## 复核

新增测试不以设备可用探测代替实际执行，同时检查数据和梯度的物理 CuPy 后端、dtype、有限值和参数所有权。float64 常规对照默认容差为 `rtol=1e-9, atol=1e-10`；高阶路径与半精度小梯度另有明确容差。目标硬件门禁不得跳过。backend、参考训练测试与新增测试已通过 ruff 检查和 diff 空白检查。

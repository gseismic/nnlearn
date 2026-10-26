# 训练性能 profile 调查

- 编写时间：2026-10-26 21:27 CST（Asia/Shanghai）
- 库代码基准：`a924b1fabf34f9064b0523f858f7b4a87f9b82e5`
- 执行者：Codex（GPT-6；当前环境未提供更细的模型版本标识）
- 关联计划：[PLAN-068](PLAN-068-training-performance-profile.md)
- 原始数据：[profiles/PLAN-068](profiles/PLAN-068/README.md)

## 1. 调查结论

用户的目标是先查清 nnlearn 与 PyTorch 的训练速度差距，为有针对性的优化选择目标。当前证据支持先优化已有计算路径：

1. **存在可避免的计算。** 两层 MLP 每步执行 6 次 MatMul；第一层输入不需要梯度，仍执行一次输入梯度矩阵乘法。固定单 CPU、单线程的较大 MLP 对照中，按梯度需求计算后耗时减少约 24%，所有参数梯度与连续更新核对通过。
2. **CUDA 路径存在大量细碎运算与主机决策。** 小 MLP 每步约 82 个 kernel，PyTorch 约 24 个。交叉熵前向分别为 34 个和 2 个；nnlearn 每步读取两个 GPU 布尔标量。标量只有两个字节，但复制 API 的主机侧成本明显高于设备实际搬运时间。
3. **CuPy 通用矩阵乘法封装值得优化。** 二维专用 `dot` 路径在两组 CUDA 对照中均表现出收益；其收益幅度受负载影响，不能承诺固定百分比。CPU 较大矩阵直接替换为 `dot` 曾变慢，不能全局替换。
4. **测量环境影响不可忽略。** 工作站存在其他 GPU 进程和较高 I/O 压力，部分采样出现明显离群值。已有倍数是具体配置下的观测，不能作为独占设备上的稳定结论，更不能推广为全库性能。

本轮增加诊断脚本、原始数据和调查报告；库的算子、自动微分及优化器实现保持基准版本。

## 2. 工作负载与环境

复用 [704 基准](../../examples/704_compat_training_performance_benchmark.py)，模型为 Linear → ReLU → Linear → CrossEntropyLoss，使用 SGD、学习率 0.1。两边共享合成输入、标签、参数和批次顺序；线性层权重按布局转置映射，输入与标签不要求梯度，`zero_grad(set_to_none=False)` 一致。

| 场景 | 样本数 | 输入维度 | 隐藏维度 | 类别数 | batch |
|---|---|---|---|---|---|
| small | 512 | 32 | 64 | 4 | 64 |
| wide | 1024 | 512 | 1024 | 64 | 256 |

- CPU：Intel Core i5-12490F，6 核、12 逻辑 CPU；主组统一 NumPy/OpenBLAS 与 PyTorch 为 6 线程。另测 1 线程，并将补充实验进程绑定到逻辑 CPU 10。
- GPU：RTX 3060，12 GiB；驱动 550.144.03。
- Python 3.12.8、NumPy 2.2.6、CuPy 13.6.0、PyTorch 2.6.0+cu124。CuPy runtime 为 11.8，系统 nvcc 为 11.5，PyTorch CUDA 为 12.4。
- 全部使用 float32；CuPy 与 PyTorch 的 TF32 均关闭。实际线程池、软件版本及环境变量见各组 `metadata.json`。
- 补充快照中，除本次测量进程外，另有 Python GPU 任务占用约 1 GiB 显存及编辑器图形上下文。CPU0 governor 为 powersave，系统 I/O PSI `some avg10` 约 99%，`full avg10` 约 58%～62%。这些是全局环境观察，不能直接证明某个训练区间被 I/O 阻塞。

本轮范围是这两个 MLP；未据此判断 CNN、Transformer、其他 dtype 或任意 Tensor API 的性能。

## 3. 时间口径

1. **正式基准**：沿用 704 的完整训练循环；数据、模型创建、预热、最终评估均不计入。CUDA 只在整段计时边界同步。下表正式基准为 100 步预热、1000 步训练、7 次重复的中位数，保留每轮原始值与离散度。
2. **阶段诊断**：另起运行统计清梯度、模型前向、损失、反向、优化器。主机时间包含调用内部的 CUDA 等待；CUDA event 区间包含主机提交间隙，不等同于纯 kernel 时间。事件和计时本身有额外成本。
3. **调用 profile**：分别使用墙钟和进程 CPU 时钟的 cProfile。独占时间与包含子调用的累计时间分开保存，不相加重复计费。两种 profile 都有明显插桩开销，不能用其总耗时计算框架性能倍数。
4. **CUDA 时间线**：使用 PyTorch 2.6 Kineto/CUPTI，并实际验证捕获了 CuPy 生成 kernel、cuBLAS、DtoH 复制和同步。按运行区间过滤，kernel/复制活动取区间并集，保留无法归因的活动计数。这里仅捕获本进程活动，间隙不代表整卡空闲。
5. **追加资源观测**：训练区间同时记录墙钟、进程和主线程 CPU 时间、上下文切换、轻/重缺页；系统快照放在计时区间外。进程 CPU 时间也可能包含 CUDA 驱动忙等，不能全部解释成 Python 计算。

这些口径遵循 [Python profiler 说明](https://docs.python.org/3.12/library/profile.html)、[进程 CPU 时钟定义](https://docs.python.org/3.12/library/time.html#time.process_time)、[CuPy 13.6 性能测量说明](https://docs.cupy.dev/en/v13.6.0/user_guide/performance.html) 和 [PyTorch 2.6 profiler](https://docs.pytorch.org/docs/2.6/profiler.html)。

## 4. 端到端时间

### 4.1 统一 6 线程的长采样

单位：ms/步。括号为 7 轮的最小值～最大值，最后一列为两个中位数之比。这是有其他工作站负载时的观察，不是独占硬件结果。

| 场景 | nnlearn | PyTorch | 耗时比 |
|---|---|---|---|
| small-cpu | 0.3843（0.3703～0.3892） | 0.1802（0.1701～0.2043） | 2.13 倍 |
| small-cuda | 2.0875（2.0502～2.1688） | 0.3120（0.2904～0.3290） | 6.69 倍 |
| wide-cpu | 5.4411（5.1115～6.1081） | 2.6489（2.0913～3.0130） | 2.05 倍 |
| wide-cuda | 3.4387（3.2474～3.6126） | 0.5360（0.5048～0.5528） | 6.42 倍 |

证据：[长采样配置](profiles/PLAN-068/baseline-long/metadata.json)及同目录 `*-baseline.json`。首轮 200 步结果仍保留在 `threads-6/`，没有用长采样覆盖。

### 4.2 单线程与固定亲和性的补充测量

未固定亲和性的单线程 CPU 基准中，小/较大 MLP 的耗时比分别为 2.21/1.56 倍，见 `threads-1/`。下面为绑定逻辑 CPU 10、单线程、每轮 300 步、7 轮的追加观察：

| 场景 | 框架 | 墙钟 ms/步 | 进程 CPU ms/步 | 主线程 CPU ms/步 |
|---|---|---|---|---|
| small-cpu | nnlearn | 0.4049 | 0.4049 | 0.4049 |
| small-cpu | pytorch | 0.1853 | 0.1853 | 0.1853 |
| small-cuda | nnlearn | 3.1995 | 3.1989 | 3.1934 |
| small-cuda | pytorch | 0.3536 | 0.3535 | 0.2141 |
| wide-cpu | nnlearn | 11.3959 | 11.3948 | 11.3715 |
| wide-cpu | pytorch | 7.0948 | 7.0946 | 7.0810 |
| wide-cuda | nnlearn | 3.4498 | 3.4497 | 3.4439 |
| wide-cuda | pytorch | 0.5283 | 0.5283 | 0.3014 |

这些追加训练区间的重缺页计数均为 0，墙钟与进程 CPU 时间接近，因此不能把差距全部归因于进程被操作系统挂起；CUDA 调用链中的忙等仍会计入 CPU 时间。固定亲和性并没有排除其他 GPU 任务，也会影响 PyTorch 的辅助线程调度，不应把这一组当作默认部署配置。

扩展 A/B 组中，较大 CPU MLP 原实现曾出现 110.17 ms/步的离群轮，而通常约 4～5 ms/步。该轮没有同时记录资源计数，**无法确定离群的具体原因**。原始值见 [A/B 数据](profiles/PLAN-068/ab-controls/wide-cpu-probes.json)，未删除该轮。

## 5. 阶段耗时

以下为首轮独立阶段诊断的主机时间中位数，单位 ms/步；各配置 200 步、5 轮。它包含 Python、底层库调用及调用内部等待。最后一列是该诊断运行的整体时间，不与上一节长采样混用；各列中位数之和也不必等于整体中位数。

| 场景 | 框架 | 清梯度 | 前向 | 损失 | 反向 | 更新 | 诊断整体 |
|---|---|---|---|---|---|---|---|
| small-cpu | nnlearn | 0.0021 | 0.0788 | 0.0706 | 0.2334 | 0.0110 | 0.4029 |
| small-cpu | pytorch | 0.0210 | 0.0348 | 0.0129 | 0.0833 | 0.0405 | 0.1994 |
| small-cuda | nnlearn | 0.0117 | 0.3212 | 0.6682 | 1.1076 | 0.1549 | 2.2954 |
| small-cuda | pytorch | 0.0422 | 0.0637 | 0.0228 | 0.1774 | 0.0682 | 0.4341 |
| wide-cpu | nnlearn | 0.1039 | 1.3827 | 0.2077 | 3.1728 | 0.4193 | 5.2978 |
| wide-cpu | pytorch | 0.0458 | 0.6679 | 0.0312 | 0.8358 | 0.1235 | 1.7212 |
| wide-cuda | nnlearn | 0.0158 | 0.3610 | 0.7321 | 1.2110 | 0.1691 | 2.5267 |
| wide-cuda | pytorch | 0.0485 | 0.0817 | 0.0281 | 0.2229 | 0.0717 | 0.5222 |

小 MLP CUDA 的反向与交叉熵是主要阶段；较大 CPU MLP 的前向和反向矩阵运算占比提升。对应 `*-phases.json` 还保存了 CUDA stream event 时间和每轮原始值，用于检查提交与等待区间，而非计算纯 GPU FLOPs 性能。

## 6. 已查明的具体路径

### 6.1 未请求的输入梯度仍被计算

[MatMul.backward](../../nnlearn/functional/matrix.py#L609) 无条件执行 `gx` 与 `gW` 两次矩阵乘法；外层 [反向引擎](../../nnlearn/tensor.py#L89) 随后才跳过不需要梯度的输入。本轮第一层输入 `requires_grad=False`，因此产生一次没有被使用的 `gx` 计算。

调用计数确认原实现每步 6 次 MatMul。二维按需求反向探针将其减为 5 次；两个输入都需要梯度时沿用原反向实现，非二维输入也沿用原实现。探针只在测量进程内替换并恢复函数，没有修改库。

固定逻辑 CPU 10、单线程，5 轮、每轮 200 步的结果：

| 场景 | 原实现 ms/步 | 按需求反向 ms/步 | 中位耗时减少 |
|---|---|---|---|
| small-cpu | 0.4112（0.3827～0.5257） | 0.3876（0.3764～0.3988） | 5.75% |
| wide-cpu | 11.6177（11.3620～11.8385） | 8.7909（8.6323～9.2054） | 24.33% |

较大 MLP 的两组范围不重叠，支持优先修复这条路径；小 MLP 的幅度较小，需保留对噪声的判断。这里的候选为二维按需求反向实现，不应把其全部差值严格等同于单次 GEMM 的设备执行时间。

### 6.2 Python 调度与数组操作

两个 MLP 的结构相同，首轮计数均为每步约 26 次 Function 调用、29 次 Tensor 构造、5 次 clone 和接近 6 次 Add；每 100 步的 cProfile 记录约 23.3 万次 Python/C 调用。

小 MLP CPU 的墙钟 cProfile 中，`Function.__call__` 独占 19.53 ms/100 步，累计 72.03 ms/100 步；类型判断、`typing.__instancecheck__`、`np.issubdtype` 也出现在热点列表。对应 [Function 输入归一化](../../nnlearn/function.py#L16) 每次执行浮点上下文推导，即使本次输入没有原生 Python 数字。

较大 MLP CPU 的同组 profile 中，`MatMul.forward` 独占约 251.23 ms/100 步，占据主要调用耗时，参数更新、数组加法和复制也增加。进程 CPU 时钟 profile 得到相同的规模差异：较大 MLP 以矩阵运算为主要热点，小 MLP 更受框架与类型检查开销影响。**这些 profile 时间含插桩成本，不能直接换算为无插桩运行的时间占比。**

已有 clone 用于保障梯度独立可写及高阶依赖，见 [梯度累积](../../nnlearn/tensor.py#L102)；不能通过直接删除 clone 来宣称完成安全优化。

### 6.3 CUDA kernel 数量

首轮 CUPTI 窗口为每框架 20 步。以下为小 MLP 平均每步 kernel 数量：

| 框架 | 清梯度 | 前向 | 损失 | 反向 | 更新 | 总数 |
|---|---|---|---|---|---|---|
| nnlearn | 0.0 | 5.0 | 34.0 | 31.0 | 12.0 | 82.0 |
| pytorch | 3.8 | 3.0 | 2.0 | 13.8 | 1.0 | 23.6 |

第一步尚无保留梯度，所以 PyTorch 的清零/累积平均数不是整数；nnlearn 的清零还使用 GPU memset，不在 kernel 列内。较大 MLP 总数分别为 85.0 和 25.6 个/步。阶段按 CPU launch 所在注释区间关联，摘要保留未归因数量；本次全部 kernel 均完成归因。

`CrossEntropyLoss` 使用 [多步指数、规约、索引与掩码](../../nnlearn/nn/loss.py#L155)；前向为 34 个 kernel。SGD 对每个参数分别处理，4 个参数共 12 个更新 kernel，对照 PyTorch 该配置为 1 个更新 kernel。这些结构差异使融合算子具有明确调查价值，但 kernel 数减少不等于端到端耗时同比减少。

### 6.4 标量读回的成本位于主机调用链

读回来源为 [非法标签检查](../../nnlearn/nn/loss.py#L160) 与 [均值分母非零判断](../../nnlearn/nn/loss.py#L231)，经 [as_numpy](../../nnlearn/backend.py#L99) 和 `_array_scalar_bool` 返回 Python bool。每步 2 次，每次 1 字节。固定亲和性补充 trace 的 20 步统计如下，时间已经除以步数：

| 场景 | 主机 memcpy API ms/步 | 设备实际复制 ms/步 | 显式 stream sync API ms/步 | DtoH 字节/步 |
|---|---|---|---|---|
| small-cuda | 0.5293 | 0.0023 | 0.0107 | 2 |
| wide-cuda | 1.6783 | 0.0028 | 0.0131 | 2 |

主机复制 API 明显长于设备搬运，说明不能只根据两个字节的带宽成本判断影响，也不能只搜索名称含 Synchronize 的 API。主机 API 时间包含等待、驱动调度等因素，本轮尚不能进一步拆分本进程排队、驱动资源竞争和其他 GPU 工作各占多少。

[CuPy 13.6 `asnumpy`](https://docs.cupy.dev/en/v13.6.0/reference/generated/cupy.asnumpy.html) 默认对主机阻塞；[NVIDIA 同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html) 也说明 Async 名称不保证主机始终不阻塞。立即消费布尔值时，简单改为异步复制不能安全消除依赖。

首轮小/较大 MLP 的本进程 kernel 总执行时间分别约 0.153/0.493 ms/步；它们属于带 CUPTI 的独立运行，不能从无插桩端到端时间中直接相减。时间线空隙也不能全部算作可优化的 Python 开销。

### 6.5 二维矩阵乘法与标量缓存对照

[MatMul.forward](../../nnlearn/functional/matrix.py#L602) 使用 `xp.matmul`。当前 CuPy 13.6 经过通用 gufunc 包装，其转置/参数整理出现在 profile 热点。探针只在两个输入均为二维时改用 `xp.dot`，其余保持原路径。

| 首轮场景 | 原实现 ms/步 | 二维 dot ms/步 | dot 耗时减少 | 仅缓存系数耗时减少 |
|---|---|---|---|---|
| small-cuda | 2.3031 | 1.9297 | 16.21% | 2.69% |
| wide-cuda | 2.4309 | 2.0481 | 15.75% | 2.78% |

后续 7 轮扩展 A/B 中，二维 dot 的 CUDA 中位收益约为 small 10%、wide 27%，但各轮明显波动；因此这里只确认它是值得实施并复核的候选，不承诺固定收益。CPU 较大 MLP 的首轮 dot 反而增加约 11% 耗时，后续 6 线程组也出现回退，证明需要区分后端与形状。

[SGD 系数构造](../../nnlearn/optim/optimizer.py#L112) 每步对四个参数调用 `asarray`。缓存后计数由每步 5 次主机来源 `asarray` 请求降为 1 次，剩余一次来自 ReLU 的零常量。这里统计的是 API 请求，**不等于 5 次物理 HtoD 复制**；CUPTI 才是实际复制数量的依据。缓存的端到端收益小且存在不一致，优先级低于额外 GEMM 与交叉熵。组合探针的收益不能把各单项百分比相加。

## 7. 建议的后续优化顺序

| 优先级 | 目标 | 已有证据 | 实施与验收要点 |
|---|---|---|---|
| P1 | MatMul 按梯度需求计算 | 6 次降为 5 次；较大单线程 CPU 时间减少约 24% | 保留两边梯度、高阶导、向量与批次广播语义；分别覆盖单边/双边 requires_grad |
| P1 | CUDA 二维矩阵乘法专用路径 | CuPy 包装为调用热点；多组 dot 探针有收益 | 仅在支持的维度/后端启用；CPU 保留原路径；覆盖连续/转置/切片、空维度及 dtype |
| P2 | 交叉熵减少跨设备决策并融合计算 | 前向 34 kernel、2 次标量读回；主机复制链路成本明显 | 先将分母分支留在 GPU，再评估融合前后向；保留非法标签 ValueError、ignore_index、weight、label smoothing、reduction、半精度与高阶导 |
| P2 | 小 MLP 的输入/类型归一化 | Function、协议 isinstance、dtype 推导为热点 | 仅在需要原生 Python 常量上下文时推导；以 dtype 与自动微分回归约束语义 |
| P3 | SGD 融合与系数复用 | 更新 kernel 数较多；纯缓存收益较小 | 先测一参数一 kernel 的收益，再考虑多参数融合；保持参数、状态 dtype 与存储所有权 |

生产优化应逐项提交并用相同配置交替测量，保留原始样本；在更空闲的 GPU 环境复核端到端收益。当前证据没有要求先建设完全独立的 CUDA 内存/数组运行时，也没有证明重写 GEMM 能解决主要差距。

## 8. 验证与证据边界

- 已核对 77 个原始 JSON、164 个统计摘要，74 组三步前向/梯度/更新对照全部通过；原始 trace 共 8 份，字节数、SHA-256 与摘要一致。见 [audit.json](profiles/PLAN-068/audit.json)。
- 全部记录中的最大绝对差：logits `1.7166138e-05`、loss `2.8610229e-06`、参数梯度 `3.1664968e-07`、更新后参数 `5.9604645e-08`。
- 最终脚本对两个 MLP、CPU/CUDA 的全部基础诊断模式与 6 种探针完成烟测，并明确拒绝非有限的前向、梯度与参数。烟测步数很短，仅证明流程，不用其耗时讨论性能。
- 无 CuPy 的系统 Python 已完成 CPU-only 基准、计数与 cProfile 烟测。原 704 dtype 回归 `test_benchmark_models_and_training_use_float32` 通过；Ruff 通过。
- 验证了探针、计数钩子在异常退出时恢复原函数；区间并集通过重叠区间核对。
- 没有使用 profiler 放大后的总耗时计算性能倍数，没有删除离群值，也没有将本进程 GPU 活动比例说成整卡利用率。
- 未完成独占 GPU 的发布级性能验收；尚未把主机复制 API 的内部等待进一步分解；未对 CNN/Transformer 或全部算子进行性能普查。本轮确认的是两个 MLP 的具体路径与优化候选。

## 9. 复现命令

以下命令在仓库根目录运行，使用已经通过 CUDA 验收的解释器；其他机器可替换 `--python` 路径。性能实验应串行运行。

```bash
# 首轮完整诊断：200 步、5 轮，两个 MLP 与 CPU/CUDA
uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python python scripts/profile_training.py --steps 200 --warmup 50 --repeats 5 --profile-steps 100 --trace-steps 20 --threads 6 --output-dir /tmp/nnlearn-profile-reproduce/initial --trace-dir /tmp/nnlearn-profile-reproduce/traces

# 无插桩的长采样
uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python python scripts/profile_training.py --modes baseline --steps 1000 --warmup 100 --repeats 7 --threads 6 --output-dir /tmp/nnlearn-profile-reproduce/long

# 单线程、固定亲和性，记录资源与进程 CPU 时钟 profile
# CPU 10 为本机逻辑编号；其他机器需选择实际允许的逻辑 CPU。
taskset -c 10 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python python scripts/profile_training.py --modes observed cprofile --profile-clock process --steps 300 --warmup 50 --repeats 7 --profile-steps 100 --threads 1 --output-dir /tmp/nnlearn-profile-reproduce/observed

# 单线程 CPU 的候选路径对照
taskset -c 10 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python python scripts/profile_training.py --devices cpu --modes probes --steps 200 --warmup 50 --repeats 5 --profile-steps 30 --threads 1 --output-dir /tmp/nnlearn-profile-reproduce/cpu-probes

# 同亲和性条件的 CUDA 时间线与调用计数
taskset -c 10 uv run --no-project --python /tmp/nnlearn-cuda-validation-venv/bin/python python scripts/profile_training.py --devices cuda --modes trace counts --trace-steps 20 --profile-steps 30 --warmup 50 --threads 1 --output-dir /tmp/nnlearn-profile-reproduce/cuda-trace --trace-dir /tmp/nnlearn-profile-reproduce/affinity-traces
```

每组 `metadata.json` 记录当时的命令、库提交和脚本 SHA-256。脚本在本轮开发中逐步增加了资源观测、梯度需求探针和更严格的有限值检查；旧记录保持当时内容，不补造缺失字段。核心库代码始终为上方基准提交。用最终脚本重跑首轮命令会额外输出新增探针字段，结果应按其各自配置与运行环境解读。

完整 Chrome trace 约 13 MiB，保存在本机 `/tmp/nnlearn-profile-068-traces/` 与 `/tmp/nnlearn-profile-068-affinity-traces/`，可以导入 Perfetto 查看。仓库提交时间线摘要、计数、校验哈希和重建命令，不提交这些大型 trace。

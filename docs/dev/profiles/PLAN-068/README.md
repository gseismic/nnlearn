# PLAN-068 原始测量数据

调查报告：[训练性能 profile](../../20261026-2127-PRE-a924b1f-training-performance.md)。

所有 JSON 为实际运行结果，时间单位见各字段；`metadata.json` 记录配置、版本、线程池和当时脚本哈希。各轮中位数不能跨组拼成优化前后对照。所有文件均保留，包含有明显负载波动的轮次。

| 目录 | 内容 |
|---|---|
| `threads-6` | 首轮：两个 MLP、CPU/CUDA，200 步、50 步预热、5 轮；阶段、墙钟 cProfile、100 步计数、20 步 CUPTI、早期 4 种探针 |
| `threads-1` | 单线程 CPU：500 步、5 轮及墙钟 cProfile |
| `baseline-long` | 无插桩长采样：1000 步、100 步预热、7 轮、6 线程 |
| `ab-controls` | 扩展 6 种探针：300 步、7 轮、6 线程；包含较大 CPU 的 110 ms/步离群轮 |
| `affinity-10-threads-1` | CPU 10 亲和性、1 线程：300 步、7 轮，墙钟/进程/线程 CPU、缺页、切换、系统快照及进程时钟 cProfile |
| `affinity-10-cpu-probes` | 固定亲和性单线程 CPU 的 6 种探针：200 步、5 轮 |
| `affinity-10-trace` | 固定亲和性、1 线程的 CUDA 20 步 CUPTI 与 30 步调用计数 |
| `audit.json` | 原始统计复算、三步数值对照汇总、8 份 trace 哈希与诊断钩子恢复检查 |

字段说明：`baseline` 为未加 profiler 的训练整体；`phases` 为另起诊断运行的主机与 stream 区间；`cprofile` 含插桩开销；`trace` 仅统计被 profile 进程的 GPU 活动，间隙不等于整卡空闲；`counts` 的主机来源 asarray 请求不等于物理 HtoD 复制。

探针 `original` 为原实现；`dot_2d` 只替换二维矩阵乘法；`scalar_cache` 缓存 GPU 更新系数；`combined` 为前两项；`required_grads` 对单边无需梯度的二维矩阵使用按需求反向；`all_three` 为三项组合。所有探针仅在调查进程内生效并恢复，不是库实现变更。

测量工作站存在其他 GPU 进程和 I/O 压力。旧数据采样早于系统快照功能，不能据后来的快照倒推每一轮离群的原因；详细限制与复现命令见报告。

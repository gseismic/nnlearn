"""复用 704 基准调查训练耗时；诊断开销与正式基准分开保存。"""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import contextmanager
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PHASES = ("zero_grad", "forward", "loss", "backward", "optimizer")
WORKLOADS = {
    "small": dict(SAMPLES=512, FEATURES=32, HIDDEN=64, CLASSES=4, BATCH_SIZE=64),
    "wide": dict(SAMPLES=1024, FEATURES=512, HIDDEN=1024, CLASSES=64, BATCH_SIZE=256),
}


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--workloads", nargs="+", choices=tuple(WORKLOADS), default=list(WORKLOADS))
    parser.add_argument("--modes", nargs="+", choices=("baseline", "phases", "cprofile", "trace", "counts", "probes", "observed"),
                        default=["baseline", "phases", "cprofile", "trace", "counts", "probes"])
    parser.add_argument("--steps", type=_positive, default=300)
    parser.add_argument("--warmup", type=_positive, default=50)
    parser.add_argument("--repeats", type=_positive, default=5)
    parser.add_argument("--profile-steps", type=_positive, default=30)
    parser.add_argument("--profile-clock", choices=("wall", "process"), default="wall")
    parser.add_argument("--trace-steps", type=_positive, default=10)
    parser.add_argument("--threads", type=_positive, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/nnlearn-training-profile-traces"))
    return parser.parse_args()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _summary(values):
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "raw": values,
    }


def _command_output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    except OSError as error:
        return str(error)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def _load_benchmark(workload):
    source = ROOT / "examples/704_compat_training_performance_benchmark.py"
    spec = importlib.util.spec_from_file_location("_training_benchmark_" + workload, source)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    for key, value in WORKLOADS[workload].items():
        setattr(benchmark, key, value)
    return benchmark


def _system_snapshot():
    result = {"created_at": datetime.now(timezone.utc).isoformat(),
              "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None}
    for name in ("cpu", "memory", "io"):
        path = Path("/proc/pressure") / name
        if path.exists():
            result[name + "_pressure"] = path.read_text().strip()
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if path.exists():
        result["cpu0_governor"] = path.read_text().strip()
    result["gpu_processes"] = _command_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"])
    result["gpu_state"] = _command_output(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.current.sm,temperature.gpu,power.draw", "--format=csv,noheader"])
    result["measurement_pid"] = os.getpid()
    return result


def _metadata(args, benchmark, cp):
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = "未安装 threadpoolctl；以进程启动前设置的线程环境变量为准"
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "working_tree": _command_output(["git", "status", "--short"]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": sys.argv,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": benchmark.np.__version__,
        "pytorch": benchmark.torch.__version__,
        "pytorch_cuda": benchmark.torch.version.cuda,
        "pytorch_cpu_threads": benchmark.torch.get_num_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "system_snapshot": _system_snapshot(),
        "thread_pools": pools,
        "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUPY_TF32", "CUPY_ACCELERATORS", "CUDA_VISIBLE_DEVICES",
        )},
        "tf32": {"torch_matmul": benchmark.torch.backends.cuda.matmul.allow_tf32,
                 "torch_cudnn": benchmark.torch.backends.cudnn.allow_tf32},
        "measurement_notes": [
            "baseline 为未加 profiler 的整段墙钟时间，CUDA 只在整段计时边界同步。",
            "phases 为独立诊断运行；host 时间包含调用内部同步，stream event 时间包含等待主机提交的空隙。",
            "cProfile 和 CUPTI trace 都会增加开销，其时间不能用于性能倍数。",
            "trace 的 GPU active 为 kernel/拷贝区间的并集；GPU 空隙不等于可以完全消除的开销。",
            "trace 只包含当前进程的 CUDA 活动，其空隙不能当作整卡空闲；其他 GPU 进程见系统快照。",
            "临时 probes 仅用于验证候选原因，进入和退出时恢复库实现；不等于已交付优化。",
        ],
    }
    if cp is not None:
        result["cuda"] = {
            "cupy": cp.__version__,
            "cupy_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "gpu": benchmark.torch.cuda.get_device_name(),
            "gpu_details": _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
            "nvcc": _command_output(["nvcc", "--version"]),
        }
    return result


def _numpy(benchmark, framework, value):
    if framework == "pytorch":
        return value.detach().cpu().numpy()
    return value.numpy()


def _verify(benchmark, device, state, batches):
    """前向、全部参数梯度及三次更新均对照，计时前排除工作负载不一致。"""
    parts = {framework: benchmark._build_training_parts(framework, device, state)
             for framework in ("nnlearn", "pytorch")}
    maxima = dict(logits=0.0, loss=0.0, gradient=0.0, parameter=0.0)
    for step in range(3):
        values = {}
        for framework, (model, optimizer, criterion) in parts.items():
            x, y = batches[framework][step % len(batches[framework])]
            assert not x.requires_grad and not y.requires_grad
            optimizer.zero_grad(set_to_none=False)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            parameters = dict(model.named_parameters())
            assert len(parameters) == 4
            for parameter in parameters.values():
                assert parameter.grad is not None
                assert str(parameter.dtype).endswith("float32")
                assert parameter.grad.dtype == parameter.dtype
                assert str(parameter.device).startswith(device)
            values[framework] = (output, loss, parameters)
        for category, index in (("logits", 0), ("loss", 1)):
            actual = _numpy(benchmark, "nnlearn", values["nnlearn"][index])
            expected = _numpy(benchmark, "pytorch", values["pytorch"][index])
            assert benchmark.np.isfinite(actual).all() and benchmark.np.isfinite(expected).all()
            benchmark.np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=3e-5, equal_nan=False)
            maxima[category] = max(maxima[category], float(benchmark.np.max(benchmark.np.abs(actual - expected))))
        for name, parameter in values["nnlearn"][2].items():
            reference = values["pytorch"][2][name]
            actual = parameter.grad.numpy()
            expected = reference.grad.detach().cpu().numpy()
            if name.endswith("weight"):
                expected = expected.T
            assert benchmark.np.isfinite(actual).all() and benchmark.np.isfinite(expected).all()
            benchmark.np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=3e-5, equal_nan=False)
            maxima["gradient"] = max(maxima["gradient"], float(benchmark.np.max(benchmark.np.abs(actual - expected))))
        for _, optimizer, _ in parts.values():
            optimizer.step()
        for name, parameter in values["nnlearn"][2].items():
            actual = parameter.numpy()
            expected = values["pytorch"][2][name].detach().cpu().numpy()
            if name.endswith("weight"):
                expected = expected.T
            assert benchmark.np.isfinite(actual).all() and benchmark.np.isfinite(expected).all()
            benchmark.np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=3e-5, equal_nan=False)
            maxima["parameter"] = max(maxima["parameter"], float(benchmark.np.max(benchmark.np.abs(actual - expected))))
    return {"steps": 3, "max_absolute_errors": maxima, "passed": True}


def _baseline(args, benchmark, device, state, batches, cp):
    samples = {framework: [] for framework in batches}
    for repeat in range(args.repeats):
        order = ("nnlearn", "pytorch") if repeat % 2 == 0 else ("pytorch", "nnlearn")
        for framework in order:
            result = benchmark._run_once(framework, device, state, batches[framework],
                                         *benchmark._make_dataset(), args.steps, args.warmup, cp)
            samples[framework].append(result)
    result = {
        "steps": args.steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "wall_ms_per_step": {framework: _summary([run["seconds"] * 1000 / args.steps for run in runs])
                             for framework, runs in samples.items()},
        "runs": samples,
    }
    result["nnlearn_over_pytorch"] = result["wall_ms_per_step"]["nnlearn"]["median"] / result["wall_ms_per_step"]["pytorch"]["median"]
    return result


def _prepared(args, benchmark, framework, device, state, batches, cp):
    model, optimizer, criterion = benchmark._build_training_parts(framework, device, state)
    benchmark._train_steps(model, optimizer, criterion, batches, args.warmup)
    benchmark._synchronize(framework, device, cp)
    parts = benchmark._build_training_parts(framework, device, state)
    benchmark._synchronize(framework, device, cp)
    return parts


def _operations(model, optimizer, criterion, batch):
    """仅诊断运行使用闭包分段；正式 baseline 直接运行原始训练循环。"""
    x, y = batch
    values = {}

    def forward():
        values["output"] = model(x)

    def loss():
        values["loss"] = criterion(values["output"], y)

    return (lambda: optimizer.zero_grad(set_to_none=False), forward, loss,
            lambda: values["loss"].backward(), optimizer.step)


def _phases(args, benchmark, framework, device, state, batches, cp):
    samples = []
    for _ in range(args.repeats):
        model, optimizer, criterion = _prepared(args, benchmark, framework, device, state, batches, cp)
        events = None
        if device == "cuda":
            factory = cp.cuda.Event if framework == "nnlearn" else lambda: benchmark.torch.cuda.Event(enable_timing=True)
            events = [[factory() for _ in range(len(PHASES) + 1)] for _ in range(args.steps)]
            # 预先创建底层 event，避免把第一次创建成本混入阶段测量。
            for row in events:
                for event in row:
                    event.record()
            benchmark._synchronize(framework, device, cp)
        host = Counter()
        start = time.perf_counter()
        for step in range(args.steps):
            operations = _operations(model, optimizer, criterion, batches[step % len(batches)])
            if events is not None:
                events[step][0].record()
            for index, (name, operation) in enumerate(zip(PHASES, operations)):
                before = time.perf_counter()
                operation()
                host[name] += time.perf_counter() - before
                if events is not None:
                    events[step][index + 1].record()
        benchmark._synchronize(framework, device, cp)
        sample = {"wall_ms_per_step": (time.perf_counter() - start) * 1000 / args.steps,
                  "host_ms_per_step": {name: host[name] * 1000 / args.steps for name in PHASES}}
        if events is not None:
            elapsed = cp.cuda.get_elapsed_time if framework == "nnlearn" else lambda first, last: first.elapsed_time(last)
            sample["stream_ms_per_step"] = {
                name: sum(elapsed(row[index], row[index + 1]) for row in events) / args.steps
                for index, name in enumerate(PHASES)
            }
        samples.append(sample)
    return {
        "steps": args.steps,
        "runs": samples,
        "wall_ms_per_step": _summary([sample["wall_ms_per_step"] for sample in samples]),
        "host_ms_per_step": {name: _summary([sample["host_ms_per_step"][name] for sample in samples]) for name in PHASES},
        "stream_ms_per_step": ({name: _summary([sample["stream_ms_per_step"][name] for sample in samples]) for name in PHASES}
                               if device == "cuda" else None),
    }


def _function_location(filename, line, name):
    try:
        filename = str(Path(filename).relative_to(ROOT))
    except ValueError:
        pass
    return f"{filename}:{line}:{name}"


def _cprofile(args, benchmark, framework, device, state, batches, cp):
    parts = _prepared(args, benchmark, framework, device, state, batches, cp)
    profiler = cProfile.Profile(timer=time.process_time) if args.profile_clock == "process" else cProfile.Profile()
    profiler.enable()
    benchmark._train_steps(*parts, batches, args.profile_steps)
    benchmark._synchronize(framework, device, cp)
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append(dict(function=_function_location(filename, line, name), calls=calls, primitive_calls=primitive,
                         self_ms=own * 1000, cumulative_ms=cumulative * 1000))
    return {"steps": args.profile_steps, "clock": args.profile_clock, "profiled_total_ms": stats.total_tt * 1000,
            "total_calls": stats.total_calls, "functions": sorted(rows, key=lambda row: -row["self_ms"])}


def _observed(args, benchmark, device, state, batches, cp):
    """同一训练区间并列记录墙钟、进程 CPU、线程 CPU 与资源变化。"""
    import resource

    samples = {framework: [] for framework in batches}
    for repeat in range(args.repeats):
        order = ("nnlearn", "pytorch") if repeat % 2 == 0 else ("pytorch", "nnlearn")
        for framework in order:
            parts = _prepared(args, benchmark, framework, device, state, batches[framework], cp)
            before_system = _system_snapshot()
            before_usage = resource.getrusage(resource.RUSAGE_SELF)
            before_process, before_thread = time.process_time(), time.thread_time()
            start = time.perf_counter()
            benchmark._train_steps(*parts, batches[framework], args.steps)
            benchmark._synchronize(framework, device, cp)
            wall = time.perf_counter() - start
            process, thread = time.process_time() - before_process, time.thread_time() - before_thread
            after_usage = resource.getrusage(resource.RUSAGE_SELF)
            resources = {key: getattr(after_usage, key) - getattr(before_usage, key)
                         for key in ("ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")}
            samples[framework].append(dict(
                wall_ms_per_step=wall * 1000 / args.steps,
                process_cpu_ms_per_step=process * 1000 / args.steps,
                thread_cpu_ms_per_step=thread * 1000 / args.steps,
                resource_delta=resources, before=before_system, after=_system_snapshot(),
            ))
    return {
        "steps": args.steps, "repeats": args.repeats, "runs": samples,
        "statistics": {framework: {key: _summary([sample[key] for sample in runs])
                                    for key in ("wall_ms_per_step", "process_cpu_ms_per_step", "thread_cpu_ms_per_step")}
                       for framework, runs in samples.items()},
    }


def _interval_union(events):
    ranges = sorted((event["ts"], event["ts"] + event["dur"]) for event in events)
    total, end = 0.0, float("-inf")
    for start, stop in ranges:
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _trace_summary(path, steps):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    root = next(event for event in events if event.get("cat") == "user_annotation" and event.get("name") == "profile_training")
    start, stop = root["ts"], root["ts"] + root["dur"]
    selected = [event for event in events if event.get("ph") == "X" and event.get("dur", 0) >= 0
                and event["ts"] >= start and event["ts"] + event["dur"] <= stop]
    kernels = [event for event in selected if event.get("cat") == "kernel"]
    copies = [event for event in selected if event.get("cat") in ("gpu_memcpy", "gpu_memset")]
    runtime = [event for event in selected if event.get("cat") in ("cuda_runtime", "cuda_driver")]
    groups = defaultdict(lambda: dict(count=0, total_us=0.0, bytes=0))
    for event in kernels + copies + runtime:
        item = groups[(event["cat"], event["name"])]
        item["count"] += 1
        item["total_us"] += event["dur"]
        item["bytes"] += event.get("args", {}).get("bytes", 0)
    phases = sorted((event for event in selected if event.get("cat") == "user_annotation" and event["name"].startswith("phase/")),
                    key=lambda event: event["ts"])
    phase_starts = [event["ts"] for event in phases]
    correlations = {}
    for event in runtime:
        index = bisect_right(phase_starts, event["ts"]) - 1
        if index >= 0 and event["ts"] <= phases[index]["ts"] + phases[index]["dur"]:
            correlations[event.get("args", {}).get("correlation")] = phases[index]["name"]
    kernel_phases = defaultdict(lambda: dict(count=0, total_us=0.0))
    for event in kernels:
        phase = correlations.get(event.get("args", {}).get("correlation"), "unattributed")
        kernel_phases[phase]["count"] += 1
        kernel_phases[phase]["total_us"] += event["dur"]
    active_us = _interval_union(kernels + copies)
    return {
        "activity_scope": "当前被 profile 进程的 CUDA 活动；不包含其他进程，不能代替整卡利用率",
        "steps": steps,
        "raw_trace": str(path.resolve()),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trace_bytes": path.stat().st_size,
        "profiled_window_ms": root["dur"] / 1000,
        "kernel_count": len(kernels),
        "kernel_sum_ms": sum(event["dur"] for event in kernels) / 1000,
        "kernel_union_ms": _interval_union(kernels) / 1000,
        "copy_count": len(copies),
        "copy_union_ms": _interval_union(copies) / 1000,
        "gpu_active_union_ms": active_us / 1000,
        "gpu_active_fraction_of_profiled_window": active_us / root["dur"],
        "gpu_gap_ms": (root["dur"] - active_us) / 1000,
        "kernel_launch_phase_attribution": dict(kernel_phases),
        "activities": sorted([dict(category=category, name=name, **values) for (category, name), values in groups.items()],
                             key=lambda row: (row["category"], -row["total_us"])),
    }


def _trace(args, benchmark, framework, state, batches, cp, name):
    torch = benchmark.torch
    model, optimizer, criterion = _prepared(args, benchmark, framework, "cuda", state, batches, cp)
    # 先使 profiler 的自身初始化完成，再开始待分析区间。
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]):
        benchmark._train_steps(model, optimizer, criterion, batches, 1)
        benchmark._synchronize(framework, "cuda", cp)
    model, optimizer, criterion = _prepared(args, benchmark, framework, "cuda", state, batches, cp)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        with torch.profiler.record_function("profile_training"):
            for step in range(args.trace_steps):
                operations = _operations(model, optimizer, criterion, batches[step % len(batches)])
                for phase, operation in zip(PHASES, operations):
                    with torch.profiler.record_function("phase/" + phase):
                        operation()
            benchmark._synchronize(framework, "cuda", cp)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    path = args.trace_dir / (name + "-" + framework + ".json")
    profiler.export_chrome_trace(str(path))
    result = _trace_summary(path, args.trace_steps)
    if result["kernel_count"] == 0:
        raise RuntimeError("时间线没有捕获 CUDA kernel，不能据此推断 GPU 开销")
    return result


def _counts(args, benchmark, device, state, batches, cp):
    from nnlearn import backend
    from nnlearn.function import Function
    from nnlearn.tensor import Tensor

    parts = _prepared(args, benchmark, "nnlearn", device, state, batches, cp)
    functions, host_reads, conversions, matmuls = Counter(), Counter(), Counter(), Counter()
    tensor_count = 0
    original_call, original_init, original_read = Function.__call__, Tensor.__init__, backend.as_numpy
    original_asarray = cp.asarray if cp is not None else None

    def call(function, *inputs):
        functions[type(function).__name__] += 1
        if type(function).__name__ == "MatMul":
            key = str([(tuple(value.shape), value.requires_grad) for value in inputs])
            matmuls[key] += 1
        return original_call(function, *inputs)

    def initialize(tensor, *values, **kwargs):
        nonlocal tensor_count
        tensor_count += 1
        return original_init(tensor, *values, **kwargs)

    def read(value):
        if backend.is_cuda_array(value):
            frame = sys._getframe(1)
            location = _function_location(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
            host_reads[location] += 1
        return original_read(value)

    def asarray(value, *values, **kwargs):
        if not isinstance(value, cp.ndarray):
            frame = sys._getframe(1)
            source = "numpy_array" if isinstance(value, benchmark.np.ndarray) else type(value).__name__
            conversions[source + " " + _function_location(frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)] += 1
        return original_asarray(value, *values, **kwargs)

    try:
        Function.__call__, Tensor.__init__, backend.as_numpy = call, initialize, read
        if cp is not None:
            cp.asarray = asarray
        benchmark._train_steps(*parts, batches, args.profile_steps)
        benchmark._synchronize("nnlearn", device, cp)
    finally:
        Function.__call__, Tensor.__init__, backend.as_numpy = original_call, original_init, original_read
        if cp is not None:
            cp.asarray = original_asarray
    return dict(steps=args.profile_steps, tensor_constructions=tensor_count, functions=dict(functions),
                gpu_host_reads=dict(host_reads), host_source_asarray_requests=dict(conversions), matmul_inputs=dict(matmuls))


@contextmanager
def _probe(variant):
    from nnlearn import backend
    from nnlearn.functional.matrix import MatMul
    from nnlearn.optim.optimizer import Optimizer

    forward, backward, scalar = MatMul.forward, MatMul.backward, Optimizer._scalar_like
    cache = {}

    def dot(function, x, weight):
        if x.ndim == 2 and weight.ndim == 2:
            function.x_shape, function.W_shape = x.shape, weight.shape
            return backend.get_array_module(x, weight).dot(x, weight)
        return forward(function, x, weight)

    def cached(optimizer, value, parameter):
        if not backend.is_cuda_array(parameter.data) or type(value) not in (float, int):
            return scalar(optimizer, value, parameter)
        key = (value, parameter.dtype.str, parameter.data.device.id)
        if key not in cache:
            cache[key] = scalar(optimizer, value, parameter)
        return cache[key]

    def required_gradients(function, gy):
        from nnlearn.functional.matrix import matmul, sum_to

        x, weight = function.inputs
        if x.ndim != 2 or weight.ndim != 2 or (x.requires_grad and weight.requires_grad):
            return backward(function, gy)
        gx = sum_to(matmul(gy, weight.transpose(-1, -2)), x.shape) if x.requires_grad else None
        gweight = sum_to(matmul(x.transpose(-1, -2), gy), weight.shape) if weight.requires_grad else None
        return gx, gweight

    try:
        if variant in ("dot_2d", "combined", "all_three"):
            MatMul.forward = dot
        if variant in ("scalar_cache", "combined", "all_three"):
            Optimizer._scalar_like = cached
        if variant in ("required_grads", "all_three"):
            MatMul.backward = required_gradients
        yield
    finally:
        MatMul.forward, MatMul.backward, Optimizer._scalar_like = forward, backward, scalar


def _probes(args, benchmark, device, state, batches, cp):
    variants = ("original", "dot_2d", "scalar_cache", "combined", "required_grads", "all_three")
    validation = {}
    counts = {}
    samples = {variant: [] for variant in variants}
    for variant in variants:
        with _probe(variant):
            validation[variant] = _verify(benchmark, device, state, batches)
            counts[variant] = _counts(args, benchmark, device, state, batches["nnlearn"], cp)
    for repeat in range(args.repeats):
        order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
        for variant in order:
            with _probe(variant):
                sample = benchmark._run_once("nnlearn", device, state, batches["nnlearn"],
                                             *benchmark._make_dataset(), args.steps, args.warmup, cp)
            samples[variant].append(sample)
    return {variant: dict(validation=validation[variant], counts=counts[variant],
                          wall_ms_per_step=_summary([sample["seconds"] * 1000 / args.steps for sample in runs]), runs=runs)
            for variant, runs in samples.items()}


def main():
    args = _parse_args()
    # 在导入 NumPy/PyTorch 前统一子进程线程配置，不修改用户的外部环境。
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["CUPY_TF32"] = "0"
    sys.path.insert(0, str(ROOT))
    first = _load_benchmark(args.workloads[0])
    first.torch.set_num_threads(args.threads)
    first.torch.backends.cuda.matmul.allow_tf32 = False
    first.torch.backends.cudnn.allow_tf32 = False
    cp = None
    if "cuda" in args.devices:
        if not first.torch.cuda.is_available() or not first.nnlearn.cuda.is_available():
            raise SystemExit("CUDA 测量需要 PyTorch 与 CuPy 同时可用")
        import cupy as cp
    _write(args.output_dir / "metadata.json", _metadata(args, first, cp))
    for workload in args.workloads:
        benchmark = _load_benchmark(workload)
        features, labels = benchmark._make_dataset()
        state = benchmark._make_initial_state()
        for device in args.devices:
            name = workload + "-" + device
            batches = {framework: benchmark._make_batches(features, labels, device, framework) for framework in ("nnlearn", "pytorch")}
            validation = _verify(benchmark, device, state, batches)
            _write(args.output_dir / (name + "-validation.json"), dict(workload=WORKLOADS[workload], **validation))
            print(name, "数值核对通过", flush=True)
            for mode in args.modes:
                if mode == "trace" and device != "cuda":
                    continue
                if mode == "baseline":
                    result = _baseline(args, benchmark, device, state, batches, cp)
                    print(name, "耗时比", round(result["nnlearn_over_pytorch"], 3), flush=True)
                elif mode == "phases":
                    result = {framework: _phases(args, benchmark, framework, device, state, values, cp)
                              for framework, values in batches.items()}
                elif mode == "cprofile":
                    result = {framework: _cprofile(args, benchmark, framework, device, state, values, cp)
                              for framework, values in batches.items()}
                elif mode == "trace":
                    result = {framework: _trace(args, benchmark, framework, state, values, cp, name)
                              for framework, values in batches.items()}
                elif mode == "counts":
                    result = _counts(args, benchmark, device, state, batches["nnlearn"], cp)
                elif mode == "observed":
                    result = _observed(args, benchmark, device, state, batches, cp)
                else:
                    result = _probes(args, benchmark, device, state, batches, cp)
                _write(args.output_dir / (name + "-" + mode + ".json"), result)
                print(name, mode, "已保存", flush=True)


if __name__ == "__main__":
    main()

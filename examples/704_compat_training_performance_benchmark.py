"""在相同训练工作负载下比较 nnlearn 与 PyTorch 的速度和训练结果。"""

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as torch_nn
    import torch.optim as torch_optim
except ImportError as error:
    raise SystemExit(
        "此示例需要安装 PyTorch 作为对照实现。请参考 "
        "https://pytorch.org/get-started/locally/ 安装。"
    ) from error

import nnlearn
import nnlearn.nn as nnlearn_nn
import nnlearn.optim as nnlearn_optim


SAMPLES = 512
FEATURES = 32
HIDDEN = 64
CLASSES = 4
BATCH_SIZE = 64
LEARNING_RATE = 0.1
SEED = 20261001


def _make_dataset():
    """生成两边共用的、无需下载的合成分类数据。"""
    rng = np.random.default_rng(SEED)
    centers = rng.normal(0.0, 1.0, size=(CLASSES, FEATURES)).astype(np.float32)
    centers *= 2.5

    labels = np.arange(SAMPLES, dtype=np.int64) % CLASSES
    rng.shuffle(labels)
    features = centers[labels] + rng.normal(
        0.0, 0.65, size=(SAMPLES, FEATURES)
    ).astype(np.float32)
    return features.astype(np.float32), labels


def _make_initial_state():
    """创建两层 MLP 共用的初始参数，避免框架随机数生成器差异。"""
    rng = np.random.default_rng(SEED + 1)
    return {
        "0.weight": rng.normal(
            0.0, math.sqrt(2.0 / FEATURES), size=(HIDDEN, FEATURES)
        ).astype(np.float32),
        "0.bias": np.zeros(HIDDEN, dtype=np.float32),
        "2.weight": rng.normal(
            0.0, math.sqrt(2.0 / HIDDEN), size=(CLASSES, HIDDEN)
        ).astype(np.float32),
        "2.bias": np.zeros(CLASSES, dtype=np.float32),
    }


def _make_batches(features, labels, device, framework):
    """预先建立相同的批次，避免把数据生成和切分时间算进训练耗时。"""
    rng = np.random.default_rng(SEED + 2)
    indices = rng.permutation(len(labels))
    batches = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]
        batch_x = features[batch_indices]
        batch_y = labels[batch_indices]
        if framework == "pytorch":
            x = torch.tensor(batch_x, dtype=torch.float32, device=device)
            y = torch.tensor(batch_y, dtype=torch.int64, device=device)
        else:
            x = nnlearn.tensor(
                batch_x, dtype=nnlearn.float32, device=device
            )
            y = nnlearn.tensor(
                batch_y, dtype=nnlearn.int64, device=device
            )
        batches.append((x, y))
    return batches


def _make_model(framework, device, initial_state):
    """使用同一结构与初始参数创建对应框架的 MLP。"""
    if framework == "pytorch":
        model = torch_nn.Sequential(
            torch_nn.Linear(FEATURES, HIDDEN),
            torch_nn.ReLU(),
            torch_nn.Linear(HIDDEN, CLASSES),
        )
        state = {
            name: torch.tensor(value, dtype=torch.float32)
            for name, value in initial_state.items()
        }
    else:
        model = nnlearn_nn.Sequential(
            nnlearn_nn.Linear(FEATURES, HIDDEN),
            nnlearn_nn.ReLU(),
            nnlearn_nn.Linear(HIDDEN, CLASSES),
        )
        # nnlearn.Linear 将权重保存为 (输入维度, 输出维度)，与 PyTorch 相反。
        state = {
            name: value.T.copy() if name.endswith(".weight") else value
            for name, value in initial_state.items()
        }

    model.load_state_dict(state)
    model.to(device)
    return model


def _synchronize(framework, device, cupy_module):
    """等待设备上的异步运算完成，保证计时边界准确。"""
    if device != "cuda":
        return
    if framework == "pytorch":
        torch.cuda.synchronize()
    else:
        cupy_module.cuda.get_current_stream().synchronize()


def _train_steps(model, optimizer, criterion, batches, steps):
    """按固定批次执行前向、反向与参数更新。"""
    model.train()
    for step in range(steps):
        inputs, targets = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=False)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()


def _build_training_parts(framework, device, initial_state):
    """创建一轮运行所需的模型、优化器和损失函数。"""
    model = _make_model(framework, device, initial_state)
    if framework == "pytorch":
        optimizer = torch_optim.SGD(model.parameters(), lr=LEARNING_RATE)
        criterion = torch_nn.CrossEntropyLoss()
    else:
        optimizer = nnlearn_optim.SGD(model.parameters(), lr=LEARNING_RATE)
        criterion = nnlearn_nn.CrossEntropyLoss()
    return model, optimizer, criterion


def _evaluate(framework, model, features, labels, device):
    """在完整合成数据上计算训练 loss 与 accuracy；不计入训练耗时。"""
    if framework == "pytorch":
        inputs = torch.tensor(features, dtype=torch.float32, device=device)
        targets = torch.tensor(labels, dtype=torch.int64, device=device)
        no_grad = torch.no_grad
        criterion = torch_nn.CrossEntropyLoss()
        api = torch
    else:
        inputs = nnlearn.tensor(
            features, dtype=nnlearn.float32, device=device
        )
        targets = nnlearn.tensor(
            labels, dtype=nnlearn.int64, device=device
        )
        no_grad = nnlearn.no_grad
        criterion = nnlearn_nn.CrossEntropyLoss()
        api = nnlearn

    model.eval()
    with no_grad():
        logits = model(inputs)
        loss = criterion(logits, targets)
        predictions = api.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
    return float(loss.item()), float(accuracy.item())


def _run_once(framework, device, initial_state, batches, features, labels,
              steps, warmup, cupy_module):
    """先用独立模型预热，再测量全新模型的训练循环。"""
    warmup_model, warmup_optimizer, warmup_criterion = _build_training_parts(
        framework, device, initial_state
    )
    if warmup:
        _train_steps(
            warmup_model, warmup_optimizer, warmup_criterion, batches, warmup
        )
        _synchronize(framework, device, cupy_module)
    del warmup_model, warmup_optimizer, warmup_criterion

    model, optimizer, criterion = _build_training_parts(
        framework, device, initial_state
    )
    _synchronize(framework, device, cupy_module)
    start = time.perf_counter()
    _train_steps(model, optimizer, criterion, batches, steps)
    _synchronize(framework, device, cupy_module)
    elapsed = time.perf_counter() - start

    loss, accuracy = _evaluate(framework, model, features, labels, device)
    processed_samples = sum(
        len(batches[step % len(batches)][1]) for step in range(steps)
    )
    return {
        "seconds": elapsed,
        "loss": loss,
        "accuracy": accuracy,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="用相同的 MLP 训练任务比较 nnlearn 与 PyTorch。"
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="对照设备，默认 CPU；CUDA 需要两边均可用。",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="计入耗时的 SGD 更新步数，默认 1000。",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="不计入耗时的预热更新步数，默认 10。",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="每个框架重复运行次数，最终耗时取中位数，默认 3。",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps 必须大于 0")
    if args.warmup < 0:
        parser.error("--warmup 不能小于 0")
    if args.repeats <= 0:
        parser.error("--repeats 必须大于 0")
    return args


def main():
    args = _parse_args()
    cupy_module = None
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("PyTorch 当前无法使用 CUDA，不能进行同设备对照。")
        if not nnlearn.cuda.is_available():
            raise SystemExit(
                "nnlearn 当前无法使用 CUDA；请安装与本机 CUDA 匹配的 CuPy。"
            )
        import cupy as cupy_module

    features, labels = _make_dataset()
    initial_state = _make_initial_state()
    batches = {
        "nnlearn": _make_batches(features, labels, args.device, "nnlearn"),
        "pytorch": _make_batches(features, labels, args.device, "pytorch"),
    }

    timings = {"nnlearn": [], "pytorch": []}
    latest_results = {}
    for repeat in range(args.repeats):
        order = ("nnlearn", "pytorch")
        if repeat % 2:
            order = tuple(reversed(order))
        for framework in order:
            result = _run_once(
                framework=framework,
                device=args.device,
                initial_state=initial_state,
                batches=batches[framework],
                features=features,
                labels=labels,
                steps=args.steps,
                warmup=args.warmup,
                cupy_module=cupy_module,
            )
            timings[framework].append(result["seconds"])
            latest_results[framework] = result

    median_seconds = {
        framework: statistics.median(values)
        for framework, values in timings.items()
    }
    processed_samples = sum(
        len(batches["nnlearn"][step % len(batches["nnlearn"])][1])
        for step in range(args.steps)
    )

    print("nnlearn 与 PyTorch 训练基准")
    print(f"device={args.device}")
    print(
        f"任务：样本={SAMPLES}，特征={FEATURES}，隐藏单元={HIDDEN}，"
        f"类别={CLASSES}，batch_size={BATCH_SIZE}，学习率={LEARNING_RATE}"
    )
    print(
        f"训练配置：steps={args.steps}，warmup={args.warmup}，"
        f"repeats={args.repeats}，每次训练样本数={processed_samples}"
    )
    print(
        f"numpy={np.__version__}，pytorch={torch.__version__}，"
        f"pytorch_cpu_threads={torch.get_num_threads()}"
    )
    print("计时范围：预热后的前向、损失、反向和 SGD 更新；使用多次运行中位数。")
    print()

    for framework in ("nnlearn", "pytorch"):
        median = median_seconds[framework]
        result = latest_results[framework]
        print(
            f"{framework}: median={median:.6f} 秒，"
            f"steps/s={args.steps / median:.2f}，"
            f"samples/s={processed_samples / median:.2f}，"
            f"loss={result['loss']:.6f}，accuracy={result['accuracy']:.4f}"
        )

    elapsed_ratio = median_seconds["nnlearn"] / median_seconds["pytorch"]
    print(
        f"耗时比 nnlearn/PyTorch={elapsed_ratio:.2f}x "
        "（大于 1 表示 nnlearn 在本次基准中耗时更长）"
    )
    loss_gap = abs(
        latest_results["nnlearn"]["loss"] - latest_results["pytorch"]["loss"]
    )
    accuracy_gap = abs(
        latest_results["nnlearn"]["accuracy"]
        - latest_results["pytorch"]["accuracy"]
    )
    print(f"最终指标差值：loss={loss_gap:.6g}，accuracy={accuracy_gap:.6g}")
    print("提示：结果只适用于本示例的 MLP、参数和当前运行环境。")


if __name__ == "__main__":
    main()

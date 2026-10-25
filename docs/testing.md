# 测试与发行包验收

nnlearn 的运行依赖只有 NumPy 和 loguru。PyTorch 仅用于测试时的数值对照，不参与 nnlearn 的前向、自动微分或训练。以下命令从仓库根目录执行，使用固定的 uv 0.9.20；建议为每套环境使用独立目录。

## 固定测试环境

| 环境 | Python | NumPy | PyTorch | 验收内容 |
| --- | --- | --- | --- | --- |
| 最低兼容基线 | 3.8.20 | 1.24.4 | 不安装 | CPU 回归、无参考后端训练和状态恢复 |
| NumPy 2 基线 | 3.12.12 | 2.2.6 | 不安装 | 相同用例下的 NumPy 2 兼容性 |
| 完整 CPU 对照 | 3.12.12 | 2.2.6 | 2.6.0+cpu | 全量 CPU 回归、PyTorch 前向及梯度对照 |

`requirements/ci-common.txt` 固定测试工具及间接依赖；`ci-py38.txt`、`ci-py312.txt` 固定对应 NumPy；`ci-reference.txt` 额外固定 PyTorch 及其间接依赖。参考环境使用官方 CPU wheel 地址和 SHA-256，目标平台为 Linux x86_64 / CPython 3.12。`ci-build.txt` 单独固定 PEP 517 构建后端，避免隔离构建时绕过测试约束。

无 PyTorch 的 Python 3.8 环境：

```bash
uv venv --managed-python --python 3.8.20 .venv-py38
uv pip install --python .venv-py38/bin/python --constraint requirements/ci-py38.txt --build-constraints requirements/ci-build.txt -e '.[test]'
uv run --no-project --python .venv-py38/bin/python scripts/smoke_installed.py
uv run --no-project --python .venv-py38/bin/python -m pytest tests -q -ra
```

无 PyTorch 的 NumPy 2 环境：

```bash
uv venv --managed-python --python 3.12.12 .venv-py312
uv pip install --python .venv-py312/bin/python --constraint requirements/ci-py312.txt --build-constraints requirements/ci-build.txt -e '.[test]'
uv run --no-project --python .venv-py312/bin/python scripts/smoke_installed.py
uv run --no-project --python .venv-py312/bin/python -m pytest tests -q -ra
```

烟测首先确认环境无法导入 PyTorch，再执行真实线性回归训练，检查有限梯度、损失下降、模型恢复和 Adam 状态恢复后下一步更新的一致性。没有安装 PyTorch 的参考测试使用 `pytest.importorskip` 显示跳过，其余 CPU 测试照常执行。

Linux x86_64 的完整 CPU 对照环境：

```bash
uv venv --managed-python --python 3.12.12 .venv-reference
UV_HTTP_TIMEOUT=300 uv pip install --python .venv-reference/bin/python --constraint requirements/ci-reference.txt --build-constraints requirements/ci-build.txt -e '.[test-reference]'
uv run --no-project --python .venv-reference/bin/python python -c 'import torch, numpy; assert torch.__version__ == "2.6.0+cpu"; assert torch.version.cuda is None; assert numpy.__version__ == "2.2.6"; print(torch.__version__, numpy.__version__)'
uv run --no-project --python .venv-reference/bin/python -m pytest tests -q -ra
```

`test-reference` 中的 PyTorch 依赖适用于 Python 3.9 及以上；Python 3.8 使用无 PyTorch 兼容基线。其他平台可自行安装对应 CPU 参考 wheel，但不应使用 Linux 专用的 `ci-reference.txt`。

## wheel 与 sdist 安装验收

运行以下命令，无需事先安装 nnlearn 或 PyTorch：

```bash
uv run --no-project --managed-python --python 3.8.20 scripts/check_distributions.py --python 3.8.20 --constraints requirements/ci-py38.txt --out-dir dist/ci-py38
uv run --no-project --managed-python --python 3.12.12 scripts/check_distributions.py --python 3.12.12 --constraints requirements/ci-py312.txt --out-dir dist/ci-py312
```

脚本分别构建根目录 `nnlearn` 和 `torch_protocol/` 目录下独立 `torch-protocol` 的 wheel、sdist，每套 Python 创建四个临时环境，逐个安装产物并运行烟测。sdist 会在安装时重新构建 wheel。所有烟测均在仓库外工作目录运行，并拒绝从源码目录导入，从而发现包漏文件、依赖声明错误等问题。

nnlearn 产物须完成无 PyTorch 的训练和恢复验收；独立协议产物须在没有 nnlearn、NumPy、loguru 和 PyTorch 的环境中导入和校验协议。验收完成后删除临时环境，四个产物保留在指定 `--out-dir` 下。

新增脚本可单独检查：

```bash
uv run --no-project --with ruff==0.11.13 ruff check scripts
```

## 持续集成

`.github/workflows/ci.yml` 在 push、pull request 和手动触发时执行五个作业：两套无 PyTorch 兼容回归、一个完整 CPU 对照回归、两套 Python 的发行包安装验收。Python、NumPy、uv 和测试依赖均固定版本；`checkout`、`setup-uv` 固定到核实过的发行提交。工作流只读取仓库内容，不需要额外密钥。

推送工作流后，可在 GitHub Actions 的 `CI` 页面查看结果，也可使用：

```bash
gh workflow run ci.yml
gh run list --workflow ci.yml --limit 5
gh run watch RUN_ID --exit-status
```

每个测试作业使用 `pytest -ra` 输出跳过原因。必须同时检查通过、失败及跳过数量，不能把无 PyTorch 或无 CUDA 的跳过项视为对应能力已验收。

## CUDA 验收边界

默认 CI 使用 CPU runner，无法证明 GPU 正确性。既有 CUDA 测试在 `nnlearn.cuda.is_available()` 为假时调用 `pytest.skip`，明确显示“GPU 路径未验收”。CUDA 实机需具备兼容的 NVIDIA 驱动、GPU 和 CuPy，并单独运行 GPU 验收；可先筛选既有 CUDA 测试：

```bash
uv run --no-project --python .venv-cuda/bin/python -m pytest tests -q -ra -k cuda
```

CUDA 实机报告应记录硬件、驱动、Python、CuPy、NumPy 版本和实际执行的用例；只有 CPU 回归通过或 GPU 测试跳过，不构成 CUDA 验收完成。

## 官方参考

- [uv 的 GitHub Actions 集成](https://docs.astral.sh/uv/guides/integration/github/)
- [setup-uv v6.4.3](https://github.com/astral-sh/setup-uv/releases/tag/v6.4.3)
- [checkout v4.2.2](https://github.com/actions/checkout/releases/tag/v4.2.2)
- [PyTorch 官方 CPU wheel 索引](https://download.pytorch.org/whl/cpu/torch/)

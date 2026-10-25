# PLAN-066 实施结果：持续集成、可复现测试与安装验收

## 已实施

1. 根 `pyproject.toml` 增加 `test`、`test-reference`、`dev` 可选依赖，运行依赖仍只有 NumPy、loguru。新增 `requirements/ci-*.txt`，固定两套 Python / NumPy 基线、测试工具及间接依赖、构建后端与 PyTorch 2.6.0 CPU 对照环境。CPU wheel 使用官方地址及已核实的 SHA-256。
2. 新增 `.github/workflows/ci.yml`，push、pull request 和手动触发均运行五个作业：两套无 PyTorch 兼容测试、全量 PyTorch CPU 对照、两套 Python 的发行包验收。uv 固定为 0.9.20；checkout v4.2.2、setup-uv v6.4.3 固定到官方 tag 对应提交；工作流只读仓库权限。
3. 新增 `scripts/smoke_installed.py`，验证无 PyTorch 的真实训练、有限梯度、模型及 Adam 恢复后更新一致性；独立协议环境检查无框架和第三方运行依赖。支持拒绝源码目录导入。
4. 新增 `scripts/check_distributions.py`，两个包各构建 wheel、sdist，四个产物分别在新环境安装，并在仓库外运行烟测。Python 3.8、3.12 各验收一遍，共八个干净安装环境；输出产物保留在 `dist/`。
5. 既有 22 处 CUDA 测试由不可用时直接返回改为显式 `pytest.skip`。15 个顶层导入 PyTorch 的既有参考测试模块改用 `pytest.importorskip`，无 PyTorch 环境能够收集整套测试。
6. 初次干净回归暴露旧线性回归测试隐式依赖 matplotlib、只绘图而没有结果断言。现改为直接验证训练 MSE 小于 0.05，不再依赖图形库。
7. 新增中文 `docs/testing.md`，记录固定环境、完整复现命令、打包安装及 CUDA 验收边界。`.gitignore` 增加文档所用 `.venv-*` 环境目录。

## 本地验收

| 验收 | 结果 |
| --- | --- |
| Python 3.8.20 / NumPy 1.24.4，无 PyTorch，全套 tests | 192 passed、51 skipped |
| Python 3.12.12 / NumPy 2.2.6，无 PyTorch，全套 tests | 192 passed、51 skipped |
| 两套环境的训练及 Adam 恢复烟测 | 均通过，MSE 从 2.263633 降到 0.000001211 |
| Python 3.8.20，根包及协议 wheel / sdist 四项干净安装 | 全部通过 |
| Python 3.12.12，根包及协议 wheel / sdist 四项干净安装 | 全部通过 |
| `go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.7 .github/workflows/ci.yml` | 通过 |
| `uv run --no-project --with ruff==0.11.13 ruff check scripts` | 通过 |
| `git diff --check` | 通过 |

上述 CPU 矩阵已包含 PLAN-063 的半精度回归和 PLAN-064 的高阶导测试文件；跳过项是缺少参考 PyTorch 或 CUDA 设备的明确报告，不代表对应能力通过。其他并行计划后续合入的测试由整体验收再次执行。

## 正在跟进的验收

- 固定的 Python 3.12.12 / NumPy 2.2.6 / PyTorch 2.6.0+cpu 干净参考环境正在安装。第一次官方下载因 30 秒网络超时终止，现已按 uv 建议增大 HTTP 超时并保留活跃安装进程；完成后执行全量 CPU 对照及最新半精度、高阶导回归。
- GitHub API 确认仓库 `gseismic/nnlearn` 的 Actions 已启用、允许本工作流所用 Action；工作流尚未随本计划推送，因此远程 CI 结果待主代理提交、push 后验收。

## Review 与边界

已逐项检查发行包导入来源、构建隔离约束、依赖分层、Python 3.8 可用性和测试跳过语义。修复了 `uv run` 改变 PATH 后误选目标 Python 目录下旧 uv 的问题，子进程优先使用 uv 提供的 `UV` 可执行路径；同时核对了所有 CI 命令及 Action 官方提交。

本计划的默认 runner 为 CPU，无法替代 PLAN-067 的 CUDA 实机验收。主代理负责统一提交、push、远程 CI 跟进和开发索引维护。

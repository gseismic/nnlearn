# PLAN-066：持续集成、可复现测试与安装验收

## 目标

建立独立于开发机的 CPU 回归与发行包验收，固定 Python、NumPy、测试工具和 PyTorch 对照基线；验证框架运行及训练不依赖 PyTorch，并让无 CUDA 环境中的 GPU 测试明确显示跳过。

## 范围与步骤

1. 检查现有依赖、测试导入、独立 `torch-protocol` 打包结构及 Actions 配置，核实官方 Action 版本与 uv 用法。
2. 在根 `pyproject.toml` 增加测试、对照测试、开发工具可选依赖；保留仅 NumPy、loguru 的运行依赖。增加版本约束文件，覆盖 Python 3.8 / NumPy 1.24.4、Python 3.12 / NumPy 2.2.6，以及 PyTorch 2.6.0 CPU 对照基线。
3. 增加可复用安装烟测脚本，检查独立协议导入、张量前后向、真实训练、模型与优化器恢复；从仓库外运行并核实导入来源。构建根包及独立协议的 wheel、sdist，并在干净环境分别安装验证。
4. 新增 GitHub Actions：无 PyTorch 的 Python / NumPy 兼容矩阵、PyTorch CPU 全量对照回归、两个发行包的 wheel / sdist 构建和安装验收。固定 uv 版本与 Action 提交，支持 push、pull request、手动触发。
5. 将既有 CUDA 测试在设备不可用时直接返回的行为改为 `pytest.skip`，明确报告未验收范围。顶层导入 PyTorch 的参考测试改用 `pytest.importorskip`，使无 PyTorch 环境也能收集整套测试。首次实跑发现旧线性回归测试依赖 matplotlib 绘图却没有结果断言，将其改成直接断言训练误差，去除与回归无关的图形依赖。
6. 在 `docs/testing.md` 记录环境准备、复现命令、构建安装步骤、CI 检查范围和 CUDA 验收方式。
7. Review 本次改动，实跑兼容矩阵、构建安装与相关测试；写入 `PLAN-066-ci-packaging-OUTCOME.md`，由主代理统一提交并 push。

## 验收

- 两套无 PyTorch 环境可以完成训练烟测与无参考后端的 CPU 测试，测试开始前确认环境没有安装 PyTorch。
- Python 3.12 / NumPy 2.2.6 / PyTorch 2.6.0 CPU 环境全量回归通过，CUDA 不可用时显示 skip。
- `nnlearn` 与 `torch-protocol` 的 wheel 和 sdist 均能构建；四个产物分别在全新环境、仓库外工作目录安装并验证，协议环境不引入 nnlearn / NumPy / PyTorch。
- Actions 文件语法、版本及命令经过核对，文档命令与实际执行一致。

## 并行边界

本计划负责打包、CI、测试约束、安装烟测与既有 CUDA 测试跳过行为；不修改数值实现、高阶导实现、新增注意力实现及其他计划的新测试文件。共享工作树由主代理统一提交、push 与维护索引。

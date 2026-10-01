# PLAN-056 实施结果：将项目打包配置迁移到 pyproject.toml

## 实施内容

1. 新建根目录 `pyproject.toml`，使用 `setuptools>=64`、`setuptools.build_meta` 和 PEP 621 项目元数据；该版本支持文档使用的 PEP 660 可编辑安装。
2. 保留发行名 `nnlearn`、版本 `0.1.0`、描述、作者、项目地址及 `numpy`、`loguru` 依赖；包数据和非 zip-safe 设置也与原配置一致。Python 最低版本设为 `3.8`，与根包捆绑并导入的 `torch_protocol` 对齐。
3. 配置 setuptools 只发现 `nnlearn` 及其子包和 `torch_protocol`，与原 `find_packages()` 结果相同。
4. 删除根目录 `setup.py`，保留 `torch_protocol/pyproject.toml`。
5. 将当前设计文档和书稿环境说明改为引用根 `pyproject.toml`，并注明可编辑安装需要 pip 21.1 或更新版本。旧计划与结果中的 `setup.py` 记载保留，作为当时项目状态的历史记录。

## Review 与检查

- 使用 Python TOML 解析器读取配置，核对构建后端及发行元数据。
- 对照原包发现结果核对：`nnlearn`、`nnlearn.functional`、`nnlearn.nn`、`nnlearn.optim`、`nnlearn.utils`、`nnlearn.utils.data` 和 `torch_protocol` 均被发现。
- 确认根目录 `setup.py` 已删除，独立的 `torch_protocol/pyproject.toml` 仍存在。
- `git diff --check` 通过。
- 未运行测试套件或构建发行包。

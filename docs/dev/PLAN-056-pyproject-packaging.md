# PLAN-056：将项目打包配置迁移到 pyproject.toml

## 目标

将根项目的打包配置从 `setup.py` 迁移到根目录 `pyproject.toml`，让 PEP 517/621 配置成为当前唯一的打包配置入口。

## 实施范围

1. 新建根目录 `pyproject.toml`，声明 setuptools 构建后端和项目元数据。
2. 保留现有发行信息：包名 `nnlearn`、版本 `0.1.0`、依赖、作者、项目地址及非 zip-safe 设置。根包包含并导入 `torch_protocol`，因此 Python 最低版本对齐该子包已声明的 `>=3.8`。
3. 显式配置包发现，使根项目仍包含 `nnlearn` 全部子包和 `torch_protocol`，与原 `find_packages()` 结果一致。
4. 删除根目录 `setup.py`，保留独立发行包 `torch_protocol/pyproject.toml`。
5. 更新当前面向用户的书稿和设计文档中对根打包配置的描述。既有计划与结果文档保留其记录当时 `setup.py` 的历史事实。

## 验收标准

- 根目录存在可读的 `pyproject.toml`，配置 setuptools 构建系统和 PEP 621 项目元数据。
- 构建要求使用 setuptools 64 或更新版本，以支持仓库文档采用的 PEP 660 editable 安装。
- 根目录不再存在 `setup.py`；`torch_protocol/pyproject.toml` 不受影响。
- 项目名、版本、依赖、作者和地址与原配置一致；Python 要求为 `>=3.8`，与捆绑的 `torch_protocol` 一致。
- setuptools 包发现包含原先的全部项目包：`nnlearn`、其子包和 `torch_protocol`。
- 当前用户文档使用 `pyproject.toml` 描述根项目配置；历史记录仍准确反映当时配置。
- `git diff --check` 通过并完成人工 review。

## 实施步骤

1. 对照现有 `setup.py` 写入根目录 `pyproject.toml`，使用 setuptools 64 或更新版本，将元数据写入 `[project]`，将包发现和打包选项写入 `[tool.setuptools]`。
2. 删除根 `setup.py`，不修改 `torch_protocol` 子项目的独立构建配置。
3. 更新书稿环境说明与当前项目改名设计文档。
4. 核对元数据字段及 setuptools 包发现范围，检查当前文档和代码中是否有仍需使用根 `setup.py` 的地方。
5. review 差异并记录结果。

## 范围说明

本任务只迁移根项目的打包配置，不调整发行名、版本和依赖，也不迁移独立的 `torch_protocol` 子项目。Python 最低版本声明修正为与根包实际导入的 `torch_protocol` 相同。

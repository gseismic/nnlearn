# nnlearn 项目命名与迁移设计

日期：2026-10-01

## 决策

项目、GitHub 仓库、Python 发行包和导入包统一使用小写名称 `nnlearn`。现有发布版本保持 `0.1.0`；本次只调整项目标识，不改变张量、自动微分和训练 API 的语义。

## 公开接口

- 安装发行包后使用 `import nnlearn`、`import nnlearn.nn` 和 `import nnlearn.optim`。
- 对照示例使用 `USE_NNLEARN` 选择 nnlearn 或官方 PyTorch 路径。
- 不提供旧 `nlearn` 导入空间的兼容包；使用者需要更新导入名和示例环境变量。
- 仓库地址使用 `https://github.com/pai-studio/nnlearn`。

## 迁移范围

更新 Python 包目录、源码导入、发行元数据、示例、测试、README、教程、书稿和当前设计文档中的项目标识；重命名仓库图片和本地检出目录。历史计划和结果文件保留当时记录的名称，避免改写历史事实。

## PyTorch 边界

名称迁移只影响本项目标识。代码中以 `torch` 或 `PyTorch` 指官方 PyTorch、比较对象或其风格时继续保留。

## 完成标准

1. 源码目录、导入、发行元数据、运行时版本、当前项目文档和示例开关均使用 `nnlearn`。
2. `pyproject.toml` 的发行包名和 `nnlearn.__version__` 均为 `nnlearn` 与 `0.1.0`。
3. GitHub 仓库、本地检出目录和本地 `origin` 使用 `pai-studio/nnlearn`。
4. 历史计划与结果文件维持原样；当前源码和文档中的旧导入名不残留。

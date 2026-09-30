# PLAN-054：将项目重新命名为 nnlearn

## 目标

将项目、Python 包、仓库元数据及当前文档中的项目标识从 `nlearn` 统一改为 `nnlearn`，并将 GitHub 仓库和本地检出目录改为 `nnlearn`。

## 范围

1. 将 Python 包目录、源码导入、测试和示例迁移为 `nnlearn`。
2. 将发行包名、运行时版本属性、README、安装地址、示例环境变量、教程、书稿和当前设计文档统一为新名称。
3. 将 `images/nlearn.png` 重命名为 `images/nnlearn.png`。
4. 将 GitHub 仓库改名为 `pai-studio/nnlearn`，更新本地 `origin`，并将工作区目录改名为 `nnlearn`。
5. 保留既有计划、结果和标明历史快照的设计文档，避免改写过去的决策记录。
6. 完成静态名称、元数据和 diff 检查；按仓库约定生成结果文件，使用中文提交并立即推送。

## 实施顺序

1. 建立命名设计与本计划。
2. 迁移源码包目录与标识，更新当前文档和图片路径。
3. 检查包名、导入、安装命令、相对链接、旧名称残留和 Git 差异。
4. 将 GitHub 仓库改名并更新本地远程地址；生成结果文件。
5. 将本地检出目录改名，提交全部变更并推送到 `origin/main`。

## 验收标准

- Python 包目录和项目导入名为 `nnlearn`，发行包名为 `nnlearn`。
- `nnlearn.__version__` 和 `setup.py` 版本保持 `0.1.0`。
- 活跃源码、示例、测试和当前文档不再引用旧项目导入名；PyTorch 对照路径保持正确。
- GitHub 仓库、本地工作区目录及 `origin` 均指向 `pai-studio/nnlearn`。
- 计划与结果文档已提交，提交已推送。

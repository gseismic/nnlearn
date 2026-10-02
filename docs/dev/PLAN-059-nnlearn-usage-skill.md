# PLAN-059：编写 nnlearn 用法 Skill

## 目标

在仓库 `skills/` 下新增一份 `nnlearn-usage` Skill，帮助 Codex 根据仓库当前文档、示例和实现，为用户编写、调整和排查 `nnlearn` 使用代码。

## 实施范围

1. 创建符合 Codex Skill 目录格式的 `skills/nnlearn-usage/SKILL.md`，用中文说明触发场景、适用边界和使用流程。
2. 覆盖张量与自动微分、`nn`/`optim` 训练闭环、数据加载、设备选择及 PyTorch 风格 API 的兼容边界。
3. 指向 README、入门教程、主题示例及 `torch_protocol/` 等当前仓库中的权威资料，要求遇到精确 API 问题时按源码核实。
4. 不新增重复的 API 手册或辅助脚本；保持 Skill 自包含且入口简洁。

## 验收标准

- Skill 位于 `skills/nnlearn-usage/SKILL.md`，frontmatter 包含有效的 `name` 和 `description`。
- 指令能引导后续使用者找到适合的教程、示例和实现，并明确不承诺完整 PyTorch 兼容。
- 文档不包含未经仓库资料支持的 API 或环境承诺。
- Skill Creator 的 `quick_validate.py` 与 `git diff --check` 通过，完成内容复核。
- 生成对应实施结果文档，按仓库约定提交并推送。

## 实施步骤

1. 对照项目说明、入门教程、代表性示例、公开模块和兼容协议确认用法及边界。
2. 编写 Skill，并用 Skill Creator 验证格式与占位内容。
3. 复核引用路径、指令范围和兼容性表述，记录结果。
4. 中文提交本次变更并立即推送。

## 范围说明

本任务新增仓库内的使用指导 Skill 和计划/结果文档，不修改 `nnlearn` 实现、现有教程、API 或项目配置。

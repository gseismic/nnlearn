# PLAN-053：实现 Torch Protocol v1

状态：已完成。实施记录见 [PLAN-053-torch-protocol-v1-OUTCOME.md](PLAN-053-torch-protocol-v1-OUTCOME.md)。

## 目标

交付可安装、与后端解耦的 `torch_protocol` 包。其公共 Protocol 定义 PyTorch `torch` 与 nlearn 的共同接口；nlearn 的实现依赖并满足该契约，共用训练程序可以接收任一后端。

## 范围

1. 按 `docs/design/torch-protocol-20260930-overview.md` 定义 Tensor、Module、Loss、Optimizer、Dataset、DataLoader 及后端命名空间 Protocol。
2. 提供无框架依赖的 `validate_backend()` 与独立包元数据。
3. 让 nlearn 的核心类型显式实现协议，暴露 `nn`、`optim` 和 `utils.data` 根模块入口，并在导入时检查 nlearn 后端。
4. 编写同一段训练示例和共用 conformance suite；nlearn 测试必须运行，PyTorch 测试在安装官方 PyTorch 时运行。
5. 文档清楚区分 v1 承诺接口与 nlearn 扩展，不声称兼容任意 PyTorch 程序。

## 实施顺序

1. 完成协议设计文档与本计划。
2. 实现 `torch_protocol` 包、类型协议、入口检查及独立构建配置。
3. 将 nlearn 核心对象接入协议，并按共用命名空间提供后端入口。
4. 增加共享训练示例、nlearn conformance tests 和可选 PyTorch conformance tests。
5. 构建独立协议 wheel，执行 nlearn 测试、API 示例、代码编译、链接和命名检查。
6. review 所有契约及语义限制，写结果文档，中文提交并立即 push。

## 验收标准

- `torch_protocol` 可独立构建安装，且不导入 `torch` 或 `nlearn`。
- `nlearn` 显式依赖协议类型并通过启动检查及行为 conformance tests。
- 共用训练示例对 nlearn 后端运行通过；PyTorch 后端测试存在且在 PyTorch 可用环境运行。
- 协议成员表、语义边界、类型定义和测试范围一致。
- 所有设计、计划与结果文档均纳入提交并推送。

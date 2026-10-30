# Cortex · Claude Code 说明

@AGENTS.md

## 项目工作约定
- 需求：docs/PROJECT_BRIEF.md；计划与里程碑状态：PLAN.md；技术决定：docs/DECISIONS.md。
- 按 PLAN.md 的里程碑推进，满足验收标准后停下来演示，不要跳到下一个里程碑。
- 涉及版本和 API（mlx-lm、MLX 内存接口、Claude Code hooks、Codex 会话格式）先联网核实，结果记入 DECISIONS.md。
- 不确定的技术选择给 2–3 个方案和推荐，由用户决定。
- 删除文件的操作先 dry-run 并写日志。
- 永远不读取凭证：~/.claude/.credentials.json、~/.codex/auth.json、.env、macOS 钥匙串。
- 运行数据（模型、适配器、样本、评测结果）不放进仓库。

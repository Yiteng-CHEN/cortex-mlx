# Neurolume · Claude Code 说明

@AGENTS.md

## AOCI 仓库认知
- 本仓库由 AOCI-CODE 管理：`aoci.txt`（根）、`aoci.meta.txt`（标签字典）、`aoci.code.txt`（每个文件一行的认知条目）、`.aoci/`（治理状态）。规则见上面导入的 AGENTS.md。
- 会话开始先调 `aoci_rules`、`aoci_overview` 读取认知；找文件用 `aoci_search` / `aoci_get_entries`。
- 一段开发任务结束、代码和测试稳定后，调一次 `aoci_maintain`，读源码写完整批条目，用 `aoci_update_entry` 提交，直到 remaining=0；最后 `aoci --repo . verify` 和 `check`。
- `.mcp.json`、`.claude/settings.json`、`.codex/` 含本机绝对路径，不提交。

## 项目工作约定
- 需求：docs/PROJECT_BRIEF.md；计划与里程碑状态：PLAN.md；技术决定：docs/DECISIONS.md。
- 按 PLAN.md 的里程碑推进，满足验收标准后停下来演示，不要跳到下一个里程碑。
- 涉及版本和 API（mlx-lm、MLX 内存接口、Claude Code hooks、Codex 会话格式）先联网核实，结果记入 DECISIONS.md。
- 不确定的技术选择给 2–3 个方案和推荐，由用户决定。
- 删除文件的操作先 dry-run 并写日志。
- 永远不读取凭证：~/.claude/.credentials.json、~/.codex/auth.json、.env、macOS 钥匙串。
- 运行数据（模型、适配器、样本、评测结果）不放进仓库。

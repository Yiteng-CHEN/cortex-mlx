# Cortex 开发计划

需求原文见 [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)，技术决定见 [docs/DECISIONS.md](docs/DECISIONS.md)。
每个里程碑满足验收标准后停下来演示，经确认再进入下一个。

状态图例：⬜ 未开始 · 🟨 进行中 · ✅ 完成 · ⏸ 等待用户决定

## 当前状态

- ✅ 仓库初始化（git，main 分支）
- ✅ AOCI 接入：Claude Code（MCP + PreToolUse hook）
- 🟨 AOCI 接入：Codex（手动写入 `.codex/config.toml`，待 `aoci doctor` 确认）
- ✅ 重选底座候选：Qwen3.6-27B + Devstral Small 2（D13）；uv（D9）；数据目录（D10）
- 🟨 M0 评测与选型：工具完成；首次实测见 D16（两个模型都能加载；Qwen3.6 LoRA 在 2048 长度爆内存，探针已改为自动降长度重测）；⏸ 等待重跑探针和挖题仓库路径
- ✅ 底座选定 Devstral Small 2（D19）；3D 大脑原型已发布（示例信号），权重采样脚本 scripts/dump_weights.py

- 🟨 界面化（D20）：模型库、训练向导、一键启动、聊天与实时层活动已写好并通过测试；⏸ 待在 Mac 上首次真机验证

## 计划目录结构

```
Cortex/
├── PLAN.md  README.md  AGENTS.md  CLAUDE.md
├── pyproject.toml                # Python 包 cortex（uv 管理，见 D9）
├── config/
│   ├── cortex.toml               # 默认配置（进 git）：数据目录、预算、候选模型
│   └── local.toml                # 本机覆盖（不进 git）：挖题仓库路径与测试命令
├── src/cortex/
│   ├── config.py
│   ├── evals/                    # M0：挖题、任务格式、判分沙箱、对比报告
│   ├── models/                   # M0/M1：模型加载、层结构探测、后端抽象
│   ├── server/                   # M1：OpenAI 兼容接口 + WebSocket 事件
│   ├── telemetry/                # M1/M5：层 hook、激活汇总、事件契约
│   ├── scheduler/                # M1：任务队列、空闲/电源/温度/内存检测
│   ├── store/                    # M1：SQLite + sqlite-vec
│   ├── observer/                 # M2：git hook、Claude Code hook、Codex 会话解析、脱敏
│   ├── learner/                  # M2–M4：RAG、规则、影子对照、LoRA、闸门、版本
│   └── gc/                       # M3：磁盘预算与清理
├── web/                          # M1 起：React + Vite + TS（M5 热力图，M6 3D 大脑）
├── models/                       # 模型下载（不进 git，D17）
├── scripts/                      # m0-bench.sh、fixture_repo.py；以后：一键启停、launchd、hook 安装
├── docs/
│   ├── PROJECT_BRIEF.md  DECISIONS.md
│   ├── brain-events.md           # M1：事件契约说明
│   └── schemas/                  # M1：JSON Schema
└── tests/
```

运行数据放在仓库外的数据目录 `~/Library/Application Support/Cortex/`（D10）：

```
Cortex/                           # 数据目录
├── evals/{candidates,dev,holdout}/   # 评测题清单（D14），holdout-hashes.txt
├── results/                      # 评测结果 JSON + 对比表 Markdown
├── worktrees/                    # 判分用的临时 git worktree（用完即删）
├── models/                       # 转换后的模型（如需要）
└── logs/
```

---

## M0 评测与选型

目标：拿到一套能信任的评测集，用它选出底座模型。

任务：
1. ✅ **核实环境**：结论见 D13。
2. ✅ **项目骨架**：uv 项目、`cortex` 包、两层配置、`cortex` 命令行、unittest。
3. ✅ **题目格式**：一题一个 JSON 清单，只记引用（仓库名、base/target 提交、源文件、测试文件、题面、内容哈希），存数据目录（D14）。
4. ✅ **判分沙箱**：git worktree 检出 base 提交 + target 的测试文件；只允许写入题目声明的源文件；超时杀进程组；CPU/文件大小限制；最小化环境变量；用完即删，启动时清理残留。
5. ✅ **挖题器** `cortex eval mine <仓库名>`：找同时修改源码和测试的提交；过滤过大、二进制、超出上下文预算的提交；验证 base 失败、target 通过，各连跑 3 次结果一致。
6. ✅ **审核** `cortex eval review`：d=dev、h=holdout、r=拒绝、s=跳过；holdout 记录内容哈希。
7. ✅ **候选对比** `cortex eval run` / `cortex eval compare`：通过率、格式错误率、tokens/s、峰值内存。
8. ✅ **模型工具** `cortex models inspect`（层类型）、`cortex models train-probe`（LoRA 10 步的秒/步与峰值内存）。
9. ⏸ **在 Mac 上实测**：`scripts/m0-bench.sh`。
10. ⏸ **真实题库**：用你的仓库挖题并审核。

验收标准：
- 至少 15 道审核通过的题（其中约 1/3 为 holdout）
- `cortex eval validate` 全部通过
- 两个候选的对比表 + 推荐理由

⏸ 需要你：
- 提供 1–3 个有测试、提交历史较多的仓库路径（挖题用）
- 审核挖出的题目
- 看完对比表后选定底座

## M1 推理服务 + 数据契约

1. `cortex-server`：加载底座 + 当前适配器；`/v1/chat/completions`（含流式）、`/v1/models`；对话写入 SQLite。
2. **层结构探测**：自动列出每层类型（全注意力 / 线性注意力 / MLP / MoE），生成 `brain-layout.json` 供前端布局。
3. **事件契约**：`docs/brain-events.md` + `docs/schemas/*.json`，覆盖 token 事件、学习事件、版本快照、观察者状态；附 mock 事件生成器 `cortex mock-events`。
4. **调度器骨架**：任务队列、优先级（推理 > 一切）、空闲/接电/温度/内存压力检测、静默模式。
5. **资源监控**：内存预算接口（MLX 限制 API 先核实）、磁盘占用统计。
6. **launchd**：`scripts/install-launchd.sh`、一键启停脚本。
7. **最小前端**：Vite 项目、本地聊天页、连接状态。

验收标准：编辑器或 OpenCode 能接入使用；mock 事件能被前端收到；launchd 一键启停可用。

## M2 观察者 + RAG

1. **核实**：Claude Code hooks（会话结束事件）和会话记录格式；Codex 会话记录路径与格式。写入 DECISIONS.md。
2. git post-commit hook 安装器（只装在配置列出的仓库）。
3. 会话解析 → 用文件编辑记录对齐 diff → 区分 AI 改动和手改。
4. 脱敏器 + 凭证黑名单（`~/.claude/.credentials.json`、`~/.codex/auth.json`、`.env`、钥匙串一律不碰），带单元测试。
5. 样本库：待验证 → 已验证（测试通过 + 已提交 + 7 天未回滚）；去重、压缩。
6. RAG：sqlite-vec 检索，推理时注入相关样本；可在前端查看和删除条目。

验收标准：一次真实的 Claude Code 会话结束后自动产出已脱敏样本；报告有/无 RAG 的 dev 分数。

## M3 影子对照 + 磁盘治理

1. 影子任务：git worktree 中本地重做，跑相同测试，记录通过与否和意外程度；用完即删，启动时清理残留。
2. 每日次数上限、只在调度器允许时运行。
3. 评测曲线：按时间和版本记录 dev 分数、影子通过率。
4. GC：磁盘总预算、80% 告警、各类数据保留策略、`--dry-run` 和日志。

验收标准：影子任务自动运行并清理；曲线可查；GC dry-run 输出可读。

## M4 LoRA + 闸门

1. 训练集构建：已验证样本 + 旧样本回放；排除 holdout 哈希。
2. LoRA 训练（只存适配器），记录样本来源、超参、各层 ΔW 范数、梯度范数。
3. 闸门：dev 和 holdout 都不低于当前版本才上线，否则回滚；规则层（同类错误 ≥3 次生成草稿，经批准加入）。
4. 版本命令：`cortex versions list/promote/rollback/pin`。

验收标准：完整走通"训练 → 评测 → 上线或回滚"。**如果 LoRA 没有超过 RAG，报告并暂停，由你决定是否继续。**

## M5 2D 热力图

层 × 时间的激活热力图；学习事件叠加 ΔW；真实事件和 mock 事件驱动同一组件；"轻量/完整"两档。

验收标准：本地推理时热力图实时变化，切换到 mock 结果一致。

## M6 3D 大脑

脑区布局（来自 `brain-layout.json`）、思考高亮、类 EEG 波形（标注"风格化呈现"）、学习发光、版本时间轴对比、重要性视图（梯度范数、消融）、"观察中"状态。

验收标准：思考、学习、时间轴、重要性四个视图都由真实数据驱动。

---

## 风险与未决问题

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| R1 | Qwen3.6 混合架构的 LoRA 速度和内存；Devstral Small 2 的 MLX 权重由 mlx-vlm 转换，mlx-lm 能否加载未确认 | 可能换对照模型（Qwen3.6-35B-A3B） | M0 `models inspect` / `train-probe` 实测（D13） |
| R2 | 线性注意力层没有注意力矩阵 | 可视化按层类型区分 | M1 层结构探测 |
| R3 | LoRA 可能不比 RAG 强 | 可能只保留 RAG + 规则 | M4 闸门如实报告 |
| R4 | Claude Code / Codex 会话格式可能随版本变 | 解析器失效 | 解析器版本化 + 测试样本 |
| R5 | 用云端输出训练的条款限制 | 合规 | README 提醒，默认偏重项目约定类知识 |
| R6 | 挖出的题目测试不稳定或依赖环境 | 评测失真 | 连跑 3 次过滤 + 人工审核 |

## 下一步

1. 在 Mac 上首次提交，并运行 `aoci --repo . doctor` 确认 Codex MCP。
2. 在 Mac 上运行 `scripts/m0-bench.sh`（会下载约 45GB 模型），把输出发回。
3. 提供 1–3 个挖题仓库路径，写进 `config/local.toml`，然后挖题、审核。
4. 根据对比表选定底座 → M0 完成。

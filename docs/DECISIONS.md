# 技术决定记录

每条记录：背景、决定、理由、状态（已定 / 待定 / 待核实）。改变决定时不删旧条目，新增一条并注明取代关系。

## D1 单机运行 · 已定
所有组件（本地模型、观察者、Claude Code、Codex、前端）都在同一台 MacBook Pro（M5 Max，128GB）上运行，不做跨机器同步。

## D2 推理与训练用 MLX-LM，不用 Ollama · 已定
Ollama 拿不到中间层激活，也不能训练。MLX-LM 在 Apple Silicon 上原生运行，同一套库可做推理、层 hook 和 LoRA。
待核实：目标模型的具体支持情况（见 D8、PLAN R1）。

## D3 云端侧只观察，不接入 · 已定
不在 Cortex 里对接任何云端模型 API，也不使用或转存 Claude/Codex 的登录凭证。用户照常使用 Claude Code / Codex，Cortex 通过 git hook、Claude Code hooks 和本地会话记录观察结果。

## D4 评测集从 git 历史挖掘，分 dev / holdout · 已定
真实任务比手写题更能反映用户的工作。holdout 只用于上线闸门，用内容哈希保证永不进入训练和知识库。

## D5 学习分层，逐层用数据证明 · 已定
RAG → 规则 → LoRA。每层都要在评测上证明比不用更好才上线；LoRA 不比 RAG 强时暂停并由用户决定。

## D6 推理服务提供 OpenAI 兼容接口 · 已定
本地模型可直接接入编辑器 / OpenCode 等工具；任何调用都驱动大脑可视化，前端不必重做完整聊天工具。

## D7 存储用 SQLite + sqlite-vec · 已定
一个数据库同时存结构化数据和向量，少一个服务依赖。

## D8 底座候选 · 已被 D13 取代（候选部分）
- Qwen3.8-27B：稠密 27B，4bit 约 18GB，起步首选。第三方资料显示它沿用 Qwen3.5 的混合 Gated DeltaNet + gated attention 架构，且是原生视觉语言模型——需核实 mlx-lm 能否直接加载与 LoRA，还是需要 mlx-vlm。
- Qwen3-Coder-Next：80B 总参 / 3B 激活 MoE，4bit 约 46GB，面向 agentic 编程。需核实 MoE 的 LoRA 支持与专家路由可取。
- 排除 4bit 超过 100GB 的模型（训练无余量）。
以上参数来自第三方资料，M0 第 1 步联网核实后更新本条。

## D9 Python 工具链 · 已定：uv（2026-10-30）
依赖锁定、虚拟环境、Python 版本管理一体。核心包只用标准库，`mlx-lm` 作为可选依赖（`uv sync --extra mlx`，仅 macOS）。测试用标准库 unittest，不依赖额外包。

## D10 运行数据目录 · 已定：`~/Library/Application Support/Cortex/`（2026-10-30）
不进仓库，不被 AOCI 扫描。可用环境变量 `CORTEX_DATA_DIR` 或 `config/local.toml` 覆盖（测试时用临时目录）。

## D11 AOCI 作为代码认知层 · 已定
用 AOCI（0.1.0-rc23）为 Claude Code 和 Codex 维护项目地图。
- Claude Code：`aoci init --agent claude --hooks` 写入 `.mcp.json` 和 `.claude/settings.json`（PreToolUse hook）。
- Codex：第二次 `init` 被拒绝（"该命令或兼容写入路径不支持修改 Volumes v1 正式认知"），因此手动写入 `.codex/config.toml` 的 MCP 配置，未装 Codex hooks。
- `aoci.txt`、`aoci.meta.txt`、`aoci.code.txt`、`AGENTS.md` 进 git；`.aoci/`、`.mcp.json`、`.claude/settings.json`、`.codex/config.toml` 含本机绝对路径，不进 git。
- 项目代码量很少时先不建完整索引，M0 骨架完成后再由 agent 建索引。

## D12 项目名 Cortex · 已定
Python 包名 `cortex`，推理服务 `cortex-server`，命令行 `cortex`。

## D13 底座候选改为 Qwen3.6-27B（主选）+ Devstral Small 2（对照）· 已定候选，M0 实测后选定（2026-10-30）
取代 D8 的候选部分。

| | Qwen3.6-27B | Devstral Small 2 |
|---|---|---|
| 发布 / 协议 | 2026-04 / Apache 2.0 | 2025-12-09 / Apache 2.0 |
| 结构 | 稠密 27B，64 层 = 16 ×（3 × Gated DeltaNet → FFN，1 × Gated Attention → FFN），带视觉编码器，原生 262K 上下文 | 稠密 24B，`Mistral3ForConditionalGeneration`（带视觉），256K 上下文 |
| SWE-bench Verified（厂商自报） | 77.2（Terminal-Bench 2.0 59.3） | 68.0 |
| MLX 权重 | `unsloth/Qwen3.6-27B-MLX-8bit`（约 29GB） | `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit`（15.1GB，**用 mlx-vlm 转换**） |

放弃的候选：
- **Qwen3.8-27B**：2026-08 发布，尚无公开基准分；与 3.6-27B 结构相同（NVIDIA NeMo 文档）；MLX 上只有个人 LoRA 教程，且依赖未合并的 mlx-lm PR #1389 才能达到约 110 tok/s 训练速度（否则约 50 tok/s）。等工具成熟后可直接替换 3.6（流水线和大脑布局不变，用样本库重训）。
- **Qwen3-Coder-Next**：`qwen3_next` 混合注意力 + MoE，MLX 上 MoE LoRA 最不成熟。

降低风险的新事实：MLX 0.32.4（2026-10-14）加入了 Gated DeltaNet 和 SDPA 的原生反向传播（VJP），Qwen3.6 系列的 LoRA 训练不再依赖第三方补丁——仍需 M0 训练探针实测速度和内存。

待 M0 在 Mac 上实测：
1. Devstral Small 2 的 MLX 权重由 mlx-vlm 转换，mlx-lm 能否直接加载并做 LoRA 未确认；不行则用 `mlx_lm.convert` 从 `mistralai/Devstral-Small-2-24B-Instruct-2512` 自己转，仍不行则改用 Qwen3.6-35B-A3B（SWE-bench Verified 73.4）作对照。
2. 两个候选量化位数不同（8bit vs 4bit），对比时注明。

可视化影响：Qwen3.6 的 48 个 DeltaNet 层没有注意力矩阵，用递归状态范数和门控值表示；16 个 Gated Attention 层显示注意力头。Devstral 每层都是标准注意力。

来源：[Qwen3.6-27B 模型卡（镜像）](https://huggingface.co/Prince-1/Qwen3.6-27B)、[Devstral Small 2](https://www.llmreference.com/model/devstral-small-2)、[Qwen3.8-27B LoRA 教程](https://dev.to/mihai_leanzero/how-to-fine-tune-qwen38-27b-with-lora-on-a-mac-mlx-fine-tuning-from-scratch-2kl4)、[MLX 0.32.4](https://freedom.tech/posts/2026-10-14-mlx-0-32-4/)、[Devstral MLX 4bit](https://huggingface.co/mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit)、[Qwen on a Mac (OptiQ)](https://mlx-optiq.com/docs/qwen)。

## D14 评测题清单存数据目录，不进 git · 已定（2026-10-30）
清单只记录引用（仓库名 + 提交 SHA + 文件列表），不复制代码。放在 `<数据目录>/evals/{candidates,dev,holdout}/`：清单指向本机仓库，且 holdout 不应进入 git 历史被任何工具读到。挖题仓库的本机路径和测试命令写在 `config/local.toml`（不进 git）。

## D15 开发与测试环境 · 已定（2026-10-30）
Cortex 的核心代码在任何有 Python ≥3.11 和 git 的环境里都能运行和测试（测试用 mock 后端和合成仓库）。MLX 相关步骤（下载模型、基准、训练探针）只能在 Mac 原生运行，通过 `scripts/m0-bench.sh` 执行。

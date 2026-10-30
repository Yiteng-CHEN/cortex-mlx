# Cortex

本地自进化编程助手。你照常用 Claude Code / Codex 写代码，Cortex 在后台观察学习，逐渐掌握你的项目约定；3D 大脑 UI 展示它思考和成长时的真实内部信号。

> 状态：M0（评测与选型）工具已完成，等待在 Mac 上实测。计划见 [PLAN.md](PLAN.md)，需求见 [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md)，技术决定见 [docs/DECISIONS.md](docs/DECISIONS.md)。

## 日常使用（不需要命令）

双击项目里的 **Cortex.command**，浏览器会自动打开界面：
- **模型库**：一键下载、加载、卸载、删除模型；也可以粘贴 Hugging Face 上的模型名添加新模型。
- **聊天**：顶部下拉框切换模型，输入问题，右侧大脑图和下方网络层图会随模型的真实活动发光。
- **训练**：点“训练”，按向导导入数据 → 选模型和强度 → 开始；训练结果是单独的“适配器”，不改动原模型，不满意可恢复原模型。
- 第一次需要先装 [uv](https://docs.astral.sh/uv/)：`brew install uv`。

## 环境

- macOS，Apple Silicon（开发机：MacBook Pro M5 Max，128GB）
- [uv](https://docs.astral.sh/uv/)：`brew install uv`
- git
- 已登录的 Claude Code 和 Codex；AOCI（`~/.local/share/aoci/bin/aoci`）

```bash
cd ~/Documents/Cortex
uv sync                 # 核心（只用标准库）
uv sync --extra mlx     # 加上 MLX（模型相关命令需要）
uv run cortex sys config
```

运行数据在 `~/Library/Application Support/Cortex/`（可用 `CORTEX_DATA_DIR` 覆盖）。模型下载在项目的 `models/` 目录（`CORTEX_MODELS_DIR` 覆盖，见 D17）；用 `cortex` 命令时会自动把 Hugging Face 缓存指向这里。

## M0：评测与选型

1. **配置挖题仓库**：复制 `config/local.example.toml` 为 `config/local.toml`，为每个仓库写路径和测试命令（`{tests}` 会替换成题目的测试文件）。
2. **挖题**：`uv run cortex eval mine <仓库名>`——找同时改了源码和测试的提交，验证"改之前测试失败、改之后通过"，各连跑 3 次排除不稳定测试。
3. **审核**：`uv run cortex eval review`——逐题按 `d`（dev）/ `h`（holdout）/ `r`（拒绝）/ `s`（跳过）。holdout 只用于上线闸门，永不进入训练。
4. **实测候选**：`scripts/m0-bench.sh`——依次对两个候选做层结构检查、dev 评测、LoRA 训练探针，最后输出对比表。首次运行会下载约 45GB 模型。
5. **对比**：`uv run cortex eval compare --split dev`

先体验流程（不需要模型）：

```bash
python3 scripts/fixture_repo.py /tmp/cortex-fixture   # 生成合成仓库，按提示写入 config/local.toml
uv run cortex eval mine fixture
uv run cortex eval review
uv run cortex eval run --candidate mock-reference --candidate mock-noop
uv run cortex eval compare --split dev
```

其他命令：`cortex eval list -v`、`cortex eval cleanup [--apply]`（清理残留 worktree，默认只预览）、`cortex models inspect --candidate X`、`cortex models train-probe --candidate X`、`cortex sys mem --candidate X`。

## 测试

```bash
uv run python -m unittest discover -s tests -t .
```

## AOCI

本仓库已接入 AOCI：
- Claude Code：`.mcp.json` + `.claude/settings.json`（PreToolUse hook）
- Codex：`.codex/config.toml`（MCP）
- 这些宿主配置含本机绝对路径，已在 `.gitignore` 中排除；换机器后重新生成。
- 自检：`aoci --repo . doctor`

## 安全、隐私与合规

- 判分在一次性的 git worktree 中运行，只允许写入题目声明的源文件，带超时和资源限制，环境变量只传最少几项（不传任何令牌）。**这不是安全沙箱**：只挖你信任的仓库。
- Cortex 不读取任何登录凭证：`~/.claude/.credentials.json`、`~/.codex/auth.json`、`.env`、macOS 钥匙串。
- 学习样本入库前会脱敏（M2）。
- 用 Claude / Codex 的输出训练本地模型前，请自行核对 Anthropic 与 OpenAI 使用条款中的相关限制。Cortex 默认优先沉淀"项目约定和踩坑经验"类知识。

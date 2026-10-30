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

## D17 模型下载放在项目的 `models/` 目录 · 已定（2026-10-30）
取代 D10 中"模型放数据目录"的隐含设想。Cortex 启动时把 `HF_HUB_CACHE` 指向 `<项目根>/models/huggingface/hub`，因此项目以后下载的所有 Hugging Face 模型（包括 mlx-lm 的 LoRA 子进程）都落在这里。
- 只改模型缓存，不改 `HF_HOME`，所以已保存的 Hugging Face 登录令牌不受影响；用户自己导出的 `HF_HUB_CACHE` 优先。
- 可用环境变量 `CORTEX_MODELS_DIR` 或 `config/local.toml` 的 `[paths] models_dir` 覆盖。
- `models/` 已在 `.gitignore`。已下载的两个 M0 候选用 `scripts/move-models.sh` 搬入。
- 风险：若 `~/Documents` 开启了 iCloud「桌面与文档」同步，几十 GB 模型会被上传 iCloud，且未下载到本地的文件会让加载失败。处理：关闭该同步，或把目录改名为 `models.nosync`（iCloud 会忽略 `.nosync` 结尾的目录）并用 `CORTEX_MODELS_DIR` 指向它。

## D16 M0 首次实测结果（2026-10-30，M5 Max 128GB，mlx 0.32.4 / mlx-lm 0.32.2）
- **架构核实**：Qwen3.6-27B 为 64 层，模式 `(LLLA)x16`（48 层线性注意力 + 16 层全注意力），全部为稠密 MLP，8bit 权重 33.8GB。Devstral Small 2 为 40 层全注意力（`Ax40`），4bit 权重 14.2GB。与 D13 的资料一致。
- **两者都能被 stock mlx-lm 0.32.2 加载**，包括 Devstral 的 mlx-vlm 转换版本（D13 风险已消除）。
- **LoRA 探针重测（合成数据，16 层，rank 16，10 步，grad checkpoint）**：

  | | Qwen3.6-27B（8bit） | Devstral Small 2（4bit） |
  |---|---|---|
  | 可训练的最长序列 | 1024（2048 失败，峰值达 115.6GB） | ≥2048（未再往上探） |
  | 秒/步 | 10.06 | 4.4 |
  | tok/s | 77.5 | 361.5 |
  | 峰值内存 | 62.0GB（序列 1024） | 18.26GB（序列 2048） |

  结论（暂定）：Qwen3.6 的 LoRA 在真实样本长度下成本过高，且无法与推理并行；Devstral 余量充足。需要持续微调的底座暂定 Devstral Small 2，Qwen3.6 作推理对照，最终以 dev 题通过率为准（待挖题）。社区的 DeltaNet 训练加速补丁（mlx-lm PR #1389）未合并，不依赖。
- **Devstral tokenizer 警告**：mlx-community 转换版触发 "incorrect regex pattern" 警告，会导致分词不正确。已在配置中加 `tokenizer_config = { fix_mistral_regex = true }`；评测 Devstral 时必须带上。
- 注：`inspect` 记录的 load_seconds（5403 / 2519）包含首次下载时间，不是真实加载时间。

## D18 增加「读书/学习」能力：先用知识库，不靠微调灌知识 · 待定方案，M2 后实施（2026-10-30）
需求：导入教材 PDF / 电子文稿后，Cortex 像老师一样出学习计划、随时答疑，并在以后的相关提问中结合书中知识；大脑图要能看到读书时的变化。
- **知识进入模型的主路径是检索（RAG）**：解析 → 按章节切块 → 向量+全文索引（沿用 sqlite-vec，D7）→ 回答时引用原文页码。LoRA 微调不适合可靠地"记住"事实（容易记错、难更新、难删除），LoRA 只用于风格、解题套路等技能。
- **老师功能**：从目录/章节生成学习计划；每次学习前用间隔重复复习上次内容；自出题 + 批改；学习者档案记录掌握度（可在界面查看和修改）。
- **与编程共用同一套机制**：样本库/知识库、检索、调度器、评测。书籍也要有"评测"：从书里出题，比较有/无检索时的正确率。
- **大脑可视化**：读书视图显示检索命中与各层响应，数据来自真实检索与激活；目前的 3D 原型（Artifact）全部为示例信号。
- 需先核实：PDF 解析方案（扫描件需 OCR）、嵌入模型选择（本地，内存占用）、版权（只处理用户自己的文件，不上传）。

## D19 底座选定 Devstral Small 2 · 已定（2026-10-30）
由用户决定。依据 D16：LoRA 训练在 2048 长度下 4.4 s/步、峰值 18GB，可与推理并存；Qwen3.6-27B 在 1024 长度就需 62GB，且混合架构的层没有注意力矩阵。dev 通过率仍待挖题后补测，作为复核而非前提；若 Devstral 在真实任务上明显落后，再按 D13 的替换路径重选。Qwen3.6-27B 保留为推理对照候选，不再做训练探针。
可视化：Devstral 40 层全为标准注意力，大脑图和权重图按这个结构绘制。

## D20 日常操作全部在网页界面完成，不要求用户敲命令 · 已定（2026-10-30）
- **启动**：双击 `Cortex.command`（装依赖、启动本地服务、打开浏览器）。服务只监听本机，并校验 Host 头。
- **模型库**（`server/library.py`）：条目 = 配置里的 mlx 候选 + 用户按 Hugging Face 名添加的模型（存 `<数据目录>/library.json`）。下载在后台线程跑，进度 = 磁盘字节数 ÷ 仓库总大小；下载前检查磁盘空间；删除只允许在模型缓存目录内；未下载的模型不允许直接加载（避免无进度的隐式下载）。
- **训练向导**（`server/training.py`）：导入文件 → 剔除含密钥的样本、去重、按 9:1 拆出验证集 → 以子进程运行 `mlx_lm lora`（LoRA 作用于 q/o/gate/down 四个矩阵，与权重图一致）→ 实时解析 loss / tok/s / 峰值内存。训练前自动卸载当前模型；训练与聊天互斥。产物是 `<数据目录>/adapters/<模型>/<时间>/` 下的适配器，原模型文件不动。
- **真实的权重变化**：从适配器的 lora_a / lora_b 在权重图同一批 (行,列) 位置计算 ΔW = scale·(B·A)ᵀ，并用 ‖ΔW‖_F 表示每层的学习量，驱动网络层图的“训练”视图和 3D 立柱的发光。
- **未做**：适配器上线前的 dev/holdout 评测闸门、回归自动回滚、知识库（RAG）导入——属于 M2/M4。界面上已提示“适配器尚未评测，先小范围试用”。
- **未在硬件上验证**：下载（huggingface_hub 的 `tqdm_class`）、`mlx_lm lora` 输出解析的 val loss 行、适配器键名（`layers.N.<模块>.lora_a/b`）。首次真机运行时按报错调整。

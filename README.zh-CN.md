# Cortex

**看见本地大模型思考和学习的过程。** 在你的 Mac 上和模型聊天，生成时逐层看到每一层亮起；一键下载、切换模型；用向导训练 LoRA，并看到哪些权重真的变了。

简体中文 · [English](README.md)

> **状态：早期 alpha。** 工具、服务和界面已写好并通过单元测试（64 个）。模型加载和 LoRA 训练在一台机器上做过基准测试（MacBook Pro M5 Max，128 GB）。实时逐层活动探针和界面里的训练流程**还没有完整的真机运行记录**，可能有粗糙之处，欢迎提 issue。

## 它能做什么

- **和真实层数一致的 3D 视图。** 每个 Transformer 层是一圈发光节点，叠成一根全息立柱。圈数就是模型的真实层数（Devstral Small 2 是 40 层，Qwen3.6-27B 是 64 层，并标出线性注意力 / 全注意力的排列）。生成文字时，活动脉冲沿立柱上升。
- **神经网络层图。** 每层显示 4 个权重矩阵（注意力 Q / 输出、MLP 门控 / 下投影），各取 12×12 个位置；层与层之间的连线在两层同时活跃时发光。鼠标悬停任一格，看到精确的权重值和它在完整矩阵中的行列位置。
- **聊天与切换模型。** 本地的 OpenAI 兼容服务（`/v1/chat/completions`，支持流式）和内置聊天界面；编辑器等工具也可以用同一个接口。
- **模型库。** 一键下载（带进度）、加载、卸载、删除；粘贴 Hugging Face 上的 MLX 模型名即可添加新模型。模型保存在项目文件夹里。
- **向导式 LoRA 训练。** 拖入文件（`.jsonl`、`.md`、`.txt`、代码）→ 自动检查（密钥扫描、去重、拆分训练/验证集）→ 选择模型和强度 → 后台训练，实时显示 loss、速度和内存。结果是单独的适配器，不会修改原模型文件。
- **真实的权重变化。** 训练中和训练后，层图显示的 ΔW 由适配器在同一批矩阵位置上计算得出，可以看出哪些层学到了东西。

### 哪些是真实的，哪些是风格化的

| 展示的内容 | 来源 |
|---|---|
| 层数、层类型、每层大小 | 从加载的模型读取 |
| 悬停看到的权重值 | （反量化后）权重的精确采样，由 `scripts/dump_weights.py` 生成 |
| 生成时每层的活动 | 每个 block 上的探针：最后一个 token 位置的 `mean\|输出 − 输入\| / mean\|输入\|` |
| 训练 loss、tok/s、峰值内存 | 解析 `mlx_lm lora` 的输出 |
| ΔW 格子和每层学习量 | 由适配器的 `lora_a` / `lora_b` 计算 |
| 颜色、发光、全息风格、轨道环 | 风格化装饰 |

页面没有连接模型时（例如托管的预览页）会进入明确标注的**演示模式**，信号是模拟的。

## 快速开始（macOS，Apple Silicon）

```bash
brew install uv
git clone https://github.com/Yiteng-CHEN/cortex-mlx.git && cd cortex-mlx
```

然后**双击 `Cortex.command`**，或者：

```bash
uv sync --extra mlx
uv run --extra mlx cortex serve --open      # http://127.0.0.1:8765
```

在浏览器里：**模型库 → 下载**（Devstral Small 2 的 4-bit 版约 15 GB）→ **加载** → 开始聊天。对已下载的模型点**生成权重图**，层图就会显示真实权重值。点**训练**进入向导。

要求：Apple Silicon 的 macOS、[uv](https://docs.astral.sh/uv/)、足够的磁盘空间。推理 4-bit 24B 模型约需 15–20 GB 内存；Devstral Small 2 在序列长度 2048 下 LoRA 探针峰值约 18 GB（上述机器实测，其他机器未测试）。

## 目前的实测数据（M5 Max，128 GB，mlx 0.32.4 / mlx-lm 0.32.2）

| | Qwen3.6-27B（8-bit） | Devstral Small 2（4-bit） |
|---|---|---|
| 层结构 | 64 层 = 16 ×（3 线性 + 1 全注意力） | 40 层，全部为全注意力 |
| LoRA 能跑的最长序列 | 1024（2048 内存不足，峰值 115 GB） | ≥ 2048 |
| LoRA 秒/步 | 10.1 | 4.4 |
| LoRA 峰值内存 | 62 GB | 18 GB |

选择底座模型的依据见 [`docs/DECISIONS.md`](docs/DECISIONS.md)；评测、测试、AOCI 等开发细节见 [`docs/DEVELOPMENT.zh-CN.md`](docs/DEVELOPMENT.zh-CN.md)。

## 结构

```
web/index.html            单文件界面（three.js 全息立柱、层图、聊天、模型库、训练向导）
src/cortex/server/        本地服务：聊天接口、模型库、训练、权重/ΔW 接口
src/cortex/telemetry/     包裹 MLX Transformer 块的逐层探针
src/cortex/backends/      MLX 后端、模拟后端、OpenAI 兼容客户端
src/cortex/evals/         评测工具：从 git 历史挖题、隔离判分、对比
scripts/                  dump_weights.py、基准测试等
docs/                     项目简介、技术决定（D1–D20）、计划
```

服务只监听 `127.0.0.1`，并拒绝来源不是本机的 `Host` 头。Cortex 从不读取你的 Claude / Codex / Hugging Face 登录凭证。

## 路线图

已完成：评测与选型工具（M0）、本地服务、模型库、训练向导、全息立柱与层图。
接下来：观察你自己的编程过程（git 钩子、Claude Code / Codex 会话记录），沉淀为已验证、已脱敏的样本库；检索（RAG），让文档和书本可以导入；适配器上线前的影子对照和评测闸门；每层重要性（梯度、消融）视图。详见 [`PLAN.md`](PLAN.md)。

## 需要了解的限制

- 只支持 Apple Silicon（MLX）。同一时间只加载一个模型；训练和聊天不能同时进行。
- 微调适合学习风格和习惯，不适合可靠地记住事实。文档类知识计划通过检索来处理。
- 适配器目前不会在使用前自动评测。请小范围试用，效果变差就切回原模型。
- 下载 Hugging Face 模型需要网络；开发时用的沙箱环境连不上，所以下载流程只用模拟对象测试过。

## 相关项目

Cortex 关注的是**实时、本地、可训练**的模型。想了解 Transformer 的工作原理，可以看 [llm-viz](https://github.com/bbycroft/llm-viz) 和 [Transformer Explainer](https://github.com/poloclub/transformer-explainer)；做研究级的可解释性，可以看 [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) 和 [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis)。

## 参与贡献

欢迎提 issue 和 PR，尤其欢迎硬件报告（芯片、内存、模型、哪些能跑通）。运行测试：`uv run python -m unittest discover -s tests -t .`。

## 许可证

[MIT](LICENSE)

<div align="center">

# Neurolume

**Watch your model learn.**
Fine-tune an LLM on your Mac in a few clicks — and see, layer by layer, what training actually changed.

[![CI](https://github.com/Yiteng-CHEN/neurolume/actions/workflows/ci.yml/badge.svg)](https://github.com/Yiteng-CHEN/neurolume/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)
![License](https://img.shields.io/badge/license-MIT-blue)

**[▶ Try the online demo](https://yiteng-chen.github.io/neurolume/)** (no install; replays a real recorded run) · English · [简体中文](README.zh-CN.md)

</div>

<p align="center"><img src="docs/media/demo.gif" width="860" alt="Neurolume: Devstral Small 2 answers while the layer map lights up; the answer replay shows how confident the model was in each token and what else it considered; the logit lens shows every layer's guess; the Train view shows a real adapter's weight change"></p>
<p align="center"><sub>Real data, not a mock-up: one answer from Devstral Small 2 on an M5 Max (per-layer activity, logit lens, per-token confidence) and a real adapter's ΔW, replayed in the <a href="https://yiteng-chen.github.io/neurolume/">online demo</a> · <a href="docs/media/demo.mp4">video</a></sub></p>

Most fine-tuning tools stop at "training finished, loss = 1.23". Most interpretability tools show a model you cannot change. Neurolume puts both in one local app: **you train the model, and you watch its insides while you do it.**

> **Status: early alpha.** Server, UI and tooling are unit-tested (110+ tests, no GPU needed). Everything below runs on one machine so far (MacBook Pro M5 Max, 128 GB). Verified there with Devstral Small 2 and Qwen3.6-27B: live layer probe, logit lens (live, and computed afterwards for answers generated without it), per-token confidence, guided LoRA training (Devstral: 36 steps, 15 GB peak, validation loss 4.45 → 1.06), before/after comparison, MLX export and training-data generation. Ollama export is not hardware-tested yet. Reports from other machines are very welcome.

## What you can do

**Fine-tune without writing code**
- **Bring data, or make it.** Save good answers (or your corrected versions) from the chat with one click; drop in `.jsonl`, notes, code, or your ChatGPT / Claude / Claude Code chat history — or let the loaded model *generate* a dataset from your own files: Q&A pairs from documents, or "requests → your own code/writing" pairs that teach your style. You review and edit every item before training.
- **Guided LoRA training** with automatic checks (secret scan, de-duplication, train/validation split), live loss, speed and memory. The base model is never modified; the result is a separate adapter.
- **Did it work?** Validation loss *before → after* on data the model never trained on, a per-layer bar chart of how much each layer changed, and a **side-by-side comparison**: the same question answered by the original and the trained model (no reload — the adapter is switched off and on in place).
- **Take it anywhere.** Export the fused model as MLX (LM Studio loads it directly) or to **Ollama**.

**See inside the model**
- **Live brain view.** A 3D sphere or nebula with one band per real layer (Devstral Small 2: 40; Qwen3.6-27B: 64 with its linear/full-attention pattern). It discharges only while the model is generating or training.
- **Layer map.** Four weight matrices per layer with real weight values on hover, per-layer activity while generating, and the adapter's real ΔW while training.
- **Logit lens.** For each generated token, what every layer would predict "if the model stopped here" — watch the answer form layer by layer. Generated an answer with the view closed? Open it afterwards and it is computed for that answer in one extra pass, so normal chatting stays at full speed.
- **Answer replay.** One bar per token showing how confident the model was; the answer text is tinted the same way and hovering shows the alternatives it considered. Click any token and the brain and layer views go back to that moment.
- **"How to read this"** explanations for attention, MLP, linear attention, LoRA and the logit lens, in English and Chinese.

**Use it every day**
- Chat with an OpenAI-compatible local server (`/v1/chat/completions`, streaming), one-click model library, **knowledge base** (import books/docs for retrieval), opt-in **web search** for local models (also as an MCP server), and an **agent** that reads a project folder and proposes edits you approve one by one (with undo).

### Honest signals

Everything labelled as the model's activity comes from the model; simulated data is labelled. Full definitions in [`docs/SIGNALS.md`](docs/SIGNALS.md).

| Shown | Source |
|---|---|
| Layer count and types | Read from the loaded model |
| Weight values on hover | Exact strided sample of the dequantized weights |
| Per-layer activity | `mean\|out − in\| / mean\|in\|` of each block at the token being generated; raw value on hover |
| Logit lens | The model's own final norm + output head applied to intermediate layers |
| Per-token confidence and alternatives | The probabilities mlx-lm already computes while generating (no extra pass) |
| Brightness of the links between layers | Geometric mean of the two layers' activity; which matrix each thin line joins is illustrative |
| ΔW, per-layer change | Computed from the adapter (`scale · B·A`) |
| Validation loss before → after | mlx-lm's evaluation before the first update and at the end |
| Discharge paths on the sphere, colours, glow | Stylized (their frequency follows GPU/memory load and layer activity) |

## Tour of the interface

The window is split in two: **chat on the right**, and on the left **the brain on top, with the layer window and the answer replay below it**.

### 1. Chat (right)
- First row: pick a model; green “Live · real model” means the data comes from the running model, yellow “demo” means simulated. After training you also get an **adapter** dropdown (“no adapter” is the base model) and a **Compare with base** button.
- Toolbar: **Models** (download, load) · **Train** (the wizard) · **Web search** (the gear opens search settings) · **Knowledge base** · **中文/English** · **Agent** · 🗑 clear chat.

### 2. Brain (top left)
- Two shapes: **nebula** and **sphere** (toggle bottom-right). Each height band / ring is one layer, numbered on the left; Qwen3.6's full-attention layers are marked ◆.
- It only fires while the model is generating or training. Layer brightness comes from per-layer activity and the firing rate follows GPU and memory load; the paths, colours and glow of the sparks on the surface are stylised.
- Bottom line: GPU load, memory, model size, generation speed (tok/s).

### 3. Layer window (bottom left): four views
| View | What you see |
|---|---|
| **Inference** | One column per layer, four matrices per layer: attention Q, attention output, MLP gate, MLP down-projection; each cell is a real weight from that matrix (evenly spaced sample). While generating, a layer is brighter the more it changes the token's representation. The bundle of links between two neighbouring layers is lit by the geometric mean of the two, so **it is thick and bright only when both layers are busy, and faint if either is quiet**. Hover a cell for the real weight and the layer's raw measurement. |
| **Train** | The adapter's real ΔW (green = weight increased, orange-red = decreased); the bars below are the total change per layer, and you can pick any trained adapter to review. LoRA trains only the last 16 layers by default, so unchanged early layers are expected. |
| **Weights** | The static weights: orange = positive, cyan = negative, intensity = \|w\| ÷ the matrix's std (capped at 3σ). |
| **Logit lens** | A table: one column per generated token, one row per layer showing what that layer would output “if the model stopped here”; darker = more confident. A green underline means the layer already agrees with the final output; the “Settled” row says from which layer on. ◇ is a byte fragment of a multi-byte character. Choose which layers to show: all / second half / last 16 / a custom range and step. If the view was closed while the answer was generated, switching to it computes the lens for that answer afterwards. |

### 4. Answer replay (below the layer window)
- **One bar per token**: taller = the model hesitated more on it (1 − the probability it gave that token). Colour: cyan = confident, yellow = so-so, orange-red = unsure. The white line is a smoothed trend.
- **The answer text, tinted the same way**, so hesitant words stand out; unsure ones get a wavy underline. **Hover** to see what else it considered, e.g. “生命” at 32 % with “爱与” 29 % and “深刻” 15 % as alternatives.
- **Click any bar or word** and the brain and layer map freeze at the moment that token was generated; drag the slider or press “▶ Replay” to step through.
- **Select part of an AI answer in the chat** and the replay shows only that part; “Latest answer” goes back.
- In the logit-lens view the bars show the layer where each token settled instead (taller = decided later), on an axis covering only the layers that actually occur.

### 5. Training wizard
**Collect as you go**: every answer in the chat has “Save as training sample” and “Edit, then save…”. Save good answers as they are; fix the others into what you wanted first, and the model learns the fixed version.

Import data (samples saved from chat, `.jsonl`, notes, code, **`conversations.json` exported from ChatGPT or Claude, Claude Code session `.jsonl` files**, or have the model generate pairs from your files; sources can be combined into one dataset) → automatic checks (secrets removed, duplicates dropped, train/validation split) → pick a preset and train (live loss, speed, memory) → results: validation loss before → after, per-layer ΔW, side-by-side comparison with the base model → export as an MLX model or to Ollama.

### 6. Everything else
- **Knowledge base**: import books and documents; the model retrieves relevant passages before answering and cites them.
- **Web search**: when on, the model searches and reads pages when it needs to (your question goes to the search service); when off, Neurolume sends nothing out.
- **Agent**: pick a project folder; the model reads files and proposes changes one by one, each needing your approval, all undoable.
- **Local API**: `http://127.0.0.1:8765/v1/chat/completions`, OpenAI-compatible, so other tools can use the local model.

## FAQ

**I clicked a token in the replay. Why are the links between some layers thick and bright and others faint?**
The layer map is frozen at the moment that token was generated. Each layer's brightness is how much it changed the token's internal representation, compared both with that layer's typical level in this answer and with the other layers for this token. The bundle between two layers takes the geometric mean, so a bright stretch means several consecutive layers were heavily rewriting this token, and a faint stretch means those layers mostly passed it along. It is a real measurement, but **not “importance”**: changing a lot does not mean contributing a lot to the answer. The overall pattern also tends to look similar from token to token (overall activity varied only about ±10 % on real hardware); what really separates tokens is the confidence in the replay bars. Which of the 16 thin lines in a bundle joins which two matrices, and how the dots move, is illustrative; the measurement is per layer.

**Why are the early layers in the logit lens mostly English or gibberish?**
Early layers are still working out what each token is and where it sits; reading a next-token prediction off them is not meaningful yet. Most training text is English, so the intermediate “drafts” often surface in English. Later layers get closer to the final output.

**Why did training only change the later layers?**
LoRA attaches adapters to the last 16 layers by default and leaves the rest untouched, on purpose: less memory and less risk of breaking the model.

**Can fine-tuning make the model remember facts?**
It is better at style and habits (format, tone, a fixed way of answering). For reliable answers about your documents, use the knowledge base.

## Quick start (macOS, Apple Silicon)

One line, no clone needed:

```bash
uvx --from "neurolume[mlx] @ git+https://github.com/Yiteng-CHEN/neurolume" neurolume serve --open
```

Or from a checkout (double-click `Neurolume.command`, or):

```bash
git clone https://github.com/Yiteng-CHEN/neurolume.git && cd neurolume
uv run --extra mlx neurolume serve --open      # http://127.0.0.1:8765
```

Need [uv](https://docs.astral.sh/uv/) (`brew install uv`). In the browser: **Models → Download** (Devstral Small 2 4-bit ≈ 15 GB) → **Load** → chat; **Train** opens the wizard. Without a Mac you can still open `src/neurolume/web/index.html` to see the UI in demo mode.

Data lives in `~/Library/Application Support/Neurolume`; models download into the checkout's `models/` folder (or the data folder when installed with `uvx`).

## Measured so far (M5 Max, 128 GB, mlx 0.32.4 / mlx-lm 0.32.2)

| | Qwen3.6-27B (8-bit) | Devstral Small 2 (4-bit) |
|---|---|---|
| Layers | 64 = 16 × (3 linear + 1 full attention) | 40, all full attention |
| LoRA, longest sequence that fit | 1024 (2048 ran out of memory, peak 115 GB) | ≥ 2048 |
| LoRA seconds / step | 10.1 | 4.4 |
| LoRA peak memory | 62 GB | 18 GB |

## Limits

- Apple Silicon only (MLX). One model at a time; training and chat don't run together.
- Fine-tuning teaches style and habits better than facts — use the knowledge base for documents.
- Logit lens: with its view open it is computed while generating (measured on Qwen3.6-27B: about 10.8 → 9.1 tok/s); with the view closed generation is not slowed, and switching to the view afterwards computes it for that answer in one extra pass (295 tokens in about 4 s). The server keeps the last 12 answers; answers from before a model reload cannot be backfilled.
- Ollama export dequantizes the weights first (≈ 4× the size) and depends on Ollama supporting the architecture.

## How it is built

```
src/neurolume/web/         UI: index.html, css/, js/ (plain scripts, no build step)
src/neurolume/server/      local server: chat, library, training, data generation, export, knowledge base, agent
src/neurolume/telemetry/   per-layer probe and logit lens on MLX transformer blocks
src/neurolume/backends/    MLX backend, mocks, OpenAI-compatible client
src/neurolume/evals/       eval tooling: mine tasks from git history, sandboxed grading
docs/                      SIGNALS.md, decisions, plan
```

The server binds to `127.0.0.1` only. Neurolume never reads Claude / Codex / Hugging Face credentials.

## Contributing

Issues and pull requests are welcome — hardware reports (chip, memory, model, what worked) are especially valuable. See [CONTRIBUTING.md](CONTRIBUTING.md). Tests: `PYTHONPATH=src python -m unittest discover -s tests -t .`

## Related work

[M-Courtyard](https://discuss.huggingface.co/t/m-courtyard-fine-tune-llms-on-macos-with-zero-code-mlx-ollama-gui/173463) and [MLX-LoRA-Studio](https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio) (no-code MLX fine-tuning); [llm-viz](https://github.com/bbycroft/llm-viz) and [Transformer Explainer](https://github.com/poloclub/transformer-explainer) (how transformers work); [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [llm-transparency-tool](https://github.com/facebookresearch/llm-transparency-tool) (research interpretability). Neurolume's focus is the overlap: a live, local model you train and inspect at the same time.

## License

[MIT](LICENSE).

# Cortex

**See inside a local LLM while it thinks and learns.** Chat with a model on your Mac, watch every layer light up as it generates, download and swap models with one click, and fine-tune it with LoRA through a guided wizard, while seeing which weights actually change.

[简体中文](README.zh-CN.md) · English

> **Status: early alpha.** The tooling, server and UI are written and unit-tested (64 tests). Model loading and LoRA training were benchmarked on one machine (MacBook Pro M5 Max, 128 GB). The live layer-activity probe and the in-UI training flow have **not yet had a full hardware run**. Expect rough edges and please open issues.

## What it does

- **Layer-accurate 3D view.** One ring of nodes per transformer layer, stacked into a holographic column. The number of rings is the model's real layer count (Devstral Small 2: 40, Qwen3.6-27B: 64, with its linear-attention / full-attention pattern marked). Activity pulses climb the column as tokens are generated.
- **Neural-network map.** Every layer shows four weight matrices (attention Q / output, MLP gate / down) as 12×12 grids, with the connections between layers drawn and lit when both layers are active. Hover any cell to read the exact weight value and its row/column in the full matrix.
- **Chat + model switching.** A local OpenAI-compatible server (`/v1/chat/completions`, streaming) with a built-in chat UI. Your editor or other tools can use the same endpoint.
- **Model library.** One-click download with progress, load, unload and delete. Paste any MLX model name from Hugging Face to add it. Models live inside the project folder.
- **Guided LoRA training.** Drop in files (`.jsonl`, `.md`, `.txt`, code) → automatic checks (secret scan, de-duplication, train/validation split) → pick model and intensity → train in the background with live loss, speed and memory. The result is a separate adapter; the base model files are never modified.
- **Real weight changes.** After (and during) training, the map shows ΔW computed from the adapter at the same matrix positions, so you can see which layers learned.

### What is real and what is stylized

| Shown | Source |
|---|---|
| Layer count, layer types, per-layer size | Read from the loaded model |
| Weight values on hover | Exact sample of the (de-quantized) weights, via `scripts/dump_weights.py` |
| Per-layer activity while generating | `mean\|block_out − block_in\| / mean\|block_in\|` at the last token position, measured by a probe on each block |
| Training loss, tokens/s, peak memory | Parsed from `mlx_lm lora` output |
| ΔW cells and per-layer learning amount | Computed from the adapter's `lora_a` / `lora_b` |
| Colours, glow, hologram styling, orbit rings | Stylized |

When the page is not connected to a model (for example the hosted preview) it runs in a clearly labelled **demo mode** with simulated signals.

## Quick start (macOS, Apple Silicon)

```bash
brew install uv
git clone https://github.com/Yiteng-CHEN/cortex-mlx.git && cd cortex-mlx
```

Then either **double-click `Cortex.command`**, or:

```bash
uv sync --extra mlx
uv run --extra mlx cortex serve --open      # http://127.0.0.1:8765
```

In the browser: open **Model library → Download** (Devstral Small 2 4-bit is about 15 GB) → **Load** → chat. Click **Generate weight map** on a downloaded model to get real weight values in the layer map. Click **Train** for the wizard.

Requirements: macOS on Apple Silicon, [uv](https://docs.astral.sh/uv/), enough free disk for models. Inference of a 4-bit 24B model needs about 15–20 GB of memory; the LoRA probe peaked at about 18 GB for Devstral Small 2 at sequence length 2048 (measured on the machine above; other machines untested).

## Measured so far (M5 Max, 128 GB, mlx 0.32.4 / mlx-lm 0.32.2)

| | Qwen3.6-27B (8-bit) | Devstral Small 2 (4-bit) |
|---|---|---|
| Layers | 64 = 16 × (3 linear + 1 full attention) | 40, all full attention |
| LoRA, longest sequence that fit | 1024 (2048 ran out of memory, peak 115 GB) | ≥ 2048 |
| LoRA seconds / step | 10.1 | 4.4 |
| LoRA peak memory | 62 GB | 18 GB |

Details and the reasoning behind the choice of base model are in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## How it is built

```
web/index.html            single-file UI (three.js hologram, layer map, chat, library, wizard)
src/cortex/server/        local server: chat API, model library, training, weight/ΔW endpoints
src/cortex/telemetry/     per-layer probe that wraps MLX transformer blocks
src/cortex/backends/      MLX backend, mock backends, OpenAI-compatible client
src/cortex/evals/         eval tooling: mine tasks from git history, sandboxed grading, compare
scripts/                  dump_weights.py, benchmarks, helpers
docs/                     project brief, decisions (D1–D20), plan
```

The server binds to `127.0.0.1` only and rejects foreign `Host` headers. Cortex never reads your Claude / Codex / Hugging Face credentials.

## Roadmap

Done: eval and model-selection tooling (M0), local server, model library, training wizard, hologram and layer map.
Next: observe your own coding sessions (git hooks, Claude Code / Codex logs) into a verified, de-identified sample library; retrieval (RAG) so documents and books can be imported; shadow comparison and an eval gate before an adapter goes live; per-layer importance (gradient, ablation) views. See [`PLAN.md`](PLAN.md).

## Limits to know about

- Apple Silicon only (MLX). One model loaded at a time; training and chat do not run together.
- Fine-tuning is good for style and habits, not for reliably storing facts. Document knowledge is planned to go through retrieval instead.
- Adapters are not yet evaluated automatically before use. Try them on a small scale and switch back to the base model if they behave worse.
- Hugging Face downloads need network access; the sandboxed test environment used during development could not reach it, so the download path is tested with fakes only.

## Related work

Cortex focuses on a *live, local, trainable* model. For explaining how transformers work, see [llm-viz](https://github.com/bbycroft/llm-viz) and [Transformer Explainer](https://github.com/poloclub/transformer-explainer). For research-grade interpretability, see [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis).

## Contributing

Issues and pull requests are welcome, especially hardware reports (chip, memory, model, what worked). Run the tests with `uv run python -m unittest discover -s tests -t .`.

## License

[MIT](LICENSE)

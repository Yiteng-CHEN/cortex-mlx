# Contributing to Neurolume

Thanks for helping! Neurolume aims to make fine-tuning approachable **and** to show, honestly, what happens inside a model. Contributions of every size are welcome — a typo, a translation fix, a new visualization, support for another model.

[中文说明见下方](#中文)

## Project layout

| Path | What lives there |
|---|---|
| `src/neurolume/server/` | Local HTTP server: chat (`app.py`), training (`training.py`), model library, knowledge base (`books.py`), agent (`agent.py`), web search |
| `src/neurolume/backends/` | Model backends. `mlx_backend.py` loads models with MLX and `mlxutil.py` attaches the per-layer probe |
| `src/neurolume/web/` | The UI: `index.html`, `css/app.css`, and plain scripts in `js/` (no build step) |
| `src/neurolume/defaults.toml` | Default configuration and the model catalog |
| `tests/` | Standard-library `unittest` tests. They run without a GPU or a model |
| `docs/` | Design notes, decisions (`DECISIONS.md`), how the signals are measured (`SIGNALS.md`) |

The front-end scripts in `js/` are classic scripts loaded in order and share globals: `i18n → state → holo → net → chat → panels → main`. Keep top-level code that runs at load time in the file that defines what it calls, or in `main.js`.

## Running it

```bash
git clone https://github.com/Yiteng-CHEN/neurolume && cd neurolume
uv run --extra mlx neurolume serve --open      # Apple Silicon
PYTHONPATH=src python -m unittest discover -s tests -t .   # tests, any OS
```

Without a Mac you can still work on the UI: open `src/neurolume/web/index.html` directly in a browser — it runs in demo mode with simulated signals.

## Ground rules

- **Honest signals.** Anything shown as "the model's activity" must come from the model. Simulated or sample data must be labelled as such in the UI. See `docs/SIGNALS.md`.
- **Safety first for the agent.** File access stays inside the chosen folder; nothing is written without the user's approval.
- **Bilingual UI.** User-facing strings are written in Chinese in the source and translated through the table in `js/i18n.js`. When you add a string, add its English entry there.
- **No new runtime dependencies in the core.** The server is standard-library Python; MLX is the only optional extra.
- Keep pull requests focused; include a screenshot or GIF for visual changes.

## Good first issues

Look for the `good first issue` label, or pick one of these:

- Add the layer layout of another model family to `KNOWN` in `js/state.js`
- Improve an English translation in `js/i18n.js`
- Add a teaching note for a layer type (see the "Learn" panel)

---

## 中文

欢迎任何形式的贡献：改错字、改翻译、加新的可视化、支持新的模型都可以。

- 代码结构见上表。前端没有构建步骤，`js/` 里的脚本按顺序加载、共用全局变量。
- 运行：`uv run --extra mlx neurolume serve --open`；测试：`PYTHONPATH=src python -m unittest discover -s tests -t .`（不需要显卡和模型）。
- 没有 Mac 也能改界面：直接用浏览器打开 `src/neurolume/web/index.html`，会以演示模式运行。
- 规则：界面上标注为“模型活动”的数据必须来自模型，模拟数据要明确标出；Agent 只能访问用户选定的文件夹，写入前必须经用户批准；新增界面文字要在 `js/i18n.js` 里补英文；核心服务只用 Python 标准库。

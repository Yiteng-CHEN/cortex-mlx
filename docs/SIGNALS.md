# What the visuals measure / 界面上的信号是怎么来的

Neurolume tries to be honest about every pixel. This page says exactly what each view shows and where the numbers come from.
（中文在后半部分）

## Per-layer activity (brain view, "Inference" layer view)

For every generated token, each transformer block is wrapped by a probe (`src/neurolume/telemetry/probe.py`).
At the **last position** (the token being produced) it records

```
activity[l] = mean(|out_l - in_l|) / mean(|in_l|)
```

— how much block *l* changed the residual stream relative to its size. It is a real measurement from the running
model. It is **not** a claim that a layer is "important" for the answer.

**Display modes** (button "View: enhanced / raw"):

- **Raw** — `activity / recent maximum`, linear. Proportions between layers are preserved.
- **Enhanced** (default) — mixes each layer's value relative to *its own* recent range with its value relative to the
  other layers in the same step, then squares it. This makes changes visible (otherwise the first and last layers
  dominate) but exaggerates differences. Use *raw* when you want to compare magnitudes.

Hover a tile to see the measured value.

## Brain view (sphere / nebula)

- Each ring / height band is one layer, in order (bottom = first layer). The layer ruler on the left is exact.
- Golden "discharges" only appear while the model is generating or training. Their frequency follows the machine's
  GPU / memory load (read without sudo: `ioreg`, `vm_stat`) and the per-layer activity above; their paths are
  generated for aesthetics. **Where a discharge travels across the sphere is not a measurement.**
- When nothing is running the brain is completely calm.

## Logit lens ("Logit lens" view)

For ~16 evenly spaced layers plus the last one, the hidden state at the last position is passed through the model's
own final norm and output head (nostalgebraist, 2020). The cell shows the most likely token and its probability.
The last layer's row equals the real prediction. Off unless the view is open, because it costs about one extra
output-head matmul per lens layer per token.

## Training views

- **ΔW (Train view, layer bars)** — read from the adapter file: for LoRA matrices `ΔW = scale · B·A`. Tiles show
  sampled entries at the same (row, col) positions as the weight map; the bars show `‖ΔW‖_F` per layer.
- **Validation loss before → after** — mlx-lm evaluates the validation split once before the first update and again
  at the end; both numbers come from its log.
- **Weight map** — an exact strided 12×12 sample of each matrix's dequantized weights.

## What is simulated

- The demo page (no local service) and the `demo-chat` backend simulate everything; the UI marks this as
  *simulated*.
- The "Train one step (sample)" button shows a made-up ΔW until a real adapter exists.

---

## 中文

- **逐层活动**：每生成一个字，在每一层的最后一个位置测量 `mean(|输出-输入|) / mean(|输入|)`，即这一层把内部表示改了多少。这是真实测量值，不代表这一层“重要”。
  - “原始”显示：测量值除以最近的最大值，按比例显示；
  - “增强”显示（默认）：结合每层相对自己的变化和相对其他层的差别再平方，便于看出变化，但会放大差异。
- **脑图（球体 / 星云）**：每一圈 / 每段高度对应一层，左侧刻度准确。金色放电只在模型生成或训练时出现，频率跟随显卡/内存占用和逐层活动；**放电在球面上的走向是为了美观生成的，不是测量值**。
- **逐层预测（logit lens）**：把约 16 个层的中间结果送进模型自己的最终归一化和输出层，显示最可能的字和概率；最后一层就是真实输出。只在打开该视图时计算。
- **训练视图**：ΔW 直接从适配器文件计算（`scale·B·A`）；柱状图是每层 `‖ΔW‖`；“验证集损失 训练前 → 训练后”来自 mlx-lm 训练日志。
- **模拟数据**：演示页面和 demo 后端的所有信号都是模拟的，界面会标注“模拟”；“训练一步（示例）”按钮在没有真实适配器前显示的是示例 ΔW。

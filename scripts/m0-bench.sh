#!/usr/bin/env bash
# M0 benchmark — run on the Mac (Apple Silicon), from anywhere:
#   scripts/m0-bench.sh                      # both default candidates
#   scripts/m0-bench.sh qwen3.6-27b          # just one
# First run downloads model weights (~29 GB + ~15 GB) into the Hugging Face cache.
# Each step runs in its own process, so only one model is in memory at a time.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "需要在 Apple Silicon Mac 上运行 (MLX)"; exit 1
fi
command -v uv >/dev/null || { echo "需要 uv：brew install uv"; exit 1; }

if [[ $# -gt 0 ]]; then CANDIDATES=("$@"); else CANDIDATES=(qwen3.6-27b devstral-small-2); fi

uv sync --extra mlx || { echo "uv sync 失败"; exit 1; }
neurolume() { uv run --extra mlx neurolume "$@"; }

DATA_DIR="$(neurolume sys config | sed -n 's/^data_dir: //p')"
mkdir -p "$DATA_DIR/logs"
LOG="$DATA_DIR/logs/m0-bench-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log: $LOG"
uv run --extra mlx python -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__, '| mlx-lm', mlx_lm.__version__)"

DEV_TASKS="$(neurolume eval list | awk '$1=="dev"{print $2}')"

for c in "${CANDIDATES[@]}"; do
  echo; echo "================ $c ================"
  if ! neurolume sys mem --candidate "$c"; then echo "跳过 $c：可用内存不足"; continue; fi

  echo "-- inspect"
  if ! neurolume models inspect --candidate "$c"; then
    echo "!! $c 无法用 mlx-lm 加载（见上方报错）。参见 docs/DECISIONS.md D13 的备选方案。"; continue
  fi

  if [[ "${DEV_TASKS:-0}" -gt 0 ]]; then
    echo "-- eval run (dev, $DEV_TASKS tasks)"
    neurolume eval run --candidate "$c" --split dev || echo "!! eval run 失败"
  else
    echo "-- eval run 跳过：还没有 dev 题（先 mine + review）"
  fi

  echo "-- train-probe"
  neurolume sys mem --candidate "$c" && neurolume models train-probe --candidate "$c" || echo "!! train-probe 失败"
done

echo; echo "================ compare ================"
if [[ "${DEV_TASKS:-0}" -gt 0 ]]; then neurolume eval compare --split dev; fi
echo; echo "结果文件在 $DATA_DIR/results/ ；完整日志 $LOG"

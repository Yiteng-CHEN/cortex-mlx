#!/usr/bin/env bash
# Move already-downloaded Hugging Face models into this project's models/ folder (D17).
#   scripts/move-models.sh            # dry run: shows what would move
#   scripts/move-models.sh --apply    # actually move
# Only the two M0 candidates are moved; other models in ~/.cache/huggingface are left alone.
# 新版 huggingface_hub 的 blobs/ 里是指向共享存储 hub/blobs/xx/ 的符号链接，
# 所以先把链接指向的真实文件换进模型目录（同盘改名），再搬目录。
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
DST="$(pwd)/models/huggingface/hub"
APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1
REPOS=(models--unsloth--Qwen3.6-27B-MLX-8bit models--mlx-community--Devstral-Small-2-24B-Instruct-2512-4bit)

echo "from: $SRC"; echo "to:   $DST"
[[ "$SRC" == "$DST" ]] && { echo "已经在目标位置"; exit 0; }
(( APPLY )) || echo "(dry run — 加 --apply 才会真正移动)"
(( APPLY )) && mkdir -p "$DST"

for r in "${REPOS[@]}"; do
  if [[ ! -d "$SRC/$r" ]]; then echo "skip   $r (源位置没有)"; continue; fi
  if [[ -e "$DST/$r" ]]; then echo "skip   $r (目标已存在，不覆盖)"; continue; fi
  size="$(du -shL "$SRC/$r/blobs" 2>/dev/null | cut -f1)"     # -L：跟随链接，得到真实大小
  kb="$(du -skL "$SRC/$r/blobs" 2>/dev/null | cut -f1)"; kb="${kb:-0}"
  if (( kb < 1048576 )); then echo "skip   $r (跟随链接后仍只有 ${size:-?}，权重可能不全，先检查)"; continue; fi
  if (( APPLY )); then
    # 1) 把 blobs/ 里的符号链接替换成真实文件
    for f in "$SRC/$r/blobs"/*; do
      if [[ -L "$f" ]]; then
        real="$(python3 -c 'import os,sys;print(os.path.realpath(sys.argv[1]))' "$f")"
        [[ -f "$real" ]] || { echo "错误：$f 的目标不存在：$real"; exit 1; }
        rm "$f" && mv "$real" "$f"
      fi
    done
    # 2) 搬整个目录
    mv "$SRC/$r" "$DST/$r" && echo "moved  $r ($size)"
  else
    echo "would move $r ($size)"
  fi
done
(( APPLY )) && { echo; du -shL "$DST"/models--*/blobs 2>/dev/null || true; }
exit 0

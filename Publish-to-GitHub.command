#!/bin/bash
# 双击发布：在你的 GitHub 账号下创建公开仓库并上传整个项目（不含模型、数据、本机配置）。
cd "$(dirname "$0")" || exit 1
for p in /opt/homebrew/bin /usr/local/bin; do [ -x "$p/gh" ] && export PATH="$p:$PATH"; done
pause() { echo; read -n 1 -s -r -p "按任意键关闭这个窗口"; }
command -v git >/dev/null || { echo "需要 git。"; pause; exit 1; }
if ! command -v gh >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then echo "正在安装 GitHub 命令行工具 gh …"; brew install gh || { pause; exit 1; }
  else echo "请先在终端运行：brew install gh"; pause; exit 1; fi
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "需要登录 GitHub：接下来会打开浏览器，按提示授权即可。"
  gh auth login --web --git-protocol https || { pause; exit 1; }
fi
LOGIN=$(gh api user --jq .login) || { pause; exit 1; }
read -r -p "仓库名（直接回车使用 cortex-mlx）: " NAME; NAME=${NAME:-cortex-mlx}
# 把 README 里的占位用户名换成真实账号
for f in README.md README.zh-CN.md; do sed -i '' -e "s#<you>#$LOGIN#g" -e "s#<你的用户名>#$LOGIN#g" "$f"; done
echo; echo "即将在账号 [$LOGIN] 下创建【公开】仓库 [$NAME]，并上传下面这些文件："
git add -A
git status --short | sed 's/^/   /' | head -80
echo; read -r -p "确认公开发布？输入 y 继续，其他键取消: " OK
[ "$OK" = "y" ] || { echo "已取消，什么都没有上传。"; pause; exit 0; }
if ! git diff --cached --quiet; then
  git commit -q -m "Cortex: local LLM visualization, model library and guided LoRA training

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Ld6f8s7BJrkYsMFeuRRCLw" || { pause; exit 1; }
fi
git branch -M main
DESC="See inside a local LLM while it thinks and learns: layer-accurate 3D view, model library and guided LoRA training on Apple Silicon (MLX)."
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main || { pause; exit 1; }
else
  gh repo create "$NAME" --public --source=. --remote=origin --push --description "$DESC" || { pause; exit 1; }
fi
gh repo edit "$LOGIN/$NAME" --add-topic llm --add-topic mlx --add-topic apple-silicon --add-topic visualization \
  --add-topic lora --add-topic fine-tuning --add-topic local-llm --add-topic interpretability --add-topic threejs >/dev/null 2>&1
URL="https://github.com/$LOGIN/$NAME"
echo; echo "已发布：$URL"
open "$URL" 2>/dev/null
pause

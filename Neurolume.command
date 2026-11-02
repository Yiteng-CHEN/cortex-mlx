#!/bin/bash
# 双击这个文件即可启动 Neurolume（会在浏览器里打开界面）。
cd "$(dirname "$0")" || exit 1
for p in /opt/homebrew/bin /usr/local/bin "$HOME/.local/bin" "$HOME/.cargo/bin"; do
  [ -x "$p/uv" ] && export PATH="$p:$PATH"
done
pause() { echo; read -n 1 -s -r -p "按任意键关闭这个窗口"; }
if ! command -v uv >/dev/null 2>&1; then
  echo "还没有安装 uv。请打开“终端”运行：brew install uv    然后再双击这个文件。"; pause; exit 1
fi
echo "正在准备 Neurolume（第一次会安装依赖，需要几分钟）…"
if ! uv sync --extra mlx; then echo; echo "依赖安装失败，请把上面的信息发给我。"; pause; exit 1; fi
echo "启动中。关闭这个窗口或按 Ctrl+C 即可停止 Neurolume。"
uv run --extra mlx neurolume serve --open
status=$?
[ $status -ne 0 ] && { echo; echo "Neurolume 异常退出（代码 $status）。如果提示端口被占用，说明已经有一个 Neurolume 在运行：直接打开 http://127.0.0.1:8765"; pause; }
exit $status

"""Chat backends used by the server: the real MLX one, and a clearly-labelled demo."""
from __future__ import annotations

import math
import time
from typing import Any, Iterator

DEMO_REPLY = (
    "这是演示回复：当前没有加载真实模型，所以回答和各层的活动都是模拟出来的。\n\n"
    "在启动服务并选择 Devstral Small 2 后，这里会显示它真实生成的文字，"
    "大脑图和网络层图会用真实的逐层活动着色。\n\n"
    "```python\ndef add(a, b):\n    return a + b\n```"
)


class DemoChat:
    """Streams a canned reply with a synthetic layer sweep. simulated=True is sent to the UI."""
    simulated = True

    def __init__(self, name: str = "demo-chat", num_layers: int = 40, delay: float = 0.03):
        self.name = name
        self.num_layers = num_layers
        self._delay = delay

    def stream_chat(self, messages: list[dict], *, max_tokens: int = 1024,
                    temperature: float = 0.2) -> Iterator[dict[str, Any]]:
        words = DEMO_REPLY.replace("\n", " \n ").split(" ")
        n = self.num_layers
        for k, w in enumerate(words[:max_tokens]):
            centre = (k * 3.1) % (n + 8) - 4
            energy = [round(max(0.0, math.exp(-((i - centre) / 3.0) ** 2)) * (0.6 + 0.4 * ((i * 7 + k) % 5) / 4), 4)
                      for i in range(n)]
            if self._delay:
                time.sleep(self._delay)
            yield {"text": w + ("" if w == "\n" else " "), "energy": energy, "tps": None}

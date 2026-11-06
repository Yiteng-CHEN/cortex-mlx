"""In-process MLX backend (Apple Silicon only).

UNVERIFIED on hardware: follows the mlx-lm API (load / stream_generate / make_sampler and the
GenerationResponse fields text, generation_tokens, generation_tps, peak_memory). The first
real run is `scripts/m0-bench.sh`; fix this file if the installed mlx-lm differs.
"""
from __future__ import annotations

import time
from typing import Any

from . import mlxutil
from .base import Generation


class MLXBackend:
    def __init__(self, model: str, name: str | None = None,
                 chat_template_kwargs: dict[str, Any] | None = None,
                 tokenizer_config: dict[str, Any] | None = None,
                 memory_limit_gb: float | None = None,
                 probe: bool = False, adapter_path: str | None = None):
        if memory_limit_gb:
            mlxutil.set_memory_limit_gb(memory_limit_gb)
        self.name = name or model
        self._template_kwargs = dict(chat_template_kwargs or {})
        self._model, self._tokenizer = mlxutil.load_model(model, tokenizer_config, adapter_path)
        # LoRA layers must be collected before the probe wraps the blocks (the wrapper is not an nn.Module).
        self._lora = mlxutil.lora_modules(self._model) if adapter_path else []
        self._lora_scale = [m.scale for m in self._lora]
        self.has_adapter = bool(self._lora)
        self._probe = None
        if probe:
            from ..telemetry.probe import LayerProbe
            self._probe = LayerProbe(mlxutil.find_layer_container(self._model))
        self.num_layers = self._probe.n if self._probe else None
        self._lens_fn = None
        self._adapter_on = True
        # recent answers (token ids) so the logit lens can be computed afterwards, see lens_backfill
        from collections import OrderedDict
        self._gens: "OrderedDict[int, dict]" = OrderedDict()
        self._gid = 0

    # ---- adapter on/off without reloading (for before/after comparisons)
    def set_adapter_enabled(self, on: bool) -> None:
        for m, s in zip(self._lora, self._lora_scale):
            m.scale = s if on else 0.0
        self._adapter_on = on

    # ---- logit lens
    def _lens(self):
        if self._lens_fn is None:
            norm, head = mlxutil.find_output_head(self._model)
            import mlx.core as mx

            def fn(h):
                logits = head(norm(h)).astype(mx.float32)
                probs = mx.softmax(logits, axis=-1)
                ids = mx.argmax(probs, axis=-1)
                return ids, mx.take_along_axis(probs, ids[:, None], axis=-1)[:, 0]
            self._lens_fn = fn
        return self._lens_fn

    def _confidence(self, logprobs, tok: int, k: int = 4):
        """Probability of the chosen token and the top-k candidates, from the logprobs mlx-lm already computed."""
        import mlx.core as mx
        lp = logprobs.reshape(-1)
        ids = mx.argpartition(-lp, kth=k - 1)[:k]
        vals = mx.exp(lp[ids])
        p = mx.exp(lp[tok])
        mx.eval(ids, vals, p)
        pairs = sorted(zip(ids.tolist(), vals.tolist()), key=lambda x: -x[1])
        return round(float(p.item()), 4), [[self.decode_token(int(i)), round(float(v), 4)] for i, v in pairs]

    KEEP_GENS = 12

    def _prompt_ids(self, formatted: str) -> list[int]:
        """Tokenize exactly as mlx_lm.stream_generate does for a string prompt."""
        tok = self._tokenizer
        bos = getattr(tok, "bos_token", None)
        add = bos is None or not formatted.startswith(bos)
        return list(tok.encode(formatted, add_special_tokens=add))

    def lens_backfill(self, gid: int, chunk: int = 512, log=None) -> dict:
        """Logit lens for every token of an earlier answer, computed afterwards.

        Re-runs one forward pass over prompt + answer (chunked prefill with a KV cache), keeps every layer's
        hidden state at the positions that produced each answer token, and reads them through the model's
        own final norm + output head. Same numbers as the live lens, without slowing generation down.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        rec = self._gens.get(int(gid))
        if rec is None:
            raise RuntimeError("这段回答已经不在服务器的记录里（只保留最近 %d 次回答，或者模型重新加载过）" % self.KEEP_GENS)
        if not self._probe:
            raise RuntimeError("逐层预测需要层探针")
        lens = self._lens()
        prompt, gen = rec["prompt"], rec["gen"]
        if not gen:
            return {"gid": gid, "tokens": [], "lens": []}
        full = prompt + gen[:-1]                       # the pass that produced gen[j] ends at position len(prompt)-1+j
        lo = len(prompt) - 1
        restore = self._adapter_on
        if rec.get("adapter", True) != restore:
            self.set_adapter_enabled(rec.get("adapter", True))
        self._probe.enabled = True
        self._probe.capture_range(lo, len(full))
        try:
            cache = make_prompt_cache(self._model)
            arr = mx.array(full)
            for s in range(0, len(full), chunk):
                self._model(arr[s:s + chunk][None], cache=cache)
                keep = [x[-1] for x in self._probe.capture["h"] if x]
                mx.eval([c.state for c in cache] + keep)
            hs = self._probe.end_capture()
        finally:
            self._probe.capture = None
            self._probe.enabled = False
            if self._adapter_on != restore:
                self.set_adapter_enabled(restore)
        H = [mx.concatenate(h, axis=0) for h in hs]   # per layer [G, d]
        n, G = len(H), len(gen)
        vocab = int(getattr(self._tokenizer, "vocab_size", 0) or 0) or 160000
        step = max(1, int(1.2e9 // (n * max(vocab, 1) * 4)))   # keep the float32 logits of one batch around 1.2 GB
        out: list = []
        for s in range(0, G, step):
            e = min(G, s + step)
            rows = mx.concatenate([h[s:e] for h in H], axis=0)          # [n*(e-s), d], layer-major
            ids, probs = lens(rows)
            mx.eval(ids, probs)
            ids, probs = ids.tolist(), probs.tolist()
            k = e - s
            for j in range(k):
                out.append([[l, self.decode_token(int(ids[l * k + j])), round(float(probs[l * k + j]), 3)] for l in range(n)])
            if log:
                log(e, G)
        return {"gid": gid, "tokens": [self.decode_token(t) for t in gen], "lens": out}

    def decode_token(self, tid: int) -> str:
        try:
            return self._tokenizer.decode([tid])
        except Exception:
            return "?"

    def _format(self, prompt: str) -> str:
        return self._format_messages([{"role": "user", "content": prompt}])

    def _format_messages(self, messages: list[dict]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self._template_kwargs)
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 task: Any = None) -> Generation:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        formatted = self._format(prompt)
        mlxutil.reset_peak_memory()
        chunks: list[str] = []
        last = None
        start = time.monotonic()
        for resp in stream_generate(self._model, self._tokenizer, formatted,
                                    max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
            chunks.append(resp.text)
            last = resp
        seconds = time.monotonic() - start
        peak = getattr(last, "peak_memory", None)
        return Generation(
            text="".join(chunks),
            seconds=round(seconds, 3),
            prompt_tokens=getattr(last, "prompt_tokens", None),
            completion_tokens=getattr(last, "generation_tokens", None),
            tokens_per_second=round(getattr(last, "generation_tps", 0.0) or 0.0, 2) or None,
            peak_memory_gb=round(peak, 2) if peak else mlxutil.peak_memory_gb(),
        )

    def stream_chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.2,
                    lens: bool = False):
        """Yield {"text", "energy" (per-layer list or None), "tps", "lens", "token", "conf", "alts"} per generated chunk."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        formatted = self._format_messages(messages)
        prompt_ids = self._prompt_ids(formatted)
        self._gid += 1
        gid, gen_ids = self._gid, []
        self._gens[gid] = {"prompt": prompt_ids, "gen": gen_ids, "adapter": self._adapter_on}
        while len(self._gens) > self.KEEP_GENS:
            self._gens.popitem(last=False)
        if self._probe:
            self._probe.start()
            self._probe.enabled = True
            if lens:
                try:
                    self._probe.lens = self._lens()
                    self._probe.lens_layers = set(range(self._probe.n))   # all layers: one batched head pass
                except AttributeError:
                    self._probe.lens = None
        try:
            for resp in stream_generate(self._model, self._tokenizer, prompt_ids,
                                        max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
                out = {"text": resp.text, "energy": None, "gid": gid,
                       "tps": round(getattr(resp, "generation_tps", 0.0) or 0.0, 1) or None}
                if getattr(resp, "token", None) is not None:
                    gen_ids.append(int(resp.token))
                if self._probe:
                    energy, got = self._probe.take()
                    out["energy"] = energy
                    if got:
                        out["lens"] = [[l, self.decode_token(t), p] for l, t, p in got]
                    tok = getattr(resp, "token", None)
                    if tok is not None:
                        out["token"] = self.decode_token(int(tok))
                        lp = getattr(resp, "logprobs", None)
                        if lp is not None:
                            try:
                                out["conf"], out["alts"] = self._confidence(lp, int(tok))
                            except Exception:      # never let the visualisation break generation
                                pass
                yield out
        finally:
            if self._probe:
                self._probe.enabled = False
                self._probe.lens = None

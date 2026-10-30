"""Two-layer TOML config: config/cortex.toml (committed) + config/local.toml (machine-specific)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


@dataclass(frozen=True)
class RepoConfig:
    name: str
    path: Path
    test: str
    setup: str | None = None
    timeout_s: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    name: str
    backend: str
    model: str = ""
    base_url: str | None = None
    mode: str | None = None
    min_free_gb: float = 0.0
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSettings:
    timeout_s: int = 300
    cpu_seconds: int = 900
    stability_runs: int = 3
    context_budget_tokens: int = 24000
    max_source_lines: int = 200
    max_source_files: int = 5
    max_commits: int = 1000
    max_tasks_per_mine: int = 60
    source_extensions: tuple[str, ...] = (".py",)
    test_patterns: tuple[str, ...] = ("test_*.py", "*_test.py", "tests/*", "*/tests/*")


@dataclass(frozen=True)
class Config:
    data_dir: Path
    temperature: float
    max_tokens_cap: int
    eval: EvalSettings
    train_probe: dict[str, int]
    candidates: dict[str, Candidate]
    repos: dict[str, RepoConfig]
    raw: dict[str, Any]

    def candidate(self, name: str) -> Candidate:
        if name not in self.candidates:
            raise KeyError(f"unknown candidate {name!r}; known: {', '.join(sorted(self.candidates))}")
        return self.candidates[name]

    def repo(self, name: str) -> RepoConfig:
        if name not in self.repos:
            raise KeyError(
                f"unknown repo {name!r}; add [repos.{name}] to config/local.toml "
                f"(known: {', '.join(sorted(self.repos)) or 'none'})"
            )
        return self.repos[name]


def load_config(
    main: str | Path | None = None,
    local: str | Path | None = None,
    overrides: dict | None = None,
) -> Config:
    main_path = Path(main) if main else PROJECT_ROOT / "config" / "cortex.toml"
    local_path = Path(local) if local else main_path.with_name("local.toml")
    raw = tomllib.loads(main_path.read_text(encoding="utf-8"))
    if local_path.is_file():
        raw = deep_merge(raw, tomllib.loads(local_path.read_text(encoding="utf-8")))
    if overrides:
        raw = deep_merge(raw, overrides)

    data_dir = os.environ.get("CORTEX_DATA_DIR") or raw.get("paths", {}).get("data_dir")
    if not data_dir:
        raise ValueError("paths.data_dir is not set")

    ev = raw.get("eval", {})
    eval_settings = EvalSettings(
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in ev.items()
           if k in EvalSettings.__dataclass_fields__}
    )
    gen = raw.get("generation", {})

    candidates = {}
    for name, c in raw.get("candidates", {}).items():
        candidates[name] = Candidate(
            name=name,
            backend=c["backend"],
            model=c.get("model", ""),
            base_url=c.get("base_url"),
            mode=c.get("mode"),
            min_free_gb=float(c.get("min_free_gb", 0)),
            chat_template_kwargs=dict(c.get("chat_template_kwargs", {})),
        )

    repos = {}
    for name, r in raw.get("repos", {}).items():
        if "path" not in r or "test" not in r:
            raise ValueError(f"[repos.{name}] needs both 'path' and 'test'")
        repos[name] = RepoConfig(
            name=name,
            path=_expand(r["path"]),
            test=r["test"],
            setup=r.get("setup"),
            timeout_s=r.get("timeout_s"),
            env={str(k): str(v) for k, v in r.get("env", {}).items()},
        )

    return Config(
        data_dir=_expand(data_dir),
        temperature=float(gen.get("temperature", 0.0)),
        max_tokens_cap=int(gen.get("max_tokens_cap", 16384)),
        eval=eval_settings,
        train_probe={k: int(v) for k, v in raw.get("train_probe", {}).items()},
        candidates=candidates,
        repos=repos,
        raw=raw,
    )

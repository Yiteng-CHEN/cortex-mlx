"""Two-layer TOML config: the packaged defaults.toml + an optional local.toml (machine-specific).

local.toml is looked up in the project checkout (config/local.toml) and then in ~/.config/neurolume/.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULTS = PACKAGE_DIR / "defaults.toml"
WEB_DIR = PACKAGE_DIR / "web"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Running from a git checkout (models/ and config/local.toml live next to the code) or from an installed wheel.
IS_CHECKOUT = (PROJECT_ROOT / "pyproject.toml").is_file()
USER_CONFIG_DIR = Path.home() / ".config" / "neurolume"


def _env(name: str) -> str | None:
    """NEUROLUME_* environment variable."""
    return os.environ.get("NEUROLUME_" + name)


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
    tokenizer_config: dict[str, Any] = field(default_factory=dict)


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
    models_dir: Path
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
    main_path = Path(main) if main else DEFAULTS
    if local:
        local_path = Path(local)
    elif main:
        local_path = main_path.with_name("local.toml")
    else:
        local_path = next((p for p in (PROJECT_ROOT / "config" / "local.toml", USER_CONFIG_DIR / "local.toml")
                           if (IS_CHECKOUT or p.parent == USER_CONFIG_DIR) and p.is_file()),
                          USER_CONFIG_DIR / "local.toml")
    raw = tomllib.loads(main_path.read_text(encoding="utf-8"))
    if local_path.is_file():
        raw = deep_merge(raw, tomllib.loads(local_path.read_text(encoding="utf-8")))
    if overrides:
        raw = deep_merge(raw, overrides)

    data_dir = _env("DATA_DIR") or raw.get("paths", {}).get("data_dir")
    if not data_dir:
        raise ValueError("paths.data_dir is not set")
    # Relative models_dir: next to the code in a checkout (keeps existing downloads), else inside the data dir.
    base = PROJECT_ROOT if IS_CHECKOUT else _expand(data_dir)
    models_dir = _env("MODELS_DIR") or raw.get("paths", {}).get("models_dir", "models")
    models_dir = _expand(models_dir) if os.path.isabs(os.path.expanduser(str(models_dir))) \
        else (base / str(models_dir)).resolve()

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
            tokenizer_config=dict(c.get("tokenizer_config", {})),
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
        models_dir=models_dir,
        temperature=float(gen.get("temperature", 0.0)),
        max_tokens_cap=int(gen.get("max_tokens_cap", 16384)),
        eval=eval_settings,
        train_probe={k: int(v) for k, v in raw.get("train_probe", {}).items()},
        candidates=candidates,
        repos=repos,
        raw=raw,
    )


def hf_cache_dir(cfg: Config) -> Path:
    """Where Hugging Face model downloads live (D17)."""
    return cfg.models_dir / "huggingface" / "hub"


def apply_model_env(cfg: Config) -> str:
    """Point the Hugging Face cache at <project>/models so every download lands there.

    Only the model cache moves (HF_HUB_CACHE); HF_HOME, and with it any saved login token, is
    untouched. An HF_HUB_CACHE the user exported themselves is respected. Child processes
    (e.g. the mlx_lm LoRA runner) inherit it. Returns the effective cache directory.
    """
    if not os.environ.get("HF_HUB_CACHE"):
        target = hf_cache_dir(cfg)
        target.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_CACHE"] = str(target)
    return os.environ["HF_HUB_CACHE"]

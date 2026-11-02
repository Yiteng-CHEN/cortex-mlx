"""`neurolume` command line."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .config import apply_model_env, load_config
from .paths import DataPaths


def _paths(cfg) -> DataPaths:
    return DataPaths(cfg.data_dir).ensure()


# ── eval ──────────────────────────────────────────────────────────────────────

def cmd_mine(args, cfg) -> int:
    from .evals.miner import mine

    repo = cfg.repo(args.repo)
    print(f"mining {repo.name} ({repo.path}) ...")
    rep = mine(_paths(cfg), repo, cfg.eval, limit=args.limit)
    print(f"\nscanned {rep.scanned} commits, accepted {len(rep.accepted)}")
    if rep.skipped:
        print("skipped:  " + ", ".join(f"{k}={v}" for k, v in sorted(rep.skipped.items())))
    if rep.rejected:
        reasons: dict[str, int] = {}
        for r in rep.rejected.values():
            reasons[r] = reasons.get(r, 0) + 1
        print("rejected: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print("next: neurolume eval review")
    return 0


def cmd_review(args, cfg) -> int:
    from .evals.review import review

    decisions = None
    if args.decide:
        decisions = {}
        for item in args.decide:
            task_id, _, key = item.partition("=")
            decisions[task_id] = key.strip().lower()[:1]
    counts = review(cfg, _paths(cfg), decisions=decisions)
    print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    return 0


def cmd_list(args, cfg) -> int:
    from .evals.miner import split_counts
    from .evals.task import load_split

    paths = _paths(cfg)
    for split, n in split_counts(paths).items():
        print(f"{split:<11}{n}")
        if args.verbose:
            for t in load_split(paths, split):
                print(f"    {t.id}  {t.statement.splitlines()[0][:70]}")
    return 0


def cmd_run(args, cfg) -> int:
    from .backends.factory import make_backend
    from .evals.runner import run_eval, save_result
    from .evals.task import load_split

    paths = _paths(cfg)
    tasks = load_split(paths, args.split)
    if args.task:
        tasks = [t for t in tasks if t.id in set(args.task)]
    if not tasks:
        print(f"no tasks in split {args.split!r}", file=sys.stderr)
        return 2
    for name in args.candidate:
        backend = make_backend(cfg, name)          # loads the model (slow, memory heavy)
        result = run_eval(cfg, paths, backend, tasks, split=args.split, samples=args.samples)
        s = result["summary"]
        print(f"{name}: pass {s['pass_rate']:.0%}, format errors {s['format_error_rate']:.0%}, "
              f"{s['tokens_per_second'] or '-'} tok/s, peak {s['peak_memory_gb'] or '-'} GB")
        print("saved:", save_result(paths, result))
        del backend
    return 0


def cmd_compare(args, cfg) -> int:
    from .evals.report import as_text, comparison_rows, latest_results, probe_results, write_comparison

    paths = _paths(cfg)
    results = latest_results(paths, split=args.split)
    if args.candidate:
        results = [r for r in results if r["candidate"] in set(args.candidate)]
    if not results:
        print("no eval results yet; run `neurolume eval run` first", file=sys.stderr)
        return 2
    rows = comparison_rows(results, probe_results(paths))
    print(as_text(rows))
    print("\nsaved:", write_comparison(paths, rows))
    return 0


def cmd_cleanup(args, cfg) -> int:
    from .evals.sandbox import cleanup_stale

    paths = _paths(cfg)
    stale = cleanup_stale(paths, list(cfg.repos.values()), dry_run=not args.apply)
    verb = "removed" if args.apply else "would remove (dry run; add --apply)"
    print(f"{verb}: {len(stale)} stale worktree(s)")
    for p in stale:
        print("  ", p)
    return 0


# ── models / sys / dev ────────────────────────────────────────────────────────

def cmd_inspect(args, cfg) -> int:
    from .models.inspect import inspect_model

    report, out = inspect_model(cfg, _paths(cfg), args.candidate)
    s = report["summary"]
    print(f"{args.candidate}: {s['num_layers']} layers, pattern {s['pattern']} "
          f"(L=linear attention, A=attention)")
    print(f"  mixers {s['mixers']}, ffn {s['ffn']}, weights {report['weights_gb']} GB, "
          f"load {report['load_seconds']} s")
    print("saved:", out)
    return 0


def cmd_probe(args, cfg) -> int:
    from .models.train_probe import run_probe

    result, out = run_probe(cfg, _paths(cfg), args.candidate)
    if result["ok"]:
        print(f"{args.candidate}: {result['seconds_per_step'] or '-'} s/step, "
              f"{result['tokens_per_sec'] or '-'} tok/s, peak {result['peak_memory_gb'] or '-'} GB "
              f"({result['data_source']} data)")
    else:
        print(f"{args.candidate}: LoRA probe FAILED (exit {result['returncode']}); "
              f"see log_tail in {out}", file=sys.stderr)
    print("saved:", out)
    return 0 if result["ok"] else 1


def cmd_mem(args, cfg) -> int:
    from .sysinfo import available_memory_gb

    need = args.min_free_gb
    if need is None and args.candidate:
        need = cfg.candidate(args.candidate).min_free_gb
    avail = available_memory_gb()
    print(f"available memory: {avail if avail is not None else 'unknown'} GB"
          + (f" (need {need} GB)" if need else ""))
    if need and avail is not None and avail < need:
        print("not enough free memory; close other apps and retry", file=sys.stderr)
        return 1
    return 0


def cmd_config(args, cfg) -> int:
    print(f"data_dir: {cfg.data_dir}")
    print(f"models_dir: {cfg.models_dir}")
    print(f"hf_cache: {os.environ.get('HF_HUB_CACHE', '(default)')}")
    print("candidates: " + ", ".join(f"{n} ({c.backend})" for n, c in cfg.candidates.items()))
    print("repos: " + (", ".join(f"{n} -> {r.path}" for n, r in cfg.repos.items()) or "none "
                       "(add [repos.<name>] to config/local.toml)"))
    if args.json:
        print(json.dumps(cfg.raw, indent=2, default=str))
    return 0


def cmd_serve(args, cfg) -> int:
    from .server.app import serve

    try:
        serve(cfg, host=args.host, port=args.port, model=args.model, open_browser=args.open)
    except OSError as e:
        print(f"error: cannot listen on {args.host}:{args.port} ({e.strerror}). "
              f"Neurolume may already be running: open http://{args.host}:{args.port}", file=sys.stderr)
        return 1
    return 0


def cmd_mcp_search(args, cfg) -> int:
    from .mcp_search import main

    return main(cfg)


def cmd_fixture(args, cfg) -> int:
    from .devtools.fixture_repo import TEST_COMMAND, build

    repo = build(args.path)
    print(f"created {repo}")
    print("add to config/local.toml:\n")
    print(f'[repos.fixture]\npath = "{repo.resolve()}"\ntest = "{TEST_COMMAND}"')
    return 0


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neurolume", description="Neurolume local self-evolving coding assistant")
    ap.add_argument("--config", help="path to neurolume.toml (default: config/neurolume.toml)")
    ap.add_argument("--local-config", help="path to local.toml (default: next to neurolume.toml)")
    top = ap.add_subparsers(dest="group", required=True)

    ev = top.add_parser("eval", help="eval tasks: mine, review, run, compare").add_subparsers(dest="cmd", required=True)
    p = ev.add_parser("mine", help="mine tasks from a repo's git history")
    p.add_argument("repo", help="repo name from config/local.toml")
    p.add_argument("--limit", type=int, help="stop after N accepted tasks")
    p.set_defaults(fn=cmd_mine)
    p = ev.add_parser("review", help="sort mined candidates into dev / holdout / rejected")
    p.add_argument("--decide", action="append", metavar="TASK=d|h|r",
                   help="non-interactive decision (repeatable)")
    p.set_defaults(fn=cmd_review)
    p = ev.add_parser("list", help="task counts per split")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_list)
    p = ev.add_parser("run", help="run candidates over a split")
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--split", default="dev", choices=("dev", "holdout", "candidates"))
    p.add_argument("--task", action="append", help="only these task ids")
    p.add_argument("--samples", type=int, default=1)
    p.set_defaults(fn=cmd_run)
    p = ev.add_parser("compare", help="compare the latest results per candidate")
    p.add_argument("--split")
    p.add_argument("--candidate", action="append")
    p.set_defaults(fn=cmd_compare)
    p = ev.add_parser("cleanup", help="remove stale worktrees (dry run unless --apply)")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(fn=cmd_cleanup)

    mo = top.add_parser("models", help="model tools (Mac only)").add_subparsers(dest="cmd", required=True)
    p = mo.add_parser("inspect", help="layer types of a candidate")
    p.add_argument("--candidate", required=True)
    p.set_defaults(fn=cmd_inspect)
    p = mo.add_parser("train-probe", help="a few LoRA steps: s/step and peak memory")
    p.add_argument("--candidate", required=True)
    p.set_defaults(fn=cmd_probe)

    sy = top.add_parser("sys", help="host checks").add_subparsers(dest="cmd", required=True)
    p = sy.add_parser("mem", help="check free memory")
    p.add_argument("--min-free-gb", type=float)
    p.add_argument("--candidate")
    p.set_defaults(fn=cmd_mem)
    p = sy.add_parser("config", help="show effective config")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_config)

    p = top.add_parser("serve", help="local server: OpenAI-compatible chat API + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", help="candidate to load at startup (default: choose in the UI)")
    p.add_argument("--open", action="store_true", help="open the web UI in the default browser")
    p.set_defaults(fn=cmd_serve)

    p = top.add_parser("mcp-search", help="MCP server (stdio): web_search + web_fetch for Claude Code / Codex with any local model")
    p.set_defaults(fn=cmd_mcp_search)

    dv = top.add_parser("dev", help="developer tools").add_subparsers(dest="cmd", required=True)
    p = dv.add_parser("fixture-repo", help="create the synthetic test repo")
    p.add_argument("path")
    p.set_defaults(fn=cmd_fixture)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.local_config)
        apply_model_env(cfg)
        return args.fn(args, cfg)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

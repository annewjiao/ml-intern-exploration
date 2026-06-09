"""
ml-intern-logs — CLI viewer for process log files.

Usage:
  ml-intern-logs              # show story from the latest session
  ml-intern-logs --all        # show all sessions, newest first
  ml-intern-logs --cost       # show cost summary per session
  ml-intern-logs --raw        # include every event (no filtering)
  ml-intern-logs --file PATH  # view a specific .jsonl file
"""

import argparse
import json
import sys
from pathlib import Path

from agent.utils.process_logger import DEFAULT_LOG_DIR

# Events that are internal bookkeeping — hidden unless --raw
_NOISE_EVENTS = {"llm_call", "ready", "processing", "tool_state_change", "shutdown"}

# ANSI colours (degrade gracefully if piped)
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def _dim(t):    return _c("2", t)
def _green(t):  return _c("32", t)
def _red(t):    return _c("31", t)
def _yellow(t): return _c("33", t)
def _bold(t):   return _c("1", t)
def _cyan(t):   return _c("36", t)


def _format_line(r: dict) -> str:
    ts = r.get("ts", "")
    time = ts[11:19] if len(ts) >= 19 else ts
    event = r.get("event", "")
    data = r.get("data") or {}
    summary = r.get("summary", event)

    if event == "user_input":
        # Read from data so old logs with truncated summaries show full text
        text = data.get("text") or summary.removeprefix("User: ")
        return f"\n{_bold('User:')} {text}\n"
    if event == "tool_call":
        tool = data.get("tool", "?")
        args = json.dumps(data.get("arguments", {}))
        return f"{_dim(time)}  {_cyan('→')} Tool call → {tool}({args})"
    if event == "tool_output":
        tool = data.get("tool", "?")
        ok = data.get("success", False)
        marker = _green("✓") if ok else _red("✗")
        out = data.get("output") or ""
        return f"{_dim(time)}  {marker} {tool}: {out}"
    if event == "assistant_reasoning":
        # Read from data.content so old logs with truncated summaries show full text
        text = data.get("content") or summary.removeprefix("Agent reasoning: ")
        return f"{_dim(time)}  {_yellow('💭')} {text}"
    if event == "turn_complete":
        return f"{_dim(time)}  {_green('✔')} {_bold(summary)}"
    if event == "error":
        return f"{_dim(time)}  {_red('✖')} {summary}"
    if event == "approval_required":
        return f"{_dim(time)}  {_yellow('⚠')} {summary}"
    if event == "compacted":
        return f"{_dim(time)}  {_dim(summary)}"
    return f"{_dim(time)}  {summary}"


def _cost_summary(path: Path) -> dict:
    total_cost = 0.0
    total_tokens = 0
    calls = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") == "llm_call":
            d = r.get("data", {})
            total_cost += d.get("cost_usd") or 0.0
            total_tokens += d.get("total_tokens") or 0
            calls += 1
    return {"cost_usd": total_cost, "tokens": total_tokens, "calls": calls}


def _show_file(path: Path, *, raw: bool = False) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    printed = 0
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if not raw and r.get("event") in _NOISE_EVENTS:
            continue
        print(_format_line(r))
        printed += 1
    if printed == 0:
        print(_dim("  (no events yet)"))


def _list_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("session_*.jsonl"), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml-intern-logs",
        description="View ml-intern process logs",
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all sessions, newest first"
    )
    parser.add_argument(
        "--cost", action="store_true", help="Show cost summary per session"
    )
    parser.add_argument(
        "--raw", action="store_true", help="Show every event including internal ones"
    )
    parser.add_argument(
        "--file", metavar="PATH", help="View a specific .jsonl file"
    )
    parser.add_argument(
        "--dir", metavar="DIR", default=None,
        help=f"Log directory (default: {DEFAULT_LOG_DIR})"
    )
    args = parser.parse_args()

    log_dir = Path(args.dir) if args.dir else DEFAULT_LOG_DIR

    # -- Single file mode --
    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        print(_bold(f"=== {path.name} ==="))
        _show_file(path, raw=args.raw)
        if args.cost:
            s = _cost_summary(path)
            print(_dim(f"\n  Cost: ${s['cost_usd']:.4f}  |  Tokens: {s['tokens']:,}  |  LLM calls: {s['calls']}"))
        return

    logs = _list_logs(log_dir)
    if not logs:
        print(f"No session logs found in {log_dir}/")
        print("Run ml-intern first to generate logs.")
        return

    # -- Cost table mode --
    if args.cost:
        print(_bold(f"{'Session':<40}  {'Cost':>8}  {'Tokens':>10}  {'Calls':>6}"))
        print("─" * 70)
        grand_cost = 0.0
        grand_tokens = 0
        for path in logs:
            s = _cost_summary(path)
            grand_cost += s["cost_usd"]
            grand_tokens += s["tokens"]
            print(f"{path.name:<40}  ${s['cost_usd']:>7.4f}  {s['tokens']:>10,}  {s['calls']:>6}")
        print("─" * 70)
        print(f"{'TOTAL':<40}  ${grand_cost:>7.4f}  {grand_tokens:>10,}")
        return

    # -- All sessions mode --
    if args.all:
        for path in logs:
            print(_bold(f"\n=== {path.name} ==="))
            _show_file(path, raw=args.raw)
        return

    # -- Default: latest session --
    latest = logs[0]
    print(_bold(f"=== {latest.name} ===") + _dim(f"  ({len(logs)} session(s) total, use --all to see more)\n"))
    _show_file(latest, raw=args.raw)

"""
Process logger — tees the agent event_queue to a JSONL file.

Each line is a JSON object:
  { "ts": <iso8601>, "event": <event_type>, "summary": <human string>, "data": {...} }

Wire in by replacing the raw asyncio.Queue with a TeeQueue before passing it
to submission_loop and event_listener.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = Path("process_logs")


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _summarise(event_type: str, data: dict | None) -> str:
    """Return a one-line human-readable description of an event."""
    d = data or {}
    if event_type == "user_input":
        text = (d.get("text") or "").strip()
        return f"User: {text}"
    if event_type == "processing":
        return "Agent started processing user input"
    if event_type == "ready":
        return f"Agent ready ({d.get('tool_count', '?')} tools loaded)"
    if event_type == "assistant_reasoning":
        text = (d.get("content") or "")
        return f"Agent reasoning: {text}"
    if event_type == "tool_call":
        tool = d.get("tool", "?")
        args = json.dumps(d.get("arguments", {}))
        return f"Tool call → {tool}({args})"
    if event_type == "tool_output":
        tool = d.get("tool", "?")
        ok = "✓" if d.get("success") else "✗"
        out = (d.get("output") or "")
        return f"Tool result {ok} {tool}: {out}"
    if event_type == "tool_log":
        tool = d.get("tool", "")
        log = d.get("log", "")
        return f"[{tool}] {log}" if tool else log
    if event_type == "plan_update":
        todos = d.get("plan", d.get("todos", []))
        done = sum(1 for t in todos if t.get("status") == "completed")
        total = len(todos)
        current = next((t.get("content", "") for t in todos if t.get("status") == "in_progress"), None)
        if current:
            return f"Plan ({done}/{total} done) — now working on: {current}"
        return f"Plan updated ({done}/{total} steps done)"
    if event_type == "tool_state_change":
        return f"Tool {d.get('tool','?')} → {d.get('state','?')}"
    if event_type == "approval_required":
        tools = [t.get("tool", "?") for t in d.get("tools", [])]
        return f"Approval required for: {', '.join(tools)}"
    if event_type == "turn_complete":
        return f"Turn complete (history={d.get('history_size','?')} messages)"
    if event_type == "interrupted":
        return "Agent was interrupted by user"
    if event_type == "error":
        err = (d.get("error") or "unknown error")[:200]
        return f"ERROR: {err}"
    if event_type == "compacted":
        return f"Context compacted: {d.get('old_tokens','?')} → {d.get('new_tokens','?')} tokens"
    if event_type == "shutdown":
        return "Agent shutting down"
    if event_type == "new_complete":
        return "New conversation started"
    if event_type == "resume_complete":
        return f"Session resumed from {d.get('path','?')}"
    if event_type == "undo_complete":
        return "Last turn undone"
    if event_type == "session_terminated":
        return f"Session terminated: {d.get('reason','?')}"
    return event_type


class ProcessLogger:
    """Writes agent events to a JSONL file as they flow through the queue.

    Streaming chunks (assistant_chunk / assistant_stream_end) are buffered and
    flushed as a single ``assistant_reasoning`` entry when the stream ends.
    This turns dozens of noisy token fragments into one readable thought.
    """

    def __init__(self, log_path: Path):
        self._log_path = log_path
        self._file = None
        self._chunk_buf: list[str] = []
        self._chunk_start_ts: str | None = None

    def open(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a", encoding="utf-8")
        logger.info("Process log: %s", self._log_path)

    def close(self) -> None:
        self._flush_chunks()
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def _flush_chunks(self) -> None:
        if not self._chunk_buf:
            return
        full_text = "".join(self._chunk_buf)
        self._chunk_buf = []
        ts = self._chunk_start_ts or _iso_now()
        self._chunk_start_ts = None
        if full_text.strip():
            self._write_record(ts, "assistant_reasoning", {"content": full_text})

    def _write_record(self, ts: str, event_type: str, data: dict | None) -> None:
        if not self._file:
            return
        record: dict[str, Any] = {
            "ts": ts,
            "event": event_type,
            "summary": _summarise(event_type, data),
            "data": data or {},
        }
        try:
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
        except Exception as e:
            logger.warning("process_logger write failed: %s", e)

    def write_event(self, event_type: str, data: dict | None) -> None:
        if not self._file:
            return
        d = data or {}

        if event_type == "assistant_chunk":
            # Buffer — don't write yet
            if self._chunk_start_ts is None:
                self._chunk_start_ts = _iso_now()
            self._chunk_buf.append(d.get("content") or "")
            return

        if event_type == "assistant_stream_end":
            # Flush buffered chunks as one reasoning entry; skip the stream_end line
            self._flush_chunks()
            return

        if event_type == "assistant_message":
            # Non-streaming path — treat like a completed reasoning block
            self._flush_chunks()
            self._write_record(_iso_now(), "assistant_reasoning", {"content": d.get("content") or ""})
            return

        # Any other event: flush pending chunks first so ordering is preserved
        self._flush_chunks()
        self._write_record(_iso_now(), event_type, data)


class TeeQueue:
    """Wraps asyncio.Queue; every item put() is also forwarded to a ProcessLogger."""

    def __init__(self, process_logger: ProcessLogger):
        self._q: asyncio.Queue = asyncio.Queue()
        self._logger = process_logger

    # --- Queue interface used by the agent internals ---

    async def put(self, item: Any) -> None:
        event_type = getattr(item, "event_type", None)
        data = getattr(item, "data", None)
        if event_type is not None:
            self._logger.write_event(event_type, data)
        await self._q.put(item)

    async def get(self) -> Any:
        return await self._q.get()

    def get_nowait(self) -> Any:
        return self._q.get_nowait()

    def empty(self) -> bool:
        return self._q.empty()

    def task_done(self) -> None:
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()


def make_log_path(log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    """Return a timestamped log file path for the current session."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    return log_dir / f"session_{ts}_{pid}.jsonl"

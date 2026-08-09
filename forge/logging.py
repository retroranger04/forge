"""Tagged log format for parallel workers writing to one shared file.

Line format:
    <ISO8601 Z> [<worker_id>] <split_or_phase>/<event> <k=v pairs>

Example:
    2026-08-09T14:32:17Z [train_shard_2] train/progress i=1400 N=2000 elapsed_sec=612

Every line carries worker_id, split_or_phase and event so that a reader can
attribute progress to a specific worker. Session Four shipped untagged lines
from sharded workers and its watchdog could not tell which shard was alive;
that is the gap this format closes.
"""
from __future__ import annotations

import os
import re
import shlex
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

EVENTS = ("progress", "heartbeat", "complete", "error", "timeout", "warning")

_LINE = re.compile(r"^(\S+) \[([^\]\s]+)\] ([^/\s]+)/([^/\s]+)(?: (.*))?$")
_FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def _lock(path: Path):
    """Exclusive cross-process lock held on a sidecar file.

    A sidecar is used rather than the log itself so the lock byte range never
    moves as the log grows.
    """
    fd = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR)
    try:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def emit(
    log_path: str | Path,
    worker_id: str,
    split_or_phase: str,
    event: Literal["progress", "heartbeat", "complete", "error", "timeout", "warning"],
    **kv: str | int | float,
) -> None:
    """Append a tagged log line. Thread- and process-safe (uses file lock)."""
    if event not in EVENTS:
        raise ValueError(f"event must be one of {EVENTS}, got {event!r}")
    for name, val in (("worker_id", worker_id), ("split_or_phase", split_or_phase)):
        if not val or any(c.isspace() for c in val) or "/" in val or "]" in val:
            raise ValueError(f"{name} must be non-empty with no whitespace, '/' or ']': {val!r}")

    # shlex.quote keeps values containing spaces on one parseable token, but it
    # preserves newlines: an exception message carrying one would split into a
    # second physical line that parse_log drops, silently losing the event and
    # making a live worker look stalled. Reject rather than mangle.
    parts = []
    for k, v in kv.items():
        text = str(v)
        if not _FIELD_NAME.fullmatch(k):
            raise ValueError(f"invalid field name: {k!r}")
        if "\r" in text or "\n" in text:
            raise ValueError(f"field value must not contain a line break: {k!r}")
        parts.append(f"{k}={shlex.quote(text)}")
    pairs = " ".join(parts)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts} [{worker_id}] {split_or_phase}/{event}" + (f" {pairs}\n" if pairs else "\n")

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _lock(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)


def parse_log(log_path: str | Path) -> list[dict]:
    """Parse a tagged log file into a list of dicts, one per line.

    Each dict has ts (aware datetime), worker_id, split_or_phase, event and
    fields (the k=v pairs). Malformed lines are skipped, not raised on.
    """
    path = Path(log_path)
    if not path.exists():
        return []

    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LINE.match(line.strip())
        if not m or m.group(4) not in EVENTS:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        fields = {}
        try:
            for tok in shlex.split(m.group(5) or ""):
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    fields[k] = v
        except ValueError:
            continue  # unbalanced quoting
        out.append({
            "ts": ts,
            "worker_id": m.group(2),
            "split_or_phase": m.group(3),
            "event": m.group(4),
            "fields": fields,
        })
    return out

"""Worker attribution and stall detection over a tagged shared log.

Reads the format defined in forge.logging and decides, per worker, whether it
finished, went quiet, or ran past its budget. It does not kill anything: the
caller's on_stall callback owns termination, which keeps this module pure and
testable.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from forge.logging import emit, parse_log

TERMINAL = ("complete", "stalled", "budget_exceeded", "missing")


class Watchdog:
    def __init__(
        self,
        log_path: str | Path,
        expected_workers: set[str],
        stale_sec: int = 180,
        budget_sec: int = 5400,  # 90 min
        poll_interval_sec: int = 60,
        on_stall: Callable[[str, str], None] | None = None,  # (worker_id, reason)
    ):
        self.log_path = Path(log_path)
        self.expected_workers = set(expected_workers)
        self.stale_sec = stale_sec
        self.budget_sec = budget_sec
        self.poll_interval_sec = poll_interval_sec
        self.on_stall = on_stall
        self._fired: set[str] = set()
        self._started = datetime.now(timezone.utc)

    def poll(self) -> dict:
        """One pass over the log. Returns {worker_id: status}."""
        seen: dict[str, dict] = {}
        for e in parse_log(self.log_path):
            w = e["worker_id"]
            if w not in self.expected_workers:
                continue
            rec = seen.setdefault(w, {"first": e["ts"], "last": e["ts"], "complete": False})
            rec["first"] = min(rec["first"], e["ts"])
            rec["last"] = max(rec["last"], e["ts"])
            rec["complete"] |= e["event"] == "complete"

        now = datetime.now(timezone.utc)
        statuses = {}
        for w in sorted(self.expected_workers):
            rec = seen.get(w)
            if rec is None:
                # never appeared; time it out against the watchdog's own start
                # so a worker that never launches cannot block forever
                statuses[w] = ("missing" if (now - self._started).total_seconds() > self.stale_sec
                               else "running")
            elif rec["complete"]:
                statuses[w] = "complete"
            elif (now - rec["last"]).total_seconds() > self.stale_sec:
                # checked before budget: "went quiet" is the more actionable
                # diagnosis when a worker is both quiet and over budget
                statuses[w] = "stalled"
            elif (now - rec["first"]).total_seconds() > self.budget_sec:
                statuses[w] = "budget_exceeded"
            else:
                statuses[w] = "running"

        for w, st in statuses.items():
            if st in ("stalled", "budget_exceeded", "missing") and w not in self._fired:
                self._fired.add(w)
                reason = {
                    "stalled": f"no log line for over {self.stale_sec}s",
                    "budget_exceeded": f"exceeded {self.budget_sec}s budget",
                    "missing": f"no log line at all after {self.stale_sec}s",
                }[st]
                emit(self.log_path, "watchdog", "watchdog", "warning", worker=w, reason=reason)
                if self.on_stall is not None:
                    self.on_stall(w, reason)
        return statuses

    def run_until_all_complete_or_stalled(self) -> dict:
        """
        Blocks until all expected workers have emitted a `complete` event OR
        all remaining workers are stalled/budget-exceeded.
        Returns a dict of {worker_id: final_status} where status is
        'complete', 'stalled', 'budget_exceeded', or 'missing'.
        """
        while True:
            statuses = self.poll()
            if all(s in TERMINAL for s in statuses.values()):
                return statuses
            time.sleep(self.poll_interval_sec)

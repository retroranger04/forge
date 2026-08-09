import threading
from datetime import datetime, timedelta, timezone

import pytest

from forge.logging import emit
from forge.watchdog import Watchdog


def write_aged(log, worker, event, age_sec, phase="train", **kv):
    """Append a well-formed line dated age_sec in the past.

    Backdating keeps the stall tests instant instead of making them sleep.
    """
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_sec)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pairs = " ".join(f"{k}={v}" for k, v in kv.items())
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"{ts} [{worker}] {phase}/{event}" + (f" {pairs}\n" if pairs else "\n"))


def test_all_complete_workers_report_complete(tmp_path):
    log = tmp_path / "run.log"
    for w in ("w1", "w2", "w3"):
        emit(log, w, "train", "progress", i=100)
        emit(log, w, "train", "complete", saved=2000)

    wd = Watchdog(log, {"w1", "w2", "w3"}, stale_sec=5, poll_interval_sec=60)
    assert wd.run_until_all_complete_or_stalled() == {
        "w1": "complete", "w2": "complete", "w3": "complete"
    }


def test_worker_that_stops_emitting_is_stalled(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "w1", "train", "complete", saved=10)
    write_aged(log, "w2", "progress", age_sec=30, i=400)  # went quiet 30s ago

    wd = Watchdog(log, {"w1", "w2"}, stale_sec=5, poll_interval_sec=60)
    assert wd.poll() == {"w1": "complete", "w2": "stalled"}


def test_on_stall_receives_offending_worker_id(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "w1", "train", "complete", saved=10)
    write_aged(log, "w2", "progress", age_sec=30, i=400)

    calls = []
    wd = Watchdog(log, {"w1", "w2"}, stale_sec=5, poll_interval_sec=60,
                  on_stall=lambda w, r: calls.append((w, r)))
    wd.poll()

    assert len(calls) == 1
    assert calls[0][0] == "w2"
    assert "5s" in calls[0][1]

    wd.poll()  # already fired; must not fire again
    assert len(calls) == 1


def test_watchdog_exits_when_all_workers_terminal(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "w1", "train", "complete", saved=10)
    write_aged(log, "w2", "progress", age_sec=30, i=400)

    result = {}
    # poll_interval is long, so this only returns promptly if the mixed
    # complete/stalled state is recognised as terminal
    t = threading.Thread(
        target=lambda: result.update(
            Watchdog(log, {"w1", "w2"}, stale_sec=5, poll_interval_sec=60)
            .run_until_all_complete_or_stalled()
        )
    )
    t.start()
    t.join(timeout=15)

    assert not t.is_alive(), "watchdog did not exit on an all-terminal state"
    assert result == {"w1": "complete", "w2": "stalled"}


def test_budget_exceeded_for_still_active_worker(tmp_path):
    log = tmp_path / "run.log"
    write_aged(log, "w1", "progress", age_sec=200, i=1)   # started long ago
    write_aged(log, "w1", "heartbeat", age_sec=1, i=900)  # but still emitting

    wd = Watchdog(log, {"w1"}, stale_sec=60, budget_sec=100, poll_interval_sec=60)
    assert wd.poll() == {"w1": "budget_exceeded"}


def test_worker_that_never_appears_is_missing(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "w1", "train", "complete", saved=10)

    wd = Watchdog(log, {"w1", "ghost"}, stale_sec=0, poll_interval_sec=60)
    assert wd.run_until_all_complete_or_stalled() == {"w1": "complete", "ghost": "missing"}


def test_on_stall_fires_even_if_warning_logging_fails(tmp_path):
    log = tmp_path / "run.log"
    write_aged(log, "w1", "progress", age_sec=30, i=400)

    def boom(*a, **k):
        raise OSError("no space left on device")

    import forge.watchdog as wd_mod
    original, wd_mod.emit = wd_mod.emit, boom
    calls = []
    try:
        wd = Watchdog(log, {"w1"}, stale_sec=5, poll_interval_sec=60,
                      on_stall=lambda w, r: calls.append(w))
        # the callback is what terminates the worker; a logging failure must
        # not swallow it, since _fired blocks any later retry
        with pytest.raises(OSError):
            wd.poll()
    finally:
        wd_mod.emit = original

    assert calls == ["w1"]


def test_watchdog_logs_a_warning_event(tmp_path):
    log = tmp_path / "run.log"
    write_aged(log, "w1", "progress", age_sec=30, i=400)

    wd = Watchdog(log, {"w1"}, stale_sec=5, poll_interval_sec=60)
    wd.poll()

    from forge.logging import parse_log
    warnings = [e for e in parse_log(log) if e["worker_id"] == "watchdog"]
    assert len(warnings) == 1
    assert warnings[0]["event"] == "warning"
    assert warnings[0]["fields"]["worker"] == "w1"

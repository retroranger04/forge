import threading

from forge.logging import emit, parse_log


def test_emitted_line_is_parseable(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "train_shard_2", "train", "progress", i=1400, N=2000, elapsed_sec=612)

    entries = parse_log(log)
    assert len(entries) == 1
    e = entries[0]
    assert e["worker_id"] == "train_shard_2"
    assert e["split_or_phase"] == "train"
    assert e["event"] == "progress"
    assert e["fields"] == {"i": "1400", "N": "2000", "elapsed_sec": "612"}
    assert e["ts"].tzinfo is not None

    raw = log.read_text().strip()
    assert raw.endswith("[train_shard_2] train/progress i=1400 N=2000 elapsed_sec=612")
    assert raw.startswith(e["ts"].strftime("%Y-%m-%dT%H:%M:%SZ"))


def test_concurrent_writes_are_not_lost(tmp_path):
    log = tmp_path / "run.log"
    barrier = threading.Barrier(5)

    def worker(n):
        barrier.wait()  # maximise overlap on the append
        emit(log, f"w{n}", "sample_gen", "heartbeat", i=n)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = parse_log(log)
    assert len(entries) == 5
    assert {e["worker_id"] for e in entries} == {f"w{n}" for n in range(5)}
    # no interleaved/torn lines
    assert len(log.read_text().strip().splitlines()) == 5


def test_malformed_lines_are_skipped(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "w1", "train", "progress", i=1)
    with log.open("a", encoding="utf-8") as f:
        f.write("this is not a tagged line\n")
        f.write("2026-08-09T14:32:17Z missing-brackets train/progress i=2\n")
        f.write("not-a-timestamp [w2] train/progress i=3\n")
        f.write("2026-08-09T14:32:17Z [w3] train/not_an_event i=4\n")
        f.write("\n")
    emit(log, "w4", "train", "complete", i=5)

    entries = parse_log(log)
    assert [e["worker_id"] for e in entries] == ["w1", "w4"]


def test_values_with_spaces_survive_round_trip(tmp_path):
    log = tmp_path / "run.log"
    emit(log, "w1", "train", "error", msg="solver did not converge", code=3)

    entries = parse_log(log)
    assert entries[0]["fields"] == {"msg": "solver did not converge", "code": "3"}

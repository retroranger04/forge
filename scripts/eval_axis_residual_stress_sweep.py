"""Residual-stress-sweep coverage evaluation: the frozen run_01 model against seven pre-stress levels.

Inference only. The checkpoint, the locked calibration thresholds and every
existing split are read-only here; this writes coverage numbers and a log and
nothing else.

q_hat is read out of the baseline results file rather than retyped, so the
thresholds cannot drift by transcription, and asserted against the locked spec.
Results are rewritten after every level, so a failure late in the sweep still
leaves the levels that already finished on disk.
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conformal_eval import load_ema_model  # noqa: E402
from forge.data.dataset import ForgeSplitDataset  # noqa: E402
from forge.eval.coverage import evaluate_split_coverage  # noqa: E402

# P = pre-stress magnitude as a percentage of the nominal reference load 5e-4.
P_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
CONTROL_P = 0.0
RUN_ID = "residual_stress_sweep_eval"
LEVELS = (0.80, 0.90, 0.95)

DATA = ROOT / "data" / "axis_residual_stress_sweep"
OUT_DIR = ROOT / "outputs" / "run_01" / "axis_residual_stress_sweep"
LOG = OUT_DIR / "eval.log"
RESULTS = OUT_DIR / "coverage_results.json"
STATS = ROOT / "data" / "train_stats.json"
CHECKPOINT = ROOT / "outputs" / "run_01" / "checkpoints" / "final.pt"
BASELINE_RESULTS = ROOT / "outputs" / "run_01" / "axis_4_plane_strain" / "coverage_results.json"
CALIBRATION_CONFIG = ROOT / "configs" / "eval.yaml"

# Locked calibration spec values, to 6 dp; the exact thresholds come from BASELINE_RESULTS.
SPEC_Q_HAT = {0.80: 0.001667, 0.90: 0.002565, 0.95: 0.003996}
CONTROL_TOLERANCE = 0.05
PER_P_BUDGET_SEC = 25 * 60

M = 500
FM_STEPS = 20
BATCH_SIZE = 128
SEED = 1000


def baseline_reference(ckpt_step: int) -> tuple[dict[float, float], float]:
    """Locked thresholds and the ID control coverage, read from the baseline run.

    q_hat is only meaningful for the checkpoint and sampling configuration that
    produced it, and the control comparison is only meaningful if this run
    reproduces that configuration, so the provenance is asserted rather than
    assumed. Checked here: checkpoint step and M / FM steps from the results
    file, and M / FM steps / batch size from the baseline config that
    produced it. Batch size matters because `generate_split_scores` derives a
    per-batch sampler seed, so a different batching changes the draws.

    That results file sits under a directory named for its own axis (plane
    strain), but the split read here is `id_test`, the isotropic plane-stress
    split this control compares against, normalized from the same
    train_stats.json.

    Not machine-checkable: the sampler seed. The baseline run passed seed=1000 for
    its evaluation splits (scripts/conformal_eval.py) and SEED here matches, but
    neither the results file nor the config records it. Recording it would mean
    changing the baseline artifact schema, which is out of scope for this
    axis; the mismatch risk is noted rather than silently ignored.

    Both returned values are read out of the file rather than retyped, so
    neither can drift by transcription.
    """
    doc = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    q_hats = {float(k): float(v) for k, v in doc["q_hat"].items()}
    for level, spec in SPEC_Q_HAT.items():
        if round(q_hats[level], 6) != spec:
            raise SystemExit(f"q_hat at {level} is {q_hats[level]!r}, which does not round "
                             f"to the locked spec value {spec}")
    if doc["checkpoint_step"] != ckpt_step:
        raise SystemExit(f"Baseline calibrated on checkpoint step {doc['checkpoint_step']}, "
                         f"this run loaded step {ckpt_step}; q_hat does not transfer")
    if doc["num_samples_per_condition"] != M or doc["num_fm_steps"] != FM_STEPS:
        raise SystemExit(
            f"Baseline sampled M={doc['num_samples_per_condition']} at "
            f"{doc['num_fm_steps']} FM steps; this run uses M={M} at {FM_STEPS}. "
            "The P=0.0 control would not be comparable.")

    scfg = yaml.safe_load(CALIBRATION_CONFIG.read_text(encoding="utf-8"))["sampling"]
    mismatched = {k: (scfg[k], v) for k, v in
                  (("num_samples_per_condition", M), ("num_fm_steps", FM_STEPS),
                   ("batch_size", BATCH_SIZE)) if scfg[k] != v}
    if mismatched:
        raise SystemExit(f"sampling settings differ from the baseline "
                         f"(baseline, this run): {mismatched}")

    id_row = doc["splits"]["id_test"]
    if id_row["split_name"] != "id_test":
        raise SystemExit(f"expected the id_test split, got {id_row['split_name']!r}")
    return q_hats, float(id_row["coverage_at_90"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model, ckpt = load_ema_model(CHECKPOINT, device)
    q_hats, id_cov90_reference = baseline_reference(int(ckpt["step"]))
    vm_std = json.loads(STATS.read_text(encoding="utf-8"))["von_mises"]["std"]
    print(f"checkpoint step {ckpt['step']}, M={M}, {FM_STEPS} FM steps, batch {BATCH_SIZE}")
    print("q_hat (locked, normalized): " +
          ", ".join(f"{lv:.2f}->{q_hats[lv]:.6f}" for lv in LEVELS), flush=True)

    results = {
        # Flipped to "completed" only after every level lands, so a results file
        # left behind by a run that died early cannot be read as a finished one.
        "run_state": "running",
        "run_id": RUN_ID,
        "axis": "C_residual_stress",
        "checkpoint_step": int(ckpt["step"]),
        "num_samples_per_condition": M,
        "num_fm_steps": FM_STEPS,
        "score": "sample_mean_absolute_residual",
        "coverage_definition": "pooled empirical per-pixel coverage; "
                               "no finite-sample split-conformal guarantee claimed",
        "units": "normalized; physical residual = normalized * von_mises_std",
        "von_mises_std": vm_std,
        "q_hat_source": "baseline run, locked and not recomputed",
        "q_hat": {f"{lv:.2f}": q_hats[lv] for lv in LEVELS},
        "pre_stress": {"pattern": "uniform_biaxial", "sigma_ref": 5.0e-4,
                       "physics": "plane_stress",
                       "P_percent_of_sigma_ref": P_VALUES},
        "id_cov90_reference": id_cov90_reference,
        "levels": {},
        "timings_sec": {},
    }

    def flush_results():
        # Write-then-replace, not write_text: the point of flushing after every
        # level is that an interruption keeps the finished ones, and a direct
        # write truncates the file first, so a crash mid-write would take out
        # every level already on disk.
        # Derived here rather than at each stop site, so a new halt path cannot
        # forget to mark the run.
        if "halted" in results:
            results["run_state"] = "halted"
        tmp = RESULTS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
        tmp.replace(RESULTS)

    # Overwrite any completed artifact from a previous invocation up front, so a
    # run that dies during the first level cannot leave stale results in place.
    flush_results()

    for p_percent in P_VALUES:
        split = f"P_{p_percent}"
        ds = ForgeSplitDataset(DATA / split, True, STATS)
        t0 = time.time()
        row = evaluate_split_coverage(
            model, DataLoader(ds, batch_size=BATCH_SIZE), q_hats,
            num_samples_per_condition=M, num_fm_steps=FM_STEPS, batch_size=BATCH_SIZE,
            log_path=LOG, run_id=RUN_ID, worker_id=f"coverage_P_{p_percent}",
            split_name=split, progress_every=50, seed=SEED)
        elapsed = time.time() - t0

        row["P_percent"] = p_percent
        row["pre_stress_p"] = (p_percent / 100.0) * 5.0e-4

        # Validated before it is persisted: a row written first and rejected
        # after would sit in coverage_results.json with no failure status, and
        # read downstream as a completed level. A NaN anywhere in the masked
        # scores poisons both summaries, so finite summaries mean every scored
        # pixel was finite.
        def reject(reason):
            results["halted"] = f"{split}: {reason}"
            flush_results()
            return SystemExit(f"{split}: {reason}; wrote {RESULTS}")

        if row["n_samples"] != len(ds):
            raise reject(f"scored {row['n_samples']} of {len(ds)} conditions")
        for key in ("mean_absolute_residual", "median_absolute_residual"):
            if not math.isfinite(row[key]):
                raise reject(f"{key} is {row[key]}, scores contain NaN or inf")
        for lv in LEVELS:
            cov = row[f"coverage_at_{int(lv * 100)}"]
            if not 0.0 <= cov <= 1.0:
                raise reject(f"coverage_at_{int(lv * 100)} is {cov}, outside [0, 1]")

        results["levels"][split] = row
        results["timings_sec"][split] = elapsed
        flush_results()

        print(f"{split}: n={row['n_samples']} " + " ".join(
            f"cov@{int(lv * 100)}={row[f'coverage_at_{int(lv * 100)}']:.4f}" for lv in LEVELS)
            + f" MAR={row['mean_absolute_residual']:.6f}  [{elapsed / 60:.1f} min]", flush=True)

        if elapsed > PER_P_BUDGET_SEC:
            # the level itself is valid and stays in results; the sweep is what stops
            results["halted"] = (f"{split} took {elapsed / 60:.1f} min, over the "
                                 f"{PER_P_BUDGET_SEC / 60:.0f} min per-level budget")
            flush_results()
            raise SystemExit(f"STOP: {results['halted']}; wrote {RESULTS}")

        if p_percent == CONTROL_P:
            off = abs(row["coverage_at_90"] - id_cov90_reference)
            print(f"control check: P=0.0 cov@90={row['coverage_at_90']:.4f} vs baseline "
                  f"{id_cov90_reference:.4f}, |delta|={off:.4f} "
                  f"(tolerance {CONTROL_TOLERANCE})", flush=True)
            if off > CONTROL_TOLERANCE:
                results["halted"] = "P=0.0 control coverage off baseline ID coverage"
                flush_results()
                raise SystemExit(
                    f"STOP: P=0.0 control cov@90 {row['coverage_at_90']:.4f} deviates from "
                    f"baseline ID {id_cov90_reference:.4f} by {off:.4f} > "
                    f"{CONTROL_TOLERANCE}. The shared pipeline moved; wrote {RESULTS}")

    # coverage gap from the P = 0.0 zero-pre-stress control
    base = results["levels"][f"P_{CONTROL_P}"]
    for split, row in results["levels"].items():
        row[f"coverage_gap_vs_P_{CONTROL_P}"] = {
            f"{lv:.2f}": base[f"coverage_at_{int(lv * 100)}"] - row[f"coverage_at_{int(lv * 100)}"]
            for lv in LEVELS
        }

    results["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    results["total_sec"] = sum(results["timings_sec"].values())
    results["run_state"] = "completed"
    flush_results()

    print("\n| P (% of sigma_ref) | Cov @ 80 | Cov @ 90 | Cov @ 95 | MAR (normalized) |")
    print("|---|---|---|---|---|")
    for p_percent in P_VALUES:
        row = results["levels"][f"P_{p_percent}"]
        label = f"{p_percent} (control)" if p_percent == CONTROL_P else f"{p_percent}"
        print(f"| {label} | {row['coverage_at_80']:.4f} | {row['coverage_at_90']:.4f} | "
              f"{row['coverage_at_95']:.4f} | {row['mean_absolute_residual']:.6f} |")
    print(f"\npeak VRAM {results['peak_vram_bytes'] / 2**30:.2f} GiB, "
          f"total {results['total_sec'] / 60:.1f} min -> wrote {RESULTS}")


if __name__ == "__main__":
    main()

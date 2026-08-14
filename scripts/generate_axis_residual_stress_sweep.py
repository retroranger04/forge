"""Residual-stress-sweep test data: 500 samples at each of seven uniform biaxial pre-stress levels.

Runs entirely in the WSL2 venv, which carries both dolfinx and torch, so this
writes sample_<i>.pt directly and needs none of generate_dataset.py's
npz -> pt convert pass.

Every P value replays the same Halton indices 30500-30999, so samples pair up
one-to-one across the sweep and the pre-stress is the only thing that differs.
The model is not retrained and not touched here; this only produces data.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.fe.generator import SampleTimeoutError, generate_sample  # noqa: E402
from forge.logging import emit  # noqa: E402

OUT = ROOT / "data" / "axis_residual_stress_sweep"
LOG = OUT / "generation.log"
RUN_ID = "residual_stress_sweep_gen"
PHASE = "residual_stress_sweep"

# P = pre-stress magnitude as a percentage of the reference load SIGMA_REF.
P_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
# Nominal reference, fixed by the residual-stress spec, which pins the seven absolute
# magnitudes at 0, 5.0e-7, 1.25e-6, 2.5e-6, 5.0e-6, 1.0e-5, 2.5e-5. It is a
# round mid-decade value, NOT the arithmetic mean of SIGMA_RANGE, which is
# 5.5e-4. The percentages are labels on those fixed magnitudes.
SIGMA_REF = 5.0e-4
HALTON_START, N_SAMPLES = 30500, 500

L_HALF = 0.5              # plate half-width; the ratios below are r over L_HALF
R_OVER_L = (0.10, 0.30)   # hole radius / plate half-width -> r in [0.05, 0.15]
SIGMA_RANGE = (1e-4, 1e-3)
THETA_RANGE = (0.0, 90.0)

PHYSICS = "plane_stress"
TIMEOUT_SEC = 30.0


def pre_stress_of(p_percent):
    """Absolute pre-stress magnitude, in the same units as sigma_inf."""
    return (p_percent / 100.0) * SIGMA_REF


def halton_points():
    """The 500 shared (r_physical, sigma_inf, theta_deg) triples, in index order."""
    eng = qmc.Halton(d=3, scramble=False)
    eng.fast_forward(HALTON_START)
    lo, hi = R_OVER_L
    return [
        (
            (lo + h[0] * (hi - lo)) * L_HALF,
            SIGMA_RANGE[0] + h[1] * (SIGMA_RANGE[1] - SIGMA_RANGE[0]),
            THETA_RANGE[0] + h[2] * (THETA_RANGE[1] - THETA_RANGE[0]),
        )
        for h in eng.random(N_SAMPLES)
    ]


def generate_one_level(p_percent, points):
    """Solve and write all 500 samples for a single P, plus its metadata index.

    Samples land in a staging directory and are promoted only once all 500
    Halton indices are present. The loaders read a split directory rather than
    its metadata.json, so a half-written directory would read as a valid split;
    staging means an interrupted or failed level leaves the previously published
    one untouched instead.
    """
    published = OUT / f"P_{p_percent}"
    stage = OUT / f"P_{p_percent}.staging"
    backup = OUT / f"P_{p_percent}.old"
    # A backup with no published level means a previous run died between the two
    # renames below, so this is the only surviving copy. Put it back before the
    # promote step clears it, or the recovery copy is lost to the next crash.
    if backup.exists() and not published.exists():
        backup.rename(published)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    worker = f"residual_stress_gen_{p_percent}"
    p = pre_stress_of(p_percent)

    t0 = time.perf_counter()
    emit(LOG, worker, RUN_ID, PHASE, "progress", P_percent=p_percent,
         pre_stress_p=f"{p:.6e}", n=N_SAMPLES, first=HALTON_START)

    meta, timed_out = [], 0
    for k, (r, sigma_inf, theta_deg) in enumerate(points):
        idx = HALTON_START + k
        try:
            s = generate_sample(r=r, sigma_inf=sigma_inf, theta_deg=theta_deg,
                                physics=PHYSICS, pre_stress_p=p,
                                timeout_sec=TIMEOUT_SEC)
        except SampleTimeoutError:
            # logged and skipped, never retried: a retry would spend the budget
            # of the samples still ahead of it
            timed_out += 1
            emit(LOG, worker, RUN_ID, PHASE, "timeout", halton_index=idx,
                 r=f"{r:.6f}", sigma_inf=f"{sigma_inf:.6e}", theta_deg=f"{theta_deg:.4f}")
            continue

        torch.save({
            "von_mises": torch.from_numpy(s["von_mises"]),
            "sdf": torch.from_numpy(s["sdf"]),
            "mask": torch.from_numpy(s["mask"]),
            "params": torch.from_numpy(s["params"]),
            "physics": s["physics"],
            "halton_index": idx,
            "pre_stress_p": p, "P_percent": p_percent,
        }, stage / f"sample_{idx}.pt")

        meta.append({"file": f"sample_{idx}.pt", "halton_index": idx,
                     "r": r, "sigma_inf": sigma_inf, "theta_deg": theta_deg,
                     "physics": PHYSICS, "pre_stress_p": p, "P_percent": p_percent})

        if (k + 1) % 50 == 0:
            emit(LOG, worker, RUN_ID, PHASE, "progress", done=k + 1, n=N_SAMPLES,
                 elapsed_sec=int(time.perf_counter() - t0))

    (stage / "metadata.json").write_text(json.dumps(
        {"P_percent": p_percent, "pre_stress_p": p, "physics": PHYSICS,
         "sigma_ref": SIGMA_REF,
         "halton_start": HALTON_START, "halton_end": HALTON_START + N_SAMPLES - 1,
         "expected_size": N_SAMPLES, "count": len(meta), "samples": meta},
        indent=1), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    # Timed-out indices are skipped, never retried, but a short level breaks the
    # one-to-one pairing the sweep depends on. An incomplete stage is discarded
    # rather than promoted, so the previous published level survives intact and
    # no half-populated directory is ever visible to a loader.
    expected_files = {f"sample_{HALTON_START + k}.pt" for k in range(N_SAMPLES)}
    complete = {q.name for q in stage.glob("sample_*.pt")} == expected_files
    if complete:
        # Swap through a backup rather than deleting first. Between the delete
        # and the rename the level was absent from disk entirely; between the
        # two renames the previous one is still there under .old.
        shutil.rmtree(backup, ignore_errors=True)
        if published.exists():
            published.rename(backup)
        stage.rename(published)
        shutil.rmtree(backup, ignore_errors=True)
    else:
        emit(LOG, worker, RUN_ID, PHASE, "error", msg="incomplete_level_not_published",
             saved=len(meta), expected=N_SAMPLES)
        shutil.rmtree(stage)
    emit(LOG, worker, RUN_ID, PHASE, "complete", saved=len(meta), timeouts=timed_out,
         published=str(complete).lower(), elapsed_sec=int(elapsed))
    print(f"P={p_percent}%: saved {len(meta)}/{N_SAMPLES}, timeouts {timed_out}, "
          f"{elapsed / 60:.1f} min", flush=True)
    return len(meta), timed_out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    points = halton_points()

    # The sweep spans two form signatures: P=0 keeps the original von Mises
    # expression, every nonzero P uses the pre-stress one. A cold FFCx compile
    # on either runs past the per-sample timeout and would discard that whole
    # level, so both are warmed up front and the timeout only ever guards real
    # solve time. pre_stress_p is a runtime Constant, so one nonzero warmup
    # covers all six nonzero P values.
    t0 = time.perf_counter()
    for warmup_p in (0.0, pre_stress_of(P_VALUES[-1])):
        generate_sample(r=0.10, sigma_inf=SIGMA_REF, theta_deg=30.0, physics=PHYSICS,
                        pre_stress_p=warmup_p, timeout_sec=900.0)
    print(f"warmed both forms in {time.perf_counter() - t0:.1f}s", flush=True)

    total_saved, total_timeouts = 0, 0
    for p_percent in P_VALUES:
        saved, timeouts = generate_one_level(p_percent, points)
        total_saved += saved
        total_timeouts += timeouts
    expected = len(P_VALUES) * N_SAMPLES
    print(f"sweep complete: {total_saved}/{expected} samples, {total_timeouts} timeouts")
    if total_saved != expected:
        raise SystemExit(f"incomplete sweep: {total_saved}/{expected} samples written; "
                         "the paired-sample structure is broken, do not evaluate this data")

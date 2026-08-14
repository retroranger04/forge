"""Anisotropy-sweep test data: 500 orthotropic FE samples at each of six anisotropy ratios.

Runs entirely in the WSL2 venv, which carries both dolfinx and torch, so this
writes sample_<i>.pt directly and needs none of generate_dataset.py's
npz -> pt convert pass.

Every ratio replays the same Halton indices 30000-30499, so samples pair up
one-to-one across the sweep and the material is the only thing that differs.
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

OUT = ROOT / "data" / "axis_anisotropy_sweep"
LOG = OUT / "generation.log"
RUN_ID = "anisotropy_sweep_gen"
PHASE = "anisotropy_sweep"

R_VALUES = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0]  # R = E1/E2, stiff direction along x
HALTON_START, N_SAMPLES = 30000, 500

L_HALF = 0.5              # plate half-width; the ratios below are r over L_HALF
R_OVER_L = (0.10, 0.30)   # hole radius / plate half-width -> r in [0.05, 0.15]
SIGMA_RANGE = (1e-4, 1e-3)
THETA_RANGE = (0.0, 90.0)

E1 = 1.0        # stiff (fiber) direction modulus, fixed across the sweep
NU12 = 0.3      # matches the isotropic training material
G12 = 0.384615  # E1 / (2 * (1 + NU12)), held at the isotropic-equivalent value
PHYSICS = "plane_stress"
TIMEOUT_SEC = 30.0


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


def generate_one_ratio(ratio, points):
    """Solve and write all 500 samples for a single R, plus its metadata index.

    Samples land in a staging directory and are promoted only once all 500
    Halton indices are present. The loaders read a split directory rather than
    its metadata.json, so a half-written directory would read as a valid split;
    staging means an interrupted or failed ratio leaves the previously published
    one untouched instead.
    """
    published = OUT / f"R_{ratio}"
    stage = OUT / f"R_{ratio}.staging"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    worker = f"anisotropy_gen_{ratio}"
    e2 = E1 / ratio

    t0 = time.perf_counter()
    emit(LOG, worker, RUN_ID, PHASE, "progress", R=ratio, E2=f"{e2:.6f}",
         n=N_SAMPLES, first=HALTON_START)

    meta, timed_out = [], 0
    for k, (r, sigma_inf, theta_deg) in enumerate(points):
        idx = HALTON_START + k
        try:
            s = generate_sample(r=r, sigma_inf=sigma_inf, theta_deg=theta_deg,
                                physics=PHYSICS, E1=E1, E2=e2, nu12=NU12, G12=G12,
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
            "E1": E1, "E2": e2, "nu12": NU12, "G12": G12, "R": ratio,
        }, stage / f"sample_{idx}.pt")

        meta.append({"file": f"sample_{idx}.pt", "halton_index": idx,
                     "r": r, "sigma_inf": sigma_inf, "theta_deg": theta_deg,
                     "physics": PHYSICS, "E1": E1, "E2": e2, "nu12": NU12,
                     "G12": G12, "R": ratio})

        if (k + 1) % 50 == 0:
            emit(LOG, worker, RUN_ID, PHASE, "progress", done=k + 1, n=N_SAMPLES,
                 elapsed_sec=int(time.perf_counter() - t0))

    (stage / "metadata.json").write_text(json.dumps(
        {"R": ratio, "physics": PHYSICS, "E1": E1, "E2": e2, "nu12": NU12, "G12": G12,
         "halton_start": HALTON_START, "halton_end": HALTON_START + N_SAMPLES - 1,
         "expected_size": N_SAMPLES, "count": len(meta), "samples": meta},
        indent=1), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    # Timed-out indices are skipped, never retried, but a short ratio breaks the
    # one-to-one pairing the sweep depends on. An incomplete stage is discarded
    # rather than promoted, so the previous published ratio survives intact and
    # no half-populated directory is ever visible to a loader.
    expected_files = {f"sample_{HALTON_START + k}.pt" for k in range(N_SAMPLES)}
    complete = {p.name for p in stage.glob("sample_*.pt")} == expected_files
    if complete:
        # Swap through a backup rather than deleting first. Between the delete
        # and the rename the ratio was absent from disk entirely; between the
        # two renames the previous one is still there under .old.
        backup = OUT / f"R_{ratio}.old"
        shutil.rmtree(backup, ignore_errors=True)
        if published.exists():
            published.rename(backup)
        stage.rename(published)
        shutil.rmtree(backup, ignore_errors=True)
    else:
        emit(LOG, worker, RUN_ID, PHASE, "error", msg="incomplete_ratio_not_published",
             saved=len(meta), expected=N_SAMPLES)
        shutil.rmtree(stage)
    emit(LOG, worker, RUN_ID, PHASE, "complete", saved=len(meta), timeouts=timed_out,
         published=str(complete).lower(), elapsed_sec=int(elapsed))
    print(f"R={ratio}: saved {len(meta)}/{N_SAMPLES}, timeouts {timed_out}, "
          f"{elapsed / 60:.1f} min", flush=True)
    return len(meta), timed_out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    points = halton_points()

    # The orthotropic form is new to the FFCx cache, and a cold compile there
    # runs past the per-sample timeout. One serial pass up front means the
    # timeout only ever guards real solve time. Constant stiffness keeps this
    # to a single form signature, so one warmup covers every ratio.
    t0 = time.perf_counter()
    generate_sample(r=0.10, sigma_inf=5e-4, theta_deg=30.0, physics=PHYSICS,
                    E1=E1, E2=E1 / 2.0, nu12=NU12, G12=G12, timeout_sec=900.0)
    print(f"warmed orthotropic form in {time.perf_counter() - t0:.1f}s", flush=True)

    total_saved, total_timeouts = 0, 0
    for ratio in R_VALUES:
        saved, timeouts = generate_one_ratio(ratio, points)
        total_saved += saved
        total_timeouts += timeouts
    expected = len(R_VALUES) * N_SAMPLES
    print(f"sweep complete: {total_saved}/{expected} samples, {total_timeouts} timeouts")
    if total_saved != expected:
        raise SystemExit(f"incomplete sweep: {total_saved}/{expected} samples written; "
                         "the paired-sample structure is broken, do not evaluate this data")

"""Halton-sampled FE dataset generation for FORGE.

Two modes, because dolfinx lives in the WSL2 venv and torch lives in the
Windows venv:

  --mode generate  (WSL2)    solve and write data/<split>/sample_<i>.npz
  --mode convert   (Windows) rewrite those as sample_<i>.pt, drop the .npz

Splits share Halton indices 10000-10999 between id_test and ood_fidelity on
purpose: the paired-sample analysis needs identical (r, sigma_inf, theta) with
only the plane-stress/plane-strain physics differing.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOG = DATA / "generation.log"

L_HALF = 0.5  # plate half-width; spec r/L ratios are relative to this
SIGMA_RANGE = (1e-4, 1e-3)
THETA_RANGE = (0.0, 90.0)

SPLITS = {
    "train":        dict(physics="plane_stress", start=0,     size=10000, r_over_L=(0.10, 0.30)),
    "id_test":      dict(physics="plane_stress", start=10000, size=1000,  r_over_L=(0.10, 0.30)),
    "ood_fidelity": dict(physics="plane_strain", start=10000, size=1000,  r_over_L=(0.10, 0.30)),
    "ood_geometry": dict(physics="plane_stress", start=11000, size=500,   r_over_L=(0.30, 0.40)),
}


def log(line):
    with LOG.open("a") as f:
        f.write(line + "\n")


def halton_params(split, index):
    """Deterministic (r, sigma_inf, theta_deg) for one absolute Halton index."""
    from scipy.stats import qmc

    eng = qmc.Halton(d=3, scramble=False)
    eng.fast_forward(index)
    h = eng.random(1)[0]
    lo, hi = SPLITS[split]["r_over_L"]
    return (
        (lo + h[0] * (hi - lo)) * L_HALF,
        SIGMA_RANGE[0] + h[1] * (SIGMA_RANGE[1] - SIGMA_RANGE[0]),
        THETA_RANGE[0] + h[2] * (THETA_RANGE[1] - THETA_RANGE[0]),
    )


def warmup():
    """Compile both form variants into the JIT cache before workers fan out.

    FFCx caches to XDG_CACHE_HOME, which we pin inside the workspace; a cold
    compile there runs well past the 30 s per-sample budget, and concurrent
    workers racing the same cache entry deadlock it. One serial pass up front
    removes both problems, so the timeout only ever guards real solve time.
    """
    from forge.fe.generator import generate_sample

    for physics in ("plane_stress", "plane_strain"):
        t0 = time.perf_counter()
        generate_sample(r=0.10, sigma_inf=5e-4, theta_deg=30.0,
                        physics=physics, timeout_sec=900.0)
        print(f"warmed {physics} in {time.perf_counter() - t0:.1f}s", flush=True)


def generate(split, shard_start, shard_count):
    from scipy.stats import qmc

    from forge.fe.generator import SampleTimeoutError, generate_sample

    spec = SPLITS[split]
    out_dir = DATA / split
    out_dir.mkdir(parents=True, exist_ok=True)

    first = spec["start"] + shard_start
    n = shard_count if shard_count is not None else spec["size"] - shard_start
    eng = qmc.Halton(d=3, scramble=False)
    eng.fast_forward(first)
    pts = eng.random(n)

    lo, hi = spec["r_over_L"]
    t0 = time.perf_counter()
    # Tag carries the shard's first Halton index: train is sharded across
    # workers to fit the 90-min budget, and the watchdog must be able to tell
    # a stalled shard from its still-running siblings in the shared log.
    tag = f"{split}#{first}"
    log(f"[{tag}] START N={n} first={first} t={time.time():.0f}")

    done, timed_out = 0, 0
    for k, h in enumerate(pts):
        idx = first + k
        r = (lo + h[0] * (hi - lo)) * L_HALF
        sigma = SIGMA_RANGE[0] + h[1] * (SIGMA_RANGE[1] - SIGMA_RANGE[0])
        theta = THETA_RANGE[0] + h[2] * (THETA_RANGE[1] - THETA_RANGE[0])
        try:
            s = generate_sample(r=r, sigma_inf=sigma, theta_deg=theta,
                                physics=spec["physics"], timeout_sec=30.0)
        except SampleTimeoutError:
            timed_out += 1
            log(f"[{tag}] TIMEOUT halton_index={idx} r={r} sigma={sigma} theta={theta}")
            continue
        np.savez(out_dir / f"sample_{idx}.npz",
                 von_mises=s["von_mises"], sdf=s["sdf"], mask=s["mask"],
                 params=s["params"], physics=s["physics"], halton_index=idx)
        done += 1
        if (k + 1) % 100 == 0:
            log(f"[{tag}] {k + 1}/{n} elapsed={time.perf_counter() - t0:.1f}s")

    log(f"[{tag}] DONE shard first={first} n={n} saved={done} "
        f"timeouts={timed_out} elapsed={time.perf_counter() - t0:.1f}s")


def convert(split):
    """.npz -> .pt on the Windows venv, then drop the .npz."""
    import torch

    out_dir = DATA / split
    files = sorted(out_dir.glob("sample_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    meta = []
    for p in files:
        # NpzFile keeps the archive open; close it before unlinking, and pull
        # every value out first since the handle is dead afterwards.
        with np.load(p, allow_pickle=False) as z:
            idx = int(z["halton_index"])
            params = z["params"]
            physics = str(z["physics"])
            rec = {
                "von_mises": torch.from_numpy(z["von_mises"]),
                "sdf": torch.from_numpy(z["sdf"]),
                "mask": torch.from_numpy(z["mask"]),
                "params": torch.from_numpy(params),
                "physics": physics,
                "halton_index": idx,
            }
        torch.save(rec, out_dir / f"sample_{idx}.pt")
        p.unlink()
        meta.append({"halton_index": idx, "params": params.tolist(),
                     "physics": physics, "file": f"sample_{idx}.pt"})

    (out_dir / "metadata.json").write_text(json.dumps(
        {"split": split, "physics": SPLITS[split]["physics"],
         "expected_size": SPLITS[split]["size"],
         "halton_start": SPLITS[split]["start"],
         "halton_end": SPLITS[split]["start"] + SPLITS[split]["size"] - 1,
         "count": len(meta), "samples": meta}, indent=1))
    print(f"{split}: converted {len(meta)} samples")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["generate", "convert", "warmup"], required=True)
    ap.add_argument("--split", choices=list(SPLITS))
    ap.add_argument("--shard-start", type=int, default=0, help="offset within the split")
    ap.add_argument("--shard-count", type=int, default=None)
    a = ap.parse_args()
    if a.mode == "warmup":
        warmup()
    elif a.mode == "generate":
        generate(a.split, a.shard_start, a.shard_count)
    else:
        convert(a.split)

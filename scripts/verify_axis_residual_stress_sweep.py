"""Post-generation verification for the residual stress sweep.

Checks counts, on-disk parameter ranges in the same units as the spec, the
stored pre-stress metadata, and the paired-sample structure at one shared
Halton index.
"""
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_axis_residual_stress_sweep import (  # noqa: E402
    N_SAMPLES, P_VALUES, pre_stress_of,
)

OUT = ROOT / "data" / "axis_residual_stress_sweep"
PAIR_IDX = 30750
R_RANGE, SIGMA_RANGE, THETA_RANGE = (0.05, 0.15), (1e-4, 1e-3), (0.0, 90.0)

random.seed(0)
failures = []


def check(ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}", flush=True)
    if not ok:
        failures.append(msg)


print("=== counts ===", flush=True)
for p in P_VALUES:
    d = OUT / f"P_{p}"
    n = len(list(d.glob("sample_*.pt"))) if d.is_dir() else -1
    check(n == N_SAMPLES, f"P_{p}: {n} samples (expected {N_SAMPLES})")

print("\n=== 20 random samples per level: ranges, physics, pre-stress metadata ===",
      flush=True)
for p in P_VALUES:
    d = OUT / f"P_{p}"
    files = sorted(d.glob("sample_*.pt"))
    if not files:
        check(False, f"P_{p}: no samples to inspect")
        continue
    expected_p = pre_stress_of(p)
    rs, sgs, ths = [], [], []
    bad_physics = bad_p = bad_pct = 0
    for f in random.sample(files, min(20, len(files))):
        s = torch.load(f, map_location="cpu", weights_only=False)
        r, sg, th = (float(v) for v in s["params"])
        rs.append(r); sgs.append(sg); ths.append(th)
        bad_physics += s["physics"] != "plane_stress"
        # float32 round-trip of the params tuple; pre_stress_p is stored as a
        # Python float, so compare with a float32-scale tolerance.
        bad_p += abs(float(s["pre_stress_p"]) - expected_p) > 1e-12
        bad_pct += float(s["P_percent"]) != float(p)
    print(f"P_{p}: r [{min(rs):.4f}, {max(rs):.4f}]  "
          f"sigma_inf [{min(sgs):.4e}, {max(sgs):.4e}]  "
          f"theta [{min(ths):.2f}, {max(ths):.2f}]  "
          f"pre_stress_p={expected_p:.6e}", flush=True)
    check(R_RANGE[0] - 1e-4 <= min(rs) and max(rs) <= R_RANGE[1] + 1e-4,
          f"P_{p}: r_physical within {R_RANGE}")
    check(SIGMA_RANGE[0] - 1e-9 <= min(sgs) and max(sgs) <= SIGMA_RANGE[1] + 1e-9,
          f"P_{p}: sigma_inf within {SIGMA_RANGE}")
    check(THETA_RANGE[0] - 1e-3 <= min(ths) and max(ths) <= THETA_RANGE[1] + 1e-3,
          f"P_{p}: theta_deg within {THETA_RANGE}")
    check(bad_physics == 0, f"P_{p}: physics label plane_stress on all 20")
    check(bad_p == 0, f"P_{p}: stored pre_stress_p matches {expected_p:.6e} on all 20")
    check(bad_pct == 0, f"P_{p}: P_percent matches directory name on all 20")

    md = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    check(md["count"] == N_SAMPLES and md["P_percent"] == p,
          f"P_{p}: metadata.json count={md['count']} P_percent={md['P_percent']}")

print(f"\n=== paired structure at Halton index {PAIR_IDX} ===", flush=True)
paired = {}
for p in P_VALUES:
    f = OUT / f"P_{p}" / f"sample_{PAIR_IDX}.pt"
    if f.exists():
        paired[p] = torch.load(f, map_location="cpu", weights_only=False)
check(len(paired) == len(P_VALUES), f"sample_{PAIR_IDX}.pt present in all 7 levels")

if len(paired) == len(P_VALUES):
    ref = paired[P_VALUES[0]]
    r, sg, th = (float(v) for v in ref["params"])
    print(f"  shared triple: r={r:.6f}  sigma_inf={sg:.6e}  theta_deg={th:.4f}", flush=True)
    check(all(torch.equal(s["params"], ref["params"]) for s in paired.values()),
          "(r, sigma_inf, theta_deg) bitwise-identical across all 7 levels")

    ps = [float(paired[p]["pre_stress_p"]) for p in P_VALUES]
    print("  pre_stress_p: " + ", ".join(f"{v:.3e}" for v in ps), flush=True)
    check(len(set(ps)) == len(P_VALUES) and ps[0] == 0.0,
          f"pre_stress_p distinct across all 7, from 0.0 to {ps[-1]:.3e}")

    fields = [paired[p]["von_mises"] for p in P_VALUES]
    distinct = all(
        not torch.equal(fields[i], fields[j])
        for i in range(len(fields)) for j in range(i + 1, len(fields))
    )
    maes = [float((fields[i] - fields[0]).abs().mean()) for i in range(len(P_VALUES))]
    print("  MAE vs P=0.0: " + ", ".join(f"P_{P_VALUES[i]}={maes[i]:.3e}"
                                         for i in range(len(P_VALUES))), flush=True)
    check(distinct, "von_mises fields pairwise distinct across all 7 levels")
    # Reported, not gated. Von Mises is quadratic in the added pre-stress, and
    # d(vm^2)/dp = sxx + syy + 2p is negative wherever the in-plane trace is
    # compressive, so a field-level MAE need not rise monotonically in P for
    # valid data. Gating on it could reject a correct sweep.
    mono = all(maes[i] < maes[i + 1] for i in range(len(maes) - 1))
    print(f"  MAE vs P=0.0 monotonic in P: {mono}  (diagnostic, not a gate)", flush=True)

print(f"\n{'ALL CHECKS PASSED' if not failures else str(len(failures)) + ' CHECK(S) FAILED'}",
      flush=True)
if failures:
    for m in failures:
        print(f"  - {m}")
    raise SystemExit(1)

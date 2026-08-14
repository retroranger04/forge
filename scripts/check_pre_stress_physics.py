"""Residual pre-stress sanity check: does uniform biaxial pre-stress move the von Mises field?

Two gates, both mandatory before generating the sweep:
  1. relative MAE >= 1% between p=0 and p=2.5e-5 (P=5.0%) at three (r, theta).
  2. omitting pre_stress_p is bit-identical to passing pre_stress_p=0.0.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forge.fe.generator import generate_sample  # noqa: E402

SIGMA_INF = 5.0e-4      # nominal reference load, matches SIGMA_REF in the sweep script
P_MAX = 2.5e-5          # the P = 5.0% magnitude fixed by the residual-stress spec
PHYSICS = "plane_stress"
POINTS = [(0.10, 0.0), (0.06, 30.0), (0.14, 60.0)]


def solve(r, theta_deg, **kw):
    return generate_sample(r=r, sigma_inf=SIGMA_INF, theta_deg=theta_deg,
                           physics=PHYSICS, timeout_sec=900.0, **kw)


print("=== gate 1: p=0 vs p=2.5e-5 (P=5.0%) ===", flush=True)
rel_maes = []
for r, theta in POINTS:
    a = solve(r, theta, pre_stress_p=0.0)["von_mises"]
    b = solve(r, theta, pre_stress_p=P_MAX)["von_mises"]
    mae = float(np.abs(b - a).mean())
    max_abs = float(np.abs(b - a).max())
    rel = mae / float(np.abs(a).mean())
    rel_maes.append(rel)
    print(f"r={r:.2f} theta={theta:.0f}deg  MAE={mae:.6e}  max|diff|={max_abs:.6e}  "
          f"peak_p0={a.max():.6e}  peak_p5={b.max():.6e}  rel_MAE={rel * 100:.3f}%",
          flush=True)

print("\n=== gate 2: default path bit-identical ===", flush=True)
no_arg = solve(0.10, 0.0)["von_mises"]
explicit = solve(0.10, 0.0, pre_stress_p=0.0)["von_mises"]
mae0 = float(np.abs(explicit - no_arg).mean())
identical = bool(np.array_equal(no_arg, explicit))
print(f"MAE={mae0:.6e}  bitwise_equal={identical}", flush=True)

ok1 = all(x >= 0.01 for x in rel_maes)
ok2 = identical and mae0 == 0.0
print(f"\ngate1_rel_mae_ge_1pct={ok1}  gate2_bit_identical={ok2}", flush=True)
if not (ok1 and ok2):
    raise SystemExit("SANITY CHECK FAILED")
print("SANITY CHECK PASSED", flush=True)

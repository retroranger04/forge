# FORGE Session Four report — FEniCSx parametric data generation

Date: 2026-08-09. Scope: FE dataset generation, physics validation, dataset packaging.
No model training and no ML code this session.

## 1. Preflight

| Check | Result |
|---|---|
| `git status` clean | Pass (clean, no untracked files at session start) |
| CLAUDE.md explicit-venv rule applied | Pass (`grep -c "explicit venv path"` = 2) |
| Windows venv `.venv\Scripts\python.exe -c "print('ok')"` | Pass (`ok`) |
| WSL2 dolfinx version | Pass (`0.11.0`) |
| Free disk on A: | Pass (30.87 GB, bar 5 GB) |
| `data/` empty except `.gitkeep` | Pass |

One preflight gap: `torch` is not installed in `.venv-wsl` (`ModuleNotFoundError: No module named 'torch'`),
while dolfinx is not available on the Windows side. Package installation was prohibited this session,
so generation writes a per-sample `.npz` intermediate on the WSL2 side and a Windows-side pass
converts each to the specified `.pt` and deletes the `.npz`. The delivered artifact format is
unchanged. Supporting versions: gmsh 4.15.2, scipy 1.18.0, numpy 2.5.1 (WSL2); torch 2.6.0+cu124 (Windows).

## 2. Physics validation (Task 3)

Task 3 required two physics corrections during Session Four, both approved by the user before
Task 4 began. They are catalogued here and in the commit message.

**Correction 1, Case 2 reference value.** The original acceptance target of 4.83e-3 was not
reproducible. Measured peak was 3.360e-3, a 30.4% miss against a 10% bar, while Case 1 matched the
Kirsch infinite-plate limit to within 0.889%. The discrepancy was traced to the reference, not the
solver: at r/L = 0.20 with L = 0.5 the hole diameter is 0.20 against a plate width of 1.0, so
d/W = 0.20, not the 0.40 assumed in the original spec, and 4.83e-3 additionally folded in a
net-section to gross-section conversion. Case 2 was retargeted to the Peterson gross-section value
K_tg = 3.14 at d/W = 0.20. The solver was not modified.

**Correction 2, von Mises formula.** The originally specified 2D form
`sqrt(sxx^2 + syy^2 - sxx*syy + 3*sxy^2)` omits sigma_zz. For a 2D traction-only boundary value
problem with a traction-free hole the in-plane stresses are independent of the elastic constants,
so under that formula plane strain and plane stress differ by only 0.0313%, which would have made
the OOD-fidelity split a near-duplicate of ID test. A diagnostic run measured the alternative: with
sigma_zz = nu*(sxx + syy) included via the full 3D form, the two differ by 11.366%. The generator now
uses the full 3D formula, which reduces exactly to the 2D form when sigma_zz = 0, leaving all
plane-stress splits numerically unchanged.

Final results, all three cases passing:

| Case | Geometry | Physics | Measured peak sigma_vm | Reference | Rel. error | Bar | Result |
|---|---|---|---|---|---|---|---|
| 1 | r/L = 0.05 | plane stress | 3.026675e-03 | 3.00e-03 (Kirsch, 3*sigma_inf) | +0.889% | 10% | Pass |
| 2 | r/L = 0.20 | plane stress | 3.360440e-03 | 3.14e-03 (Peterson K_tg, d/W = 0.20) | +7.020% | 10% | Pass |
| 3 | r/L = 0.20 | plane strain | 2.978497e-03 | 3.360440e-03 (Case 2) | -11.366% | 15% | Pass |

Sign convention was verified independently before the batch runs: for theta = 0 the near-hole peak
lands at grid point (-0.0079, 0.1032), transverse to the applied x-direction tension as expected,
and the far-field von Mises at x = 0.45 reads 9.09e-04 against sigma_inf = 1.0e-03.

Peak stresses for validation are read from the FE field in a thin band at the hole boundary rather
than from the 64x64 output grid, which under-resolves small holes. `generate_sample` therefore
returns one key beyond the specified set, `peak_hole_von_mises`; it is not written to disk.

## 3. Sub-agent dispatch, wall clock, watchdog

The spec's "one sub-agent per split" allocation is not compatible with its own 90-minute
per-sub-agent budget: train is 10,000 of the 12,500 samples, so on a single worker it needs roughly
133 minutes at the measured per-sample cost regardless of how the other splits are parallelised.
Train was therefore sharded across 5 workers of 2,000 samples each. No timeout or watchdog
threshold was altered. Dispatch ran in two waves to respect the 5-concurrent-sub-agent ceiling.

| Wave | Worker | Samples | Saved | Timeouts | Elapsed |
|---|---|---|---|---|---|
| 1 | train#0 | 2000 | 2000 | 0 | 873.3 s |
| 1 | train#2000 | 2000 | 2000 | 0 | 879.2 s |
| 1 | train#4000 | 2000 | 2000 | 0 | 875.8 s |
| 1 | train#6000 | 2000 | 2000 | 0 | 878.2 s |
| 1 | train#8000 | 2000 | 2000 | 0 | 876.2 s |
| 2 | id_test#10000 | 1000 | 1000 | 0 | 380.3 s |
| 2 | ood_fidelity#10000 | 1000 | 1000 | 0 | 378.0 s |
| 2 | ood_geometry#11000 | 500 | 500 | 0 | 139.5 s |

Total worker time 5280.5 s against roughly 1259 s of generation wall clock, an effective
parallelisation of about 4.2x, at 0.422 s per sample. Windows-side `.npz` to `.pt` conversion ran
concurrently with wave 2 where possible.

**Watchdog triggers: zero genuine stalls and zero sample timeouts across all 12,500 solves.**

The Level 1 per-sample timeout did fire once, correctly, during Task 3 setup: a cold FFCx
compilation on the `/mnt/a` drvfs mount exceeded the 30 s budget. This was a cold-cache artifact
rather than a slow solve, and is addressed by the warmup pass described in section 6.

## 4. Post-generation validation (Task 5)

Four checks dispatched in parallel, all passing.

- **A, file counts.** train 10000/10000, id_test 1000/1000, ood_fidelity 1000/1000,
  ood_geometry 500/500. Zero timed-out Halton indices in the log. Pass.
- **B, pairing.** 1000 shared Halton indices between id_test and ood_fidelity. On 10 sampled
  indices, `(r, sigma_inf, theta)` match exactly, physics reads plane_stress against plane_strain,
  and the von Mises fields differ, independently confirming correction 2 reached the data. Pass.
- **C, integrity.** One random sample per split: fields (64, 64), params length 3, dtypes
  float32/float32/uint8/float32, no NaN or inf, and mask == 1 exactly where SDF > 0. Pass.
- **D, statistics.** Written to `data/dataset_summary.md`. All parameter ranges within spec. Pass.

Check D initially reported `in_range=False` for train. The cause was the check, not the data:
train is the only split containing Halton index 0, whose unscrambled point is exactly (0, 0, 0) and
therefore lands on every lower bound. Stored as float32, `sigma_inf` becomes 9.999999747378752e-05,
which is 2.53e-12 below the nominal 1.0e-04, outside the 1e-12 absolute tolerance the check used.
The tolerance was corrected to be relative to float32 precision and check D re-run to a pass.

## 5. Final dataset statistics

12,500 samples, 0.455 GB on disk, no residual `.npz`.

| Split | Count | Physics | von Mises max | von Mises mean | von Mises std |
|---|---|---|---|---|---|
| train | 10000 | plane_stress | 3.674905e-03 | 5.600024e-04 | 3.139922e-04 |
| id_test | 1000 | plane_stress | 3.562284e-03 | 5.606173e-04 | 3.139312e-04 |
| ood_fidelity | 1000 | plane_strain | 3.156278e-03 | 4.974559e-04 | 2.800838e-04 |
| ood_geometry | 500 | plane_stress | 4.344175e-03 | 5.870524e-04 | 4.193219e-04 |

von Mises minimum is 0.0 in every split, as expected given the hole interior is set to zero.
The id_test against ood_fidelity mean difference of roughly 11.3% is the misspecification signal
the OOD-fidelity split exists to expose, and ood_geometry's higher maximum is consistent with its
larger holes. Full per-parameter statistics are in `data/dataset_summary.md`.

## 6. Anomalies during generation

Zero sample timeouts, zero stalls, zero splits incomplete. Five implementation defects were found
and fixed during the session:

1. **FFCx recompilation per theta.** `theta` was folded into the UFL form as a literal, giving every
   sample a distinct form signature and triggering a fresh C compilation per solve. `nhat` and
   `sigma_inf` are now runtime `fem.Constant` values. Verified inert: Case 1 returns
   0.003026675078834737 both before and after the change.
2. **JIT cache outside the workspace and racing.** FFCx defaulted to `/home/arpit/.cache/fenics`,
   outside the workspace, and three concurrent validation processes deadlocked on the same cache
   entry (`FileExistsError` escalating to `TimeoutError`). `XDG_CACHE_HOME` is now pinned to the
   in-workspace `.jit-cache/` (gitignored), and a serial `--mode warmup` pass compiles both form
   variants before any worker fans out, so the per-sample timeout only ever guards real solve time.
3. **Untagged heartbeats.** Heartbeat lines were emitted as `[train]` rather than
   `[train#<first>]`, so with 5 concurrent train shards the watchdog could not attribute progress
   and would have declared all five stalled at the 180 s threshold. Caught 133 s into the first
   wave-1 launch; that run was killed, its 1,782 partial `.npz` files discarded, and the wave
   relaunched after the fix. Roughly 2.5 minutes of work lost.
4. **Untagged completion lines.** The `DONE` line had the same defect. The watchdog was made to
   reconstruct the shard tag from the `first=` field it already carries, and the emitter corrected,
   so no rerun was needed.
5. **Conversion held the `.npz` handle open.** `np.load` returns a lazy `NpzFile`, so `unlink()`
   raised `PermissionError` on Windows, and values were being read after the unlink. Conversion now
   reads within a context manager and extracts all values before deleting.

Defects 3 and 4 were in the watchdog and its emitter, meaning the stall detection this session was
built to provide was itself briefly non-functional. Both were caught by direct inspection of the
log before any false kill occurred, and the corrected watchdog ran clean for the remainder.

## 7. Ready for Session Five

**Yes.** All four splits are 100% complete against the 95% bar, with zero timeouts and zero stalls.
Physics is validated against Kirsch and Peterson references and the plane-strain differential is
confirmed present in the delivered data.

Two items to carry forward. First, the WSL2 environment has no `torch`, so any future WSL2-side
work producing `.pt` output needs either the conversion pass used here or a package installation
decision. Second, generation depends on the warmed in-workspace JIT cache; a cold cache costs about
11 s for plane stress and 20 s for plane strain, and running workers concurrently against a cold
cache will deadlock it.

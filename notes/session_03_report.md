# FORGE — Session Three Report: Hardware Benchmark Suite

**Date**: 2026-08-09

## 1. Preflight verification

All three preflight checks passed before any benchmark work began:

1. `git status`: clean except the three expected untracked report files (`session_01_report.md`, `session_02_report.md`, `session_2_5_report.md`).
2. Windows PyTorch venv: `torch.cuda.is_available()` → `True`, GPU → `NVIDIA GeForce RTX 4060 Laptop GPU`.
3. WSL2 FEniCSx venv: `dolfinx.__version__` → `0.11.0`.

## Task 0 (added mid-session): missing packages

Before B1 and B5 could run, `psutil` and `h5py` were found absent from `.venv` — both are named explicitly in the session spec's B1/B5 method descriptions. Per the session's explicit "STOP and report, do not pip install" rule, this was raised to the user rather than resolved unilaterally. The user approved installing both; they were added (`psutil==7.2.2`, `h5py==3.16.0`) and `requirements-windows.txt` was refreshed via `pip freeze`. No other packages were installed this session.

## 2. Benchmark summaries

**B1 — Environment deep-probe.** Sequential, single-shot. Captured RAM (25.48 GB total / 9.51 GB free), GPU/VRAM (8585 MB total, 7444 MB free pre-allocation, compute capability 8.9), disk free (33.14 GB on A:), torch build info (2.6.0+cu124, cuDNN 90100), and the WSL2-side versions (Ubuntu 24.04.4, Python 3.12.13, dolfinx 0.11.0, gmsh 4.15.2, MPICH 5.0.1). No issues.

**B2 — FEniCSx per-sample cost.** A dedicated agent wrote and validated `b2_fenicsx_plate.py` (plate-with-hole linear elasticity via gmsh + dolfinx), then 4 sub-agents ran the 3×4 timing sweep (12/12 solves succeeded). Fastest config: 64×64 plane strain (median 0.331 s/sample). Slowest by median: 128×128 plane strain (median 1.627 s/sample). The Kirsch sanity check (r=0.20, 64×64, plane stress) came in at 50.71% relative error against the 3σ_∞ infinite-plate reference, well outside the session's 15% acceptance bar. The validating agent's follow-up work (mesh-convergence study, net-section stress estimate, small-r/L control run) indicates this is a genuine finite-width-plate effect at r/L=0.20 rather than a script bug — flagged for the strategy chat rather than silently accepted or worked around.

**B3 — DiT training step timing and VRAM.** A dedicated agent wrote and validated `b3_dit_train_step.py` (hand-rolled DiT: self-attention + AdaLN-Zero + cross-attention, quadratic and hand-rolled linear-attention variants). The full 64-config sweep (2 resolutions × 2 depths × 2 hidden dims × 2 attention types × 4 batch sizes) completed with 64/64 configs at `status=OK` — no OOM or timeout anywhere, including the largest config (res=128, depth=8, hidden=256, batch=64, linear attention: 5.22 GB peak on the 8 GB card). The spec anticipated many OOMs; none occurred, meaning there is real headroom for larger model configs in future sizing decisions.

**B4 — Inference / sampling cost.** Sequential, 12 runs (4 step counts × 3 batch sizes), using the fastest B3 config that cleared the 4 GB budget at batch=32/64×64 (depth=4, hidden=128, quadratic attention). All 12 succeeded; peak VRAM stayed under 32 MB throughout. At batch=1, total sampling time is dominated by fixed per-step launch overhead rather than compute, given how small this config is.

**B5 — Data format probe.** Sequential. Used a real 64×64 von Mises stress field pulled from a live B2 solve (not synthetic), paired with an SDF, geometry mask, and a 3-element parameter vector (θ included as an unused placeholder — not a physical DOF in this session's B2 setup). Compared `.npz`/`.pt`/`.h5` for one record and a batch of 100. `.npz` was smallest on disk; `.pt` loaded fastest. All generated data files were deleted immediately after measurement — verified empty `benchmarks/scratch/` post-run (aside from the archived scripts moved there in Task 7).

## 3. Failures / unusable results

None of the 5 benchmarks failed outright. The one substantive quality issue is B2's Kirsch sanity-check miss (§2, B2) — reported honestly with supporting diagnostic work, not treated as a benchmark failure since B2's actual purpose this session was timing/memory, not physics correctness (explicitly out of scope per the session brief).

## 4. Sub-agent dispatch: as-specified vs. actual, and deviations

- **B1**: sequential, no sub-agents — as specified.
- **B2**: one agent to author+validate the solver script, then 4 sub-agents (one per config: 64/stress, 64/strain, 128/stress, 128/strain) run in parallel — matches the spec's suggested dispatch exactly. No GPU involved, so parallel WSL2/CPU execution was safe; some run-to-run timing noise (elevated `assembly_time_s`/`mesh_time_s` on a few individual runs) is attributed to CPU contention between the 4 concurrent agents, an accepted trade-off of this dispatch pattern.
- **B3**: **deviation from the initial dispatch, corrected mid-session.** The spec calls for "4 sub-agents split by (resolution × attention) pair," and the do-not list separately warns "do NOT run more than 5 sub-agents in parallel... VRAM contention will invalidate B3 timings." These two instructions are in tension for a single-GPU machine: dispatching all 4 GPU-bound sub-agents to run *concurrently* (which is what happened on the first attempt) would have caused exactly the VRAM/timing contention the do-not warns against, even though 4 is under the stated ceiling of 5. This was caught quickly: the 4 concurrent B3 agents were stopped, and investigation found one had left a detached background PowerShell job running independently of its parent agent — that orphaned job had spawned a *duplicate* process invoking the **global system Python** (`C:\Users\Arpit Mathur\AppData\Local\Programs\Python\Python311\python.exe`) rather than the project `.venv`, a workspace-containment violation. Both the orphaned job and the stray global-Python process were killed and confirmed clean via `nvidia-smi` and process listing before any of that run's data was used. B3 was then re-dispatched as 4 sequential (not concurrent) sub-agents, each instructed to run its 16 configs strictly one-at-a-time in the foreground via the exact `.venv\Scripts\python.exe` path, with a 120s per-run timeout as a safety net against Windows' CUDA sysmem-fallback hang behavior (discovered during B3 script validation — near the ~8GB VRAM ceiling, an over-limit allocation can hang for minutes instead of throwing a clean OOM). This corrected sequence is what produced the final B3 results in `notes/benchmark_results.md`; no data from the aborted concurrent attempt was used.
- **B4**: sequential, no sub-agents — as specified (12 short runs).
- **B5**: sequential, no sub-agents — as specified.

## 5. Total wall-clock time

Approximately 2.5 hours from preflight start to Task 7 cleanup, including the Task 0 package-approval pause and the B3 dispatch correction (stop, cleanup, and full sequential re-run of all 4 B3 sub-agents).

## 6. Ready for Session Four

Yes, with one flag for the strategy chat: B2's Kirsch sanity check exceeded the 15% acceptance bar (50.71% relative error) at r/L=0.20, and the supporting diagnostic work (mesh convergence, net-section estimate, small-r/L control) suggests the infinite-plate reference value itself may not be the right target at this hole-to-plate ratio, rather than a solver defect. Session Four's physics validation should account for this before treating 3σ_∞ as ground truth at r/L=0.20. No other blockers — hardware sizing numbers (FEniCSx per-sample cost, DiT training/inference throughput and VRAM, storage format trade-offs) are in `notes/benchmark_results.md` and ready to inform dataset size, model architecture, and training duration decisions.

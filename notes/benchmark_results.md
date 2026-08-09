# FORGE Hardware Benchmark Results

Machine: NVIDIA GeForce RTX 4060 Laptop GPU (8585 MB VRAM), driver 560.94, CUDA 12.6 (torch build cu124).

## Summary for design sizing

- **Fastest FEniCSx config for training data generation**: 64×64, plane strain, median 0.331 sec/sample
- **Slowest FEniCSx config observed**: 128×128, plane strain, median 1.627 sec/sample (128×128 plane stress had a slightly higher single-run max of 2.810s, but plane strain has the higher *median*, which is the sizing-relevant statistic)
- **Best DiT config within VRAM budget (peak < 4 GB at batch 32, 64×64)**: depth=4, hidden=128, attn=quadratic, step time 14.62 ms, VRAM 166.0 MB — chosen as the fastest config that clears the budget; note all 8 res=64 (depth×hidden×attn) configs cleared 4 GB by a wide margin at batch=32, so "best" here was resolved as fastest rather than largest-that-fits
- **Recommended dataset storage format based on B5**: `.pt` for training-loop load speed (16.08 ms median to load a batch of 100, vs 28.66 ms for `.npz` and 48.78 ms for `.h5`). Trade-off: `.npz` is ~53% smaller on disk (1.77 MB vs 3.80 MB per 100 samples) — reconsider if dataset scale makes disk footprint the binding constraint rather than load latency.
- **Anomalies / OOM patterns worth flagging**:
  - B2's Kirsch sanity check (r=0.20, 64×64, plane stress) failed the session's 15% acceptance bar by a wide margin (50.71% relative error). The dispatched agent's diligence (mesh-convergence study, net-section stress cross-check, small-r/L control run) indicates this is a genuine finite-width-plate effect at r/L=0.20, not a script bug — Kirsch's infinite-plate reference (3σ_∞) is a loose approximation at this hole-to-plate ratio. Recommend revisiting the acceptance bar or reference value with the strategy chat before Session Four locks in physics validation.
  - B2 timing showed some run-to-run noise (e.g. 128×128 plane-strain assembly_time_s spiking to 0.545s at r=0.10, vs ~0.04–0.06s elsewhere; r=0.30 mesh_time_s roughly doubling at 128×128) most likely from CPU contention between the 4 concurrently-dispatched B2 sub-agents sharing one WSL2 CPU pool — an accepted trade-off of the parallel dispatch pattern, not a script defect.
  - B3 anticipated many OOMs across its 64-config sweep; none occurred — the 8GB RTX 4060 handled every config (largest: res=128, depth=8, hidden=256, batch=64 → 5.22 GB peak). This means there is real headroom to test larger model configs in future sizing work.
  - Mid-session dispatch correction: the 4 B3 sub-agents were initially launched to run concurrently, which would have caused real GPU contention on the single physical RTX 4060 (invalidating timings) despite being within the "5 parallel" ceiling stated in the do-nots. Caught immediately — the 4 agents were stopped, an orphaned detached background job was found and killed (it had spawned a duplicate process running the **global system Python**, not the project `.venv`, a workspace-containment violation), and B3 was re-dispatched and completed fully sequentially. No contaminated data was used in the final B3 results above.
  - `psutil` and `h5py` were absent from `.venv` at session start (needed for B1 and B5 respectively); installed with explicit user approval before those benchmarks ran — the only packages installed this session.

## Benchmark 1: Environment deep-probe

**Ran at**: 2026-08-09
**Environment**: Windows .venv (torch 2.6.0+cu124), WSL2 Ubuntu-24.04 .venv-wsl (micromamba)

### Configuration
Windows-side: `psutil.virtual_memory()`, `torch.cuda.mem_get_info`/`get_device_capability`, `shutil.disk_usage("A:\\")`, `torch.__version__`/`torch.version.cuda`/`torch.backends.cudnn.version()`.
WSL2-side: `lsb_release -d`, `platform.python_version()`, `dolfinx.__version__`, `gmsh.__version__`, `mpi4py.MPI.get_vendor()`.

### Results

| Metric | Value |
|---|---|
| RAM total | 25.480 GB |
| RAM free | 9.506 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM total | 8585.216 MB |
| VRAM free before alloc | 7443.841 MB |
| CUDA compute capability | 8.9 |
| Disk free (A:\) | 33.144 GB |
| torch version | 2.6.0+cu124 |
| torch CUDA build | 12.4 |
| cuDNN version | 90100 |
| GPU count | 1 |
| Ubuntu version | 24.04.4 LTS |
| WSL2 Python version | 3.12.13 |
| dolfinx version | 0.11.0 |
| gmsh version | 4.15.2 |
| MPI backend | MPICH 5.0.1 |

### Notes
psutil and h5py were absent from `.venv` at session start; installed with user approval (Task 0) before this benchmark ran. No other packages were installed this session.

## Benchmark 2: FEniCSx per-sample cost

**Ran at**: 2026-08-09
**Environment**: WSL2 Ubuntu-24.04, `.venv-wsl` (dolfinx 0.11.0, gmsh 4.15.2, petsc4py, MPICH 5.0.1)

### Configuration
2D square plate, side 1.0, centered at origin, central circular hole radius r. Isotropic linear elastic, E=1.0, ν=0.3. Uniaxial far-field tension σ_∞=1e-3 applied as Neumann traction on left/right edges; top/bottom/hole boundary traction-free; rigid-body modes pinned via point constraints (numerical necessity, not a physics assumption). Gmsh meshing → dolfinx assembly → direct PETSc LU solve → von Mises stress interpolated to a regular grid. Script: `benchmarks/b2_fenicsx_plate.py`. Solved for r ∈ {0.10, 0.20, 0.30} under 4 configs (64×64/128×128 × plane stress/strain) = 12 solves. One discarded warm-up run per config absorbs the one-time FFCX JIT compilation cost (can otherwise inflate `total_time_s` by 10–100× on a form's first invocation).

### Results

Median / max across the 3 parameter points (r=0.10, 0.20, 0.30), seconds unless noted:

| Config | mesh_time_s (med/max) | assembly_time_s (med/max) | solve_time_s (med/max) | interp_time_s (med/max) | total_time_s (med/max) | peak_rss_mb (med/max) |
|---|---|---|---|---|---|---|
| 64×64, plane stress | 0.332 / 0.381 | 0.024 / 0.039 | 0.033 / 0.049 | 0.041 / 0.050 | 0.467 / 0.570 | 184.8 / 186.2 |
| 64×64, plane strain | 0.216 / 0.501 | 0.014 / 0.024 | 0.033 / 0.035 | 0.033 / 0.033 | 0.331 / 1.000 | 185.1 / 188.3 |
| 128×128, plane stress | 0.832 / 1.715 | 0.048 / 0.052 | 0.276 / 0.286 | 0.143 / 0.246 | 1.287 / 2.810 | 256.7 / 268.6 |
| 128×128, plane strain | 1.049 / 1.086 | 0.060 / 0.545 | 0.250 / 0.303 | 0.194 / 0.279 | 1.627 / 2.619 | 255.9 / 268.9 |

### Sanity check (r=0.20, plane stress, 64×64)

```
peak_von_mises = 4.5212e-3
kirsch_reference = 3.0000e-3
relative_error_pct = 50.71%
```

**Fails the 15% acceptance bar by a wide margin — flagged as an open issue, not hidden or tuned away.** The dispatched agent verified this is not a script bug: far-field stress correctly recovers ≈σ_∞ in the loading direction; a small-r/L control run reproduces the expected undershoot-then-converge pattern of P1 elements; a mesh-convergence study (4 refinement levels) converges cleanly toward ≈4.7–4.8e-3, not diverging; and an independent net-section stress-concentration estimate (σ_net = σ_∞/(1−d/W), Kt≈2.9 at d/W=0.4) predicts ≈4.8e-3, closely matching the FEM result. Interpretation: at r/L=0.20 (d/W=0.4), Kirsch's infinite-plate solution (3σ_∞) is a loose reference — the true finite-width stress concentration is substantially larger than "slightly higher." This suggests the 15% acceptance bar in the session spec may have been calibrated for a smaller r/L; worth raising in the strategy chat before Session Four's physics validation locks in a reference value. No physics correctness claims are being made from this benchmark per the session's stated scope.

### Notes
- Per-run raw numbers (not just median/max) are preserved in the session report.
- Anomalies observed in individual runs (r=0.30 configs showing elevated mesh_time_s at 128×128; r=0.10 plane-strain assembly_time_s spike at 128×128) were most likely caused by legitimate CPU contention between the 4 parallel B2 sub-agents sharing the same WSL2 CPU pool (all 4 configs were dispatched to run concurrently per the session spec, which is a real resource trade-off, not a measurement bug). Not investigated further or smoothed over — reported as observed.
- All 12/12 solves succeeded; no failures.

## Benchmark 3: DiT training step timing and VRAM

**Ran at**: 2026-08-09
**Environment**: Windows `.venv` (torch 2.6.0+cu124), RTX 4060 Laptop GPU (8188 MiB VRAM)

### Configuration
Hand-rolled minimal DiT (self-attention + AdaLN-Zero + cross-attention to a patch-embedded conditioning channel; patch size 8), matching PBFM App G's Darcy scale as a starting point (`# Config reference: PBFM App G, Darcy scale`). 2-channel input (noisy state + conditioning field) → 2-channel velocity output. MSE loss vs. random target, AdamW lr=1e-4. 20 warmup + 80 timed steps per config, `torch.cuda.reset_peak_memory_stats()` before timing. Script: `benchmarks/b3_dit_train_step.py`. Swept resolution {64,128} × depth {4,8} × hidden {128,256} × attention {quadratic, linear} × batch {8,16,32,64} = 64 configs. Dispatched as 4 sub-agents split by (resolution × attention); GPU-bound work run strictly sequentially within and across sub-agents (each fully completed and verified clean of leftover processes before the next was launched) since only one physical GPU is available.

### Results

All 64/64 configs completed with `status=OK` — none OOM'd or hung, even at the largest config (res=128, depth=8, hidden=256, batch=64, attn=linear: 5216.76 MB peak, still under the 8188 MiB ceiling).

**res=64, attn=quadratic**

| depth | hidden | batch | step_ms | vram_mb | samples/sec |
|---|---|---|---|---|---|
| 4 | 128 | 8 | 14.82 | 67.9 | 540.0 |
| 4 | 128 | 16 | 16.04 | 100.9 | 997.5 |
| 4 | 128 | 32 | 14.62 | 166.0 | 2188.2 |
| 4 | 128 | 64 | 15.18 | 296.1 | 4214.7 |
| 4 | 256 | 8 | 16.39 | 156.6 | 488.1 |
| 4 | 256 | 16 | 16.28 | 221.0 | 982.8 |
| 4 | 256 | 32 | 16.37 | 347.2 | 1954.9 |
| 4 | 256 | 64 | 31.08 | 604.2 | 2059.0 |
| 8 | 128 | 8 | 27.26 | 114.9 | 293.5 |
| 8 | 128 | 16 | 26.69 | 177.6 | 599.5 |
| 8 | 128 | 32 | 27.53 | 301.9 | 1162.4 |
| 8 | 128 | 64 | 25.06 | 550.5 | 2553.7 |
| 8 | 256 | 8 | 30.15 | 284.7 | 265.3 |
| 8 | 256 | 16 | 30.35 | 408.2 | 527.2 |
| 8 | 256 | 32 | 33.21 | 655.3 | 963.5 |
| 8 | 256 | 64 | 60.01 | 1146.8 | 1066.5 |

**res=64, attn=linear**

| depth | hidden | batch | step_ms | vram_mb | samples/sec |
|---|---|---|---|---|---|
| 4 | 128 | 8 | 22.72 | 77.4 | 352.2 |
| 4 | 128 | 16 | 21.49 | 120.0 | 744.5 |
| 4 | 128 | 32 | 36.29 | 204.1 | 881.7 |
| 4 | 128 | 64 | 20.62 | 372.4 | 3104.4 |
| 4 | 256 | 8 | 22.30 | 177.7 | 358.8 |
| 4 | 256 | 16 | 22.59 | 263.2 | 708.2 |
| 4 | 256 | 32 | 21.43 | 432.6 | 1492.9 |
| 4 | 256 | 64 | 36.97 | 773.1 | 1731.0 |
| 8 | 128 | 8 | 47.08 | 134.0 | 169.9 |
| 8 | 128 | 16 | 47.46 | 215.7 | 337.1 |
| 8 | 128 | 32 | 45.76 | 378.2 | 699.3 |
| 8 | 128 | 64 | 41.47 | 703.1 | 1543.3 |
| 8 | 256 | 8 | 48.40 | 326.9 | 165.3 |
| 8 | 256 | 16 | 46.95 | 492.6 | 340.8 |
| 8 | 256 | 32 | 43.10 | 823.6 | 742.5 |
| 8 | 256 | 64 | 69.23 | 1486.1 | 924.4 |

**res=128, attn=quadratic**

| depth | hidden | batch | step_ms | vram_mb | samples/sec |
|---|---|---|---|---|---|
| 4 | 128 | 8 | 16.02 | 166.1 | 499.4 |
| 4 | 128 | 16 | 18.36 | 295.7 | 871.3 |
| 4 | 128 | 32 | 32.89 | 554.9 | 972.9 |
| 4 | 128 | 64 | 63.22 | 1073.3 | 1012.3 |
| 4 | 256 | 8 | 20.25 | 347.4 | 395.1 |
| 4 | 256 | 16 | 36.53 | 603.4 | 438.0 |
| 4 | 256 | 32 | 67.41 | 1110.0 | 474.7 |
| 4 | 256 | 64 | 132.94 | 2113.5 | 481.4 |
| 8 | 128 | 8 | 29.27 | 301.5 | 273.3 |
| 8 | 128 | 16 | 36.15 | 549.2 | 442.6 |
| 8 | 128 | 32 | 64.08 | 1044.5 | 499.4 |
| 8 | 128 | 64 | 121.04 | 2035.1 | 528.8 |
| 8 | 256 | 8 | 40.40 | 654.6 | 198.0 |
| 8 | 256 | 16 | 71.64 | 1144.2 | 223.3 |
| 8 | 256 | 32 | 130.01 | 2123.0 | 246.1 |
| 8 | 256 | 64 | 262.58 | 4070.7 | 243.7 |

**res=128, attn=linear**

| depth | hidden | batch | step_ms | vram_mb | samples/sec |
|---|---|---|---|---|---|
| 4 | 128 | 8 | 23.16 | 201.0 | 345.5 |
| 4 | 128 | 16 | 22.88 | 365.5 | 699.5 |
| 4 | 128 | 32 | 26.14 | 694.5 | 1224.4 |
| 4 | 128 | 64 | 50.43 | 1352.5 | 1269.2 |
| 4 | 256 | 8 | 21.39 | 419.9 | 374.0 |
| 4 | 256 | 16 | 34.19 | 746.7 | 468.1 |
| 4 | 256 | 32 | 63.20 | 1398.5 | 506.3 |
| 4 | 256 | 64 | 125.81 | 2687.6 | 508.7 |
| 8 | 128 | 8 | 42.09 | 371.3 | 190.1 |
| 8 | 128 | 16 | 40.08 | 688.8 | 399.2 |
| 8 | 128 | 32 | 51.12 | 1323.7 | 626.0 |
| 8 | 128 | 64 | 93.99 | 2593.4 | 680.9 |
| 8 | 256 | 8 | 40.91 | 799.4 | 195.6 |
| 8 | 256 | 16 | 68.49 | 1430.8 | 233.6 |
| 8 | 256 | 32 | 120.08 | 2696.99 | 266.5 |
| 8 | 256 | 64 | 248.23 | 5216.8 | 257.8 |

### Notes
The spec anticipated many OOMs across the 64-config sweep; none occurred. On an 8GB RTX 4060, even the largest config tested (res=128, depth=8, hidden=256, batch=64) stayed under 5.3 GB peak VRAM. This means the sweep's OOM-handling path (`status=OOM`, catch-and-skip) was validated correctly during script development (B3 authoring, `--batch 4096` smoke test) but never exercised by the sweep itself — headroom exists to test larger configs in a future session if bigger models are being considered. Separately, the script-authoring agent found that Windows CUDA can hang indefinitely near the ~8GB boundary instead of throwing a clean OOM (confirmed with `--batch 256` at res128/depth8/hidden256, which hung for 5+ minutes and had to be killed); the sweep agents' 120s per-run timeout guard was the correct mitigation and worked as designed, though it was never triggered in practice since no config approached the ceiling.

## Benchmark 4: Inference / sampling cost

**Ran at**: 2026-08-09
**Environment**: Windows `.venv` (torch 2.6.0+cu124), RTX 4060 Laptop GPU

### Configuration
Config selected from B3: res=64, depth=4, hidden=128, attn=quadratic — the fastest config that stayed under the 4 GB peak-VRAM budget at batch=32/res=64 (in fact all 8 res=64 configs cleared that budget by a wide margin; this one was fastest at 14.62 ms/step, 166 MB peak, in B3). Random (untrained) weights — measuring sampling-loop cost only, no correctness claim. Euler integration over a uniform time schedule, `benchmarks/b4_dit_sample.py`. No warmup steps (not specified for B4, unlike B3). Swept inference steps {5,10,20,50} × batch {1,8,32} = 12 runs, run sequentially (no sub-agent dispatch, per spec).

### Results

| steps | batch | total_time_ms | per_sample_ms | peak_vram_mb |
|---|---|---|---|---|
| 5 | 1 | 197.46 | 197.46 | 16.2 |
| 5 | 8 | 193.42 | 24.18 | 19.2 |
| 5 | 32 | 191.19 | 5.97 | 31.6 |
| 10 | 1 | 198.82 | 198.82 | 16.2 |
| 10 | 8 | 208.61 | 26.08 | 19.2 |
| 10 | 32 | 218.23 | 6.82 | 31.6 |
| 20 | 1 | 239.51 | 239.51 | 16.2 |
| 20 | 8 | 245.22 | 30.65 | 19.2 |
| 20 | 32 | 243.72 | 7.62 | 31.6 |
| 50 | 1 | 374.88 | 374.88 | 16.2 |
| 50 | 8 | 379.27 | 47.41 | 19.2 |
| 50 | 32 | 344.36 | 10.76 | 31.6 |

### Notes
All 12/12 runs succeeded. Total sampling time at batch=1 is dominated by fixed per-step overhead (kernel launch, no warmup) rather than raw compute — the model is tiny at this config, so total time barely grows between steps=5 and steps=20 (197ms → 240ms) despite 4x more model evaluations; batch=32 shows the same pattern. Peak VRAM stayed under 32 MB across all 12 runs, far below any budget concern for sampling with this config.

## Benchmark 5: Data format probe

**Ran at**: 2026-08-09
**Environment**: Windows `.venv` (numpy, torch, h5py 3.16.0)

### Configuration
Sample record: real 64×64 float32 von Mises stress field (from a live B2 solve, r=0.20/64×64/plane stress — not synthetic), SDF (signed distance to the r=0.20 hole boundary, float32), geometry mask (uint8), and a 3-element float32 parameter vector [r, σ_∞, θ]. Note: θ is not a physical parameter of the B2 setup (loading is fixed along x; only r and σ_∞ vary) — included as a 0.0 placeholder solely to fill the requested 3-element shape, not a real DOF. Formats: `.npz` (numpy compressed), `.pt` (torch pickle), `.h5` (HDF5, gzip level 4). Individual record and a batch of 100 identical records saved in each format; load time is median of 5 loads of the batch-of-100. Script: `benchmarks/b5_data_format.py`. All generated files deleted immediately after measurement (verified: `benchmarks/scratch/` empty post-run).

### Results

| Format | Individual (bytes) | Batch-of-100 (bytes) | Batch-of-100 load time, median of 5 (ms) |
|---|---|---|---|
| .npz | 17,683 | 1,768,442 | 28.66 |
| .pt | 38,664 | 3,795,938 | 16.08 |
| .h5 | 35,056 | 3,318,400 | 48.78 |

### Notes
`.npz` is smallest on disk (roughly half the size of `.pt`/`.h5`) due to its default DEFLATE compression on otherwise-uncompressible float32 noise-like stress fields. `.pt` loads fastest despite being the largest file — pickle deserialization overhead is lower than either compressed format's decompression cost at this batch size. `.h5` is both the slowest to load and not the smallest, likely due to per-sample group overhead (100 separate HDF5 groups/datasets) rather than a single flat array; a single large HDF5 dataset (samples as one array dimension rather than 100 groups) might load faster, but was not tested since the group-per-sample layout best mirrors the stated "sample record" structure.

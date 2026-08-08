# CLAUDE.md — FORGE workspace

## Project

FORGE = Flow-matching with cOnformal Rigour for Generative Elasticity. Research project targeting NeurIPS Sim2Science 2026 Tiny Paper (deadline 29 Aug 2026 AoE). Deliverables: a GitHub repo and a 2-page paper. Hybrid Windows + WSL2 setup: Windows for most work, WSL2 for FEniCSx-based finite-element data generation.

## Workspace containment — NON-NEGOTIABLE

Working directory: `A:\Projects_new\forge` (Windows) or `/mnt/a/Projects_new/forge` (WSL2).

- Every file operation stays within this directory.
- No writes to other drives, user home, system paths, or WSL2 root filesystem.
- No global `pip install`, `npm install -g`, or system package installs.
- Python packages install to `.venv` (Windows-side) or `.venv-wsl` (WSL2-side), both project-local.
- If a task appears to require touching anything outside the workspace, STOP and ask.

## Workflow separation

Physics decisions, project design, dataset sizing, model architecture, and paper narrative are decided by the user in a separate strategy chat. You (Claude Code agent) implement the physics and code exactly as specified in each prompt. Do not:

- Invent parameter ranges, material constants, boundary conditions, or physical assumptions.
- Change the model architecture, dataset size, resolution, or evaluation metrics from what the prompt specifies.
- Add features, config knobs, or abstractions the prompt does not request.
- Reach for a library when stdlib works.

If a physical assumption or numerical parameter is unstated in the prompt, STOP and ask. Do not fill in defaults.

## Verification-first

Every task ends with a checkable pass/fail signal. If you cannot verify, the task is under-specified — ask, do not proceed. Report the exact verification output (numbers, error text) in your reply, not a paraphrase or summary.

## No invention

Never fabricate numbers, results, or file contents. If FEniCS fails, report the failure. If a computation errors out, report the error text. If a benchmark returns zero samples, report zero. Hallucinated results are the worst possible failure mode in this project.

## Task discipline

- After 2 failed attempts on the same infrastructure issue, STOP. Report the exact error and 2-3 concrete options. Do not keep trying variants.
- Read verification output as ground truth. Do not assume something worked because it should have worked.

## Path handling

- Windows commands: `A:\Projects_new\forge\...`
- WSL2 commands: `/mnt/a/Projects_new/forge/...`
- Be explicit which side each command runs on. When in doubt, state it in your reply.
- Never mix path styles in one command.

## File hygiene

- Never read, log, or echo values from `.env`, `.env.local`, `.mcp.json`, or any file with `secret` or `credential` in its name.
- Never commit secrets. `.env` is gitignored — verify before staging.
- `CLAUDE.local.md` is gitignored — use it for personal or machine-specific overrides if needed.

## Paper prose

- No em dashes in paper text.
- Professional academic language throughout. No informal descriptors, no editorial characterization of results.
- Consistent author identity: `retroranger24@gmail.com`.

## Attribution

When implementing something derived from a reference paper, add a one-line comment attributing the reference (example: `# Config from PBFM App G, Darcy scale`).

## Reference papers

- PBFM: Baldan et al. 2026, "Physics vs Distributions: Pareto Optimal Flow Matching with Physics Constraints"
- Gen-TO / BFM: Rashed et al. 2026, "Sensitivity-Conditioned Bernoulli Flow Matching"
- SAR: Lino & Thuerey 2026, "Scale-Autoregressive Modeling for Physical Fields"
- DiffuMeta: Zheng, Kumar, Kochmann 2026, arXiv:2507.15753
- MNN: Schuttert et al., JCP 2026, "Constitutive Manifold Neural Networks"
- CAISc: Mathur et al. 2026, self-authored prior work; cite as third-party after arXiv upload

## Repository etiquette

- Never commit without user approval in the prompt.
- Never push to `origin` without user approval.
- Branch naming: `feat/<short>`, `fix/<short>`, `bench/<short>`, `paper/<short>`.

## Standard verification commands

- Python package installed: `python -c "import forge; print(forge.__version__)"`
- FEniCSx (WSL2): `wsl bash -c "source /mnt/a/Projects_new/forge/.venv-wsl/bin/activate && python -c 'import dolfinx; print(dolfinx.__version__)'"`
- CUDA / PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Repo state: `git status`

## Never touch

- Anything outside `A:\Projects_new\forge` / `/mnt/a/Projects_new/forge`
- `.env`, `.env.local`, `.mcp.json`
- `data/` and `outputs/` unless the prompt explicitly names files under them
- Global Python, global Node, or WSL2 base environment

"""Split-wise generation and conformal coverage evaluation.

Generation is the expensive half and does not depend on q_hat, so a split is
sampled once into per-pixel nonconformity scores and every nominal level is
then read off those scores.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forge.eval.conformal import (
    _masked_scores,
    compute_nonconformity_scores,
    empirical_coverage,
)
from forge.eval.sampling import sample_conditional
from forge.logging import emit


def generate_split_scores(
    model: nn.Module,
    data_loader: DataLoader,
    num_samples_per_condition: int = 500,
    num_fm_steps: int = 20,
    batch_size: int = 32,
    log_path: Path | None = None,
    run_id: str = "run_01",
    worker_id: str = "coverage_eval",
    split_name: str = "unknown",
    progress_every: int = 100,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample every condition in the split and score it.

    Returns (scores, masks), both (N, 1, H, W) on CPU. Masked-out pixels are
    NaN in scores. Each loader batch gets its own sampler seed derived from
    `seed` so the run is reproducible but batches do not share noise.
    """
    scores, masks = [], []
    done = 0
    next_mark = progress_every
    started = time.time()

    for batch_idx, batch in enumerate(data_loader):
        gen = sample_conditional(
            model,
            batch["sdf"],
            batch["scalars"],
            num_fm_steps=num_fm_steps,
            num_samples_per_condition=num_samples_per_condition,
            batch_size=batch_size,
            seed=seed + batch_idx,
        )
        device = gen.device
        s = compute_nonconformity_scores(gen, batch["x"].to(device), batch["mask"].to(device))
        scores.append(s.cpu())
        masks.append(batch["mask"].cpu())
        del gen, s

        done += batch["x"].shape[0]
        if log_path is not None and done >= next_mark:
            emit(log_path, worker_id, run_id, split_name, "progress",
                 done=done, elapsed_sec=int(time.time() - started))
            next_mark = ((done // progress_every) + 1) * progress_every

    if not scores:
        # 0.0 coverage from an empty split would read as a measured result
        raise ValueError(f"split {split_name!r} produced no batches")
    return torch.cat(scores), torch.cat(masks)


def evaluate_split_coverage(
    model: nn.Module,
    data_loader: DataLoader,
    q_hats: dict[float, float],
    num_samples_per_condition: int = 500,
    num_fm_steps: int = 20,
    batch_size: int = 32,
    log_path: Path | None = None,
    run_id: str = "run_01",
    worker_id: str = "coverage_eval",
    split_name: str = "unknown",
    progress_every: int = 100,
    seed: int = 0,
) -> dict:
    """Run generation and compute coverage on a full split.

    `q_hats` maps nominal level (0.80, 0.90, 0.95) to its calibrated threshold;
    one scalar cannot produce three coverages, so the spec's single `q_hat` is
    taken per level here.

    Interval width is 2 * q_hat: the score is a symmetric absolute residual
    about the sample mean, so the interval half-width is q_hat at every pixel
    and the mean width is the same constant on every split. It is reported for
    completeness only; `mean_absolute_residual` is the per-split quantity that
    actually varies.

    Coverage here is pooled empirical per-pixel coverage. No finite-sample
    split-conformal guarantee is claimed: see the caveat in forge.eval.conformal.
    """
    for level in (0.80, 0.90, 0.95):
        if level not in q_hats:
            raise ValueError(f"q_hats missing nominal level {level}: {sorted(q_hats)}")
        if not q_hats[level] > 0:
            raise ValueError(f"q_hat at {level} must be > 0, got {q_hats[level]}")

    started = time.time()
    if log_path is not None:
        emit(log_path, worker_id, run_id, split_name, "progress",
             n_conditions=len(data_loader.dataset), M=num_samples_per_condition,
             fm_steps=num_fm_steps)

    scores, masks = generate_split_scores(
        model, data_loader,
        num_samples_per_condition=num_samples_per_condition,
        num_fm_steps=num_fm_steps, batch_size=batch_size, log_path=log_path,
        run_id=run_id, worker_id=worker_id, split_name=split_name,
        progress_every=progress_every, seed=seed,
    )

    flat = _masked_scores(scores, masks)
    out = {
        "split_name": split_name,
        "n_samples": int(scores.shape[0]),
        "coverage_definition": "pooled empirical per-pixel coverage; "
                               "no finite-sample split-conformal guarantee claimed",
        # varies per split, unlike the interval width, so this is the per-split
        # magnitude signal to read alongside coverage
        "mean_absolute_residual": flat.mean().item(),
        "median_absolute_residual": flat.median().item(),
    }
    for level in (0.80, 0.90, 0.95):
        tag = f"{int(level * 100)}"
        cov = empirical_coverage(scores, masks, q_hats[level])
        assert 0.0 <= cov <= 1.0, f"coverage out of [0, 1] at {level}: {cov}"
        out[f"coverage_at_{tag}"] = cov
        out[f"mean_interval_width_at_{tag}"] = 2.0 * q_hats[level]

    if log_path is not None:
        emit(log_path, worker_id, run_id, split_name, "complete",
             n_samples=out["n_samples"],
             mean_abs_resid=f"{out['mean_absolute_residual']:.6f}",
             cov80=f"{out['coverage_at_80']:.4f}",
             cov90=f"{out['coverage_at_90']:.4f}",
             cov95=f"{out['coverage_at_95']:.4f}",
             elapsed_sec=int(time.time() - started))
    return out

"""Split conformal prediction over per-pixel Von Mises stress.

Reference: Angelopoulos & Bates 2023, "A Gentle Introduction to Conformal
Prediction", split conformal formulation.

The scoring unit is a masked-in pixel: scores are pooled over all pixels of all
calibration samples, and q_hat is the k-th smallest with
k = ceil((n + 1) (1 - alpha)). The resulting interval at a pixel is
[sample_mean - q_hat, sample_mean + q_hat].

Caveat on the guarantee. Split conformal's finite-sample bound needs
exchangeable scores. Pixels within one stress field are not exchangeable: they
are spatially correlated, and a fixed pixel position carries its own residual
distribution (the stress concentration at the hole boundary is systematically
harder than the far field). What the functions below report is therefore
pooled empirical pixel coverage, not a certified pixel-level guarantee. A
field-level guarantee would need one score per field, e.g. the maximum
masked-in residual. The pooled quantity is the one this project measures.

Masked-out pixels (mask == 0, inside the hole) are scored as NaN rather than 0.
A 0 there would read as a perfect residual and silently inflate coverage; NaN
cannot be mistaken for a good score and is dropped explicitly at every use.
"""
from __future__ import annotations

import math

import torch


def _masked_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Flat 1D vector of the masked-in scores.

    Selection is by mask alone. Dropping non-finite scores here instead would
    make a field that generated NaN vanish from calibration and from coverage
    without a word, which is the failure mode this project cannot afford.
    """
    if scores.shape != mask.shape:
        raise ValueError(
            f"scores {tuple(scores.shape)} and mask {tuple(mask.shape)} must have equal shape"
        )
    flat = scores[mask != 0]
    n_bad = int((~torch.isfinite(flat)).sum())
    if n_bad:
        raise ValueError(f"{n_bad} non-finite score(s) at masked-in pixels")
    return flat


def compute_nonconformity_scores(
    generated_samples: torch.Tensor,   # (N, M, 1, H, W)
    ground_truth: torch.Tensor,        # (N, 1, H, W)
    mask: torch.Tensor,                # (N, 1, H, W), uint8
) -> torch.Tensor:
    """Per-pixel nonconformity: |mean over generated samples - ground_truth|.

    Returns (N, 1, H, W). Masked-out pixels (mask == 0) are set to NaN.
    """
    if generated_samples.ndim != 5:
        raise ValueError(f"generated_samples must be 5D (N, M, 1, H, W), got "
                         f"{tuple(generated_samples.shape)}")
    n = generated_samples.shape[0]
    if ground_truth.shape[0] != n:
        raise ValueError(
            f"generated_samples has N={n} but ground_truth has N={ground_truth.shape[0]}"
        )
    expected = (n,) + tuple(generated_samples.shape[2:])
    if tuple(ground_truth.shape) != expected:
        raise ValueError(f"ground_truth must be {expected}, got {tuple(ground_truth.shape)}")
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask must be {expected}, got {tuple(mask.shape)}")

    scores = (generated_samples.mean(dim=1) - ground_truth).abs()
    return scores.masked_fill(mask == 0, float("nan"))


def calibrate_quantile(
    calibration_scores: torch.Tensor,  # (N_cal, 1, H, W)
    mask: torch.Tensor,                # (N_cal, 1, H, W)
    alpha: float,                      # 1 - nominal_coverage
) -> float:
    """Split conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest of the
    masked-in calibration scores. Returns scalar threshold q_hat.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    flat = _masked_scores(calibration_scores, mask)
    n = flat.numel()
    if n == 0:
        raise ValueError("no finite masked-in calibration scores")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        # n too small for this alpha: the honest threshold is +inf, which is a
        # useless interval, so refuse rather than return the max and pretend.
        raise ValueError(
            f"alpha={alpha} needs k={k} > n={n} calibration scores; no finite q_hat exists"
        )
    q_hat = torch.kthvalue(flat.double().flatten(), k).values.item()
    assert math.isfinite(q_hat), f"non-finite q_hat: {q_hat}"
    return q_hat


def empirical_coverage(
    eval_scores: torch.Tensor,         # (N, 1, H, W)
    mask: torch.Tensor,                # (N, 1, H, W)
    q_hat: float,
) -> float:
    """Fraction of masked-in pixels across N samples where score <= q_hat."""
    flat = _masked_scores(eval_scores, mask)
    n = flat.numel()
    if n == 0:
        raise ValueError("no finite masked-in evaluation scores")
    coverage = (flat <= q_hat).sum().item() / n
    assert 0.0 <= coverage <= 1.0, f"coverage out of [0, 1]: {coverage}"
    return coverage


if __name__ == "__main__":
    torch.manual_seed(0)
    n_cal, m, h, w = 200, 8, 8, 8
    mask = torch.ones(n_cal, 1, h, w, dtype=torch.uint8)
    mask[:, :, :2, :2] = 0  # a hole that must never be scored

    gt = torch.randn(n_cal, 1, h, w)
    gen = gt.unsqueeze(1) + 0.1 * torch.randn(n_cal, m, 1, h, w)
    scores = compute_nonconformity_scores(gen, gt, mask)
    assert scores.shape == (n_cal, 1, h, w), scores.shape
    assert torch.isnan(scores[:, :, :2, :2]).all(), "masked-out pixels must be NaN"
    assert torch.isfinite(scores[:, :, 2:, :]).all()

    q80 = calibrate_quantile(scores, mask, 0.20)
    q90 = calibrate_quantile(scores, mask, 0.10)
    q95 = calibrate_quantile(scores, mask, 0.05)
    assert 0 < q80 < q90 < q95, (q80, q90, q95)

    # coverage on the calibration set itself must land at the nominal level
    for q, nominal in ((q80, 0.80), (q90, 0.90), (q95, 0.95)):
        cov = empirical_coverage(scores, mask, q)
        assert abs(cov - nominal) < 0.01, (cov, nominal)

    # zeroing the hole instead of NaN would have counted 4 free hits per sample
    assert empirical_coverage(scores, mask, q90) < empirical_coverage(
        scores.nan_to_num(0.0), mask.new_ones(mask.shape), q90
    )
    print("conformal self-check ok")

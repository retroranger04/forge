"""Batched conditional sampling from the trained flow-matching model.

Euler integration of the FM velocity field from t=0 (Gaussian noise) to t=1,
matching the conditional OT path in forge.train.flow_matching:

    x_t = (1 - (1 - sigma_min) t) x_0 + t x_1     =>     dx/dt = u_t

so integrating u_t forward from x_0 ~ N(0, I) reaches x_1 + sigma_min * x_0 at
t=1, i.e. a sample of x_1 carrying residual noise of scale sigma_min (1e-3 for
run_01). Training used the same path, so sampling inherits that same floor.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SamplerNumericalError(RuntimeError):
    """Raised when a generated batch contains NaN or inf.

    Carries the offending condition indices so the caller reports which test
    points failed rather than a bare "sampling broke".
    """

    def __init__(self, condition_indices: list[int]):
        self.condition_indices = condition_indices
        super().__init__(
            f"non-finite values in generated samples for condition indices {condition_indices}"
        )


@torch.no_grad()
def sample_conditional(
    model: nn.Module,
    sdf: torch.Tensor,          # (B, 1, H, W), normalized
    scalars: torch.Tensor,      # (B, scalar_cond_dim), normalized
    num_fm_steps: int = 20,
    num_samples_per_condition: int = 500,
    batch_size: int = 32,
    seed: int = 0,
) -> torch.Tensor:
    """Draw num_samples_per_condition samples for each condition in the batch.

    Returns tensor of shape (B, num_samples_per_condition, 1, H, W).

    Uses Euler integration of the FM velocity field from t=0 to t=1.
    Model weights are frozen (torch.no_grad()).
    Seed controls reproducibility across runs on the same conditions: the
    trajectory noise comes from a dedicated generator, so the result is fixed
    given (model_state, sdf, scalars, num_fm_steps, batch_size, seed) and is
    independent of ambient RNG state.
    """
    if sdf.ndim != 4 or sdf.shape[1] != 1:
        raise ValueError(f"sdf must be (B, 1, H, W), got {tuple(sdf.shape)}")
    if scalars.ndim != 2 or scalars.shape[0] != sdf.shape[0]:
        raise ValueError(
            f"scalars must be (B, scalar_cond_dim) matching sdf batch {sdf.shape[0]}, "
            f"got {tuple(scalars.shape)}"
        )
    if num_fm_steps < 1 or num_samples_per_condition < 1 or batch_size < 1:
        raise ValueError("num_fm_steps, num_samples_per_condition and batch_size must be >= 1")

    device = next(model.parameters()).device
    sdf, scalars = sdf.to(device), scalars.to(device)
    b, _, h, w = sdf.shape
    m = num_samples_per_condition
    total = b * m

    # trajectory index -> condition index, so a flat chunk knows its conditions
    cond_of = torch.arange(b, device=device).repeat_interleave(m)
    gen = torch.Generator(device=device).manual_seed(seed)
    dt = 1.0 / num_fm_steps

    was_training = model.training
    model.eval()
    try:
        out = torch.empty(total, 1, h, w, device=device, dtype=sdf.dtype)
        for start in range(0, total, batch_size):
            idx = cond_of[start:start + batch_size]
            n = idx.shape[0]
            x = torch.randn(n, 1, h, w, device=device, dtype=sdf.dtype, generator=gen)
            c_sdf, c_scalars = sdf[idx], scalars[idx]
            for step in range(num_fm_steps):
                t = torch.full((n,), step * dt, device=device, dtype=x.dtype)
                x = x + dt * model(torch.cat([x, c_sdf], dim=1), t, c_scalars)
            out[start:start + n] = x
    finally:
        model.train(was_training)

    out = out.reshape(b, m, 1, h, w)
    assert out.shape == (b, m, 1, h, w), out.shape

    bad = ~torch.isfinite(out)
    if bad.any():
        raise SamplerNumericalError(bad.flatten(1).any(dim=1).nonzero().flatten().tolist())
    return out


if __name__ == "__main__":
    from forge.models.dit import DiT

    m = DiT()
    s = sample_conditional(
        m, torch.randn(3, 1, 64, 64), torch.randn(3, 4),
        num_fm_steps=4, num_samples_per_condition=5, batch_size=7,
    )
    assert s.shape == (3, 5, 1, 64, 64), s.shape
    assert torch.isfinite(s).all()
    # same seed reproduces regardless of ambient RNG; a different seed does not
    c_sdf, c_scal = torch.randn(2, 1, 64, 64), torch.randn(2, 4)
    kw = dict(num_fm_steps=3, num_samples_per_condition=4)
    a = sample_conditional(m, c_sdf, c_scal, seed=1, **kw)
    torch.manual_seed(999)
    assert torch.equal(a, sample_conditional(m, c_sdf, c_scal, seed=1, **kw))
    assert not torch.equal(a, sample_conditional(m, c_sdf, c_scal, seed=2, **kw))
    print("sampling self-check ok")

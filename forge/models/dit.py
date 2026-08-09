"""DiT for 64x64 stress fields.

Reference: PBFM App G, Darcy scale; block and AdaLN-Zero structure from
Peebles & Xie 2023 (DiT). Conditioning is the SDF channel concatenated at the
input plus the four scalars through AdaLN.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# Feedforward expansion: DiT convention (Peebles & Xie 2023) as adopted by
# PBFM App G. The locked spec fixes depth, hidden size and head count but not
# this ratio.
MLP_RATIO = 4.0


def _sincos_1d(dim: int, pos: torch.Tensor) -> torch.Tensor:
    """(M,) positions -> (M, dim) sin/cos bands. Reference: DiT get_1d_sincos."""
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0))
    out = pos.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_pos_embed_2d(dim: int, grid_size: int) -> torch.Tensor:
    """(grid_size**2, dim) 2D sinusoidal position embedding, half per axis."""
    g = torch.arange(grid_size, dtype=torch.float32)
    gh, gw = torch.meshgrid(g, g, indexing="ij")
    return torch.cat([_sincos_1d(dim // 2, gh), _sincos_1d(dim // 2, gw)], dim=1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """(B,) flow-matching time in [0, 1] -> (B, dim) sinusoidal embedding.

    The x1000 scaling puts t on the [0, 1000] timestep convention of DiT
    (Peebles & Xie 2023) as adopted by PBFM App G. Unscaled, at max_period
    10000, every band would sit under one radian across the whole trajectory
    and the embedding would barely vary with t.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = (t.float() * 1000.0).reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat([args.cos(), args.sin()], dim=1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Self-attention + feedforward, both AdaLN-Zero modulated by `cond`."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * MLP_RATIO)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        # zero init: every block starts as the identity, so depth costs nothing
        # at step 0 and the signal path is the residual stream alone
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.adaLN_modulation(cond).chunk(6, dim=1)
        )
        h = modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class DiT(nn.Module):
    """Reference: PBFM App G, adapted for 64x64 stress fields."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 2,   # noisy state + SDF
        out_channels: int = 1,  # velocity field for von_mises only
        depth: int = 8,
        hidden_size: int = 256,
        num_heads: int = 4,
        scalar_cond_dim: int = 4,  # [r, sigma_inf, sin_theta, cos_theta]
        time_embed_dim: int = 256,
    ):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = img_size // patch_size
        self.time_embed_dim = time_embed_dim

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_cond_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.blocks = nn.ModuleList(DiTBlock(hidden_size, num_heads) for _ in range(depth))
        self.norm_out = nn.LayerNorm(hidden_size, eps=1e-6)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        # zero init, as DiT does: predicted velocity starts at exactly zero
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # not a learned parameter; a buffer so it is built once rather than
        # rebuilt on every forward, and moves with .to(device)
        self.register_buffer(
            "pos_embed", sincos_pos_embed_2d(hidden_size, self.grid_size), persistent=False
        )

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) -> (B, C, H, W)."""
        b, g, p, c = x.shape[0], self.grid_size, self.patch_size, self.out_channels
        x = x.reshape(b, g, g, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, g * p, g * p)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x_t).flatten(2).transpose(1, 2) + self.pos_embed
        cond = self.time_mlp(timestep_embedding(t, self.time_embed_dim)) + self.scalar_mlp(scalars)
        for block in self.blocks:
            x = block(x, cond)
        return self.unpatchify(self.proj_out(self.norm_out(x)))


if __name__ == "__main__":
    m = DiT()
    x = torch.randn(2, 2, 64, 64)
    t = torch.rand(2)
    s = torch.randn(2, 4)
    y = m(x, t, s)
    assert y.shape == (2, 1, 64, 64), y.shape
    print(f"DiT params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")

"""Split-scoped dataset over the Session Four .pt records.

One sample file holds von_mises, sdf, mask (64x64) and params [r, sigma_inf,
theta_deg]. theta is handed to the model as sin/cos so the model sees the
circular structure instead of a wrap-around discontinuity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

FIELDS = ("von_mises", "sdf")
_EPS = 1e-8


def _sample_paths(split_dir: Path) -> list[Path]:
    """Sample files in numeric index order (sample_2 before sample_10)."""
    return sorted(split_dir.glob("sample_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def _scalars(params: torch.Tensor) -> torch.Tensor:
    """[r, sigma_inf, theta_deg] -> [r, sigma_inf, sin_theta, cos_theta]."""
    r, sigma_inf, theta_deg = (float(v) for v in params)
    theta = math.radians(theta_deg)
    return torch.tensor([r, sigma_inf, math.sin(theta), math.cos(theta)], dtype=torch.float32)


def compute_train_statistics(split_dir: Path, save_path: Path) -> dict:
    """Per-field mean/std over a whole split, saved as JSON and returned.

    Standardises von_mises, sdf, and each of the four scalars independently.
    The mask is a label, not a magnitude, so it is left alone.
    """
    split_dir, save_path = Path(split_dir), Path(save_path)
    paths = _sample_paths(split_dir)
    if not paths:
        raise FileNotFoundError(f"no sample_*.pt under {split_dir}")

    # float64 accumulators: a float32 running sum over ~4e7 pixels loses
    # low-order bits and skews the variance
    total = {f: 0.0 for f in FIELDS}
    total_sq = {f: 0.0 for f in FIELDS}
    count = {f: 0 for f in FIELDS}
    s_total = torch.zeros(4, dtype=torch.float64)
    s_total_sq = torch.zeros(4, dtype=torch.float64)

    for path in paths:
        rec = torch.load(path, weights_only=True)
        for f in FIELDS:
            v = rec[f].double()
            total[f] += v.sum().item()
            total_sq[f] += (v * v).sum().item()
            count[f] += v.numel()
        s = _scalars(rec["params"]).double()
        s_total += s
        s_total_sq += s * s

    def _mean_std(sum_, sum_sq, n):
        mean = sum_ / n
        # clamp: catastrophic cancellation can push a near-zero variance below 0
        return mean, math.sqrt(max(sum_sq / n - mean * mean, 0.0))

    stats = {"n_samples": len(paths)}
    for f in FIELDS:
        mean, std = _mean_std(total[f], total_sq[f], count[f])
        stats[f] = {"mean": mean, "std": std}
    n = len(paths)
    s_mean = s_total / n
    s_std = torch.clamp(s_total_sq / n - s_mean * s_mean, min=0.0).sqrt()
    stats["scalars"] = {"mean": s_mean.tolist(), "std": s_std.tolist()}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class ForgeSplitDataset(torch.utils.data.Dataset):
    """Loads .pt sample records from a single split directory."""

    def __init__(self, split_dir: Path, normalize: bool = True, stats_path: Path | None = None):
        self.paths = _sample_paths(Path(split_dir))
        if not self.paths:
            raise FileNotFoundError(f"no sample_*.pt under {split_dir}")
        self.normalize = normalize
        if normalize:
            if stats_path is None:
                raise ValueError("normalize=True requires stats_path")
            self.stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
            self._s_mean = torch.tensor(self.stats["scalars"]["mean"], dtype=torch.float32)
            self._s_std = torch.tensor(self.stats["scalars"]["std"], dtype=torch.float32)
        else:
            self.stats = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        rec = torch.load(self.paths[idx], weights_only=True)
        x = rec["von_mises"].unsqueeze(0)
        sdf = rec["sdf"].unsqueeze(0)
        scalars = _scalars(rec["params"])
        if self.normalize:
            x = (x - self.stats["von_mises"]["mean"]) / (self.stats["von_mises"]["std"] + _EPS)
            sdf = (sdf - self.stats["sdf"]["mean"]) / (self.stats["sdf"]["std"] + _EPS)
            scalars = (scalars - self._s_mean) / (self._s_std + _EPS)
        return {
            "x": x,
            "sdf": sdf,
            "mask": rec["mask"].unsqueeze(0),
            "scalars": scalars,
            "meta": {"halton_index": rec["halton_index"], "physics": rec["physics"]},
        }

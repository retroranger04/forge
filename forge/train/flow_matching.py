"""Conditional flow matching with OT paths, and the training loop that uses it.

Reference: Lipman et al. 2023, conditional OT probability paths.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from forge.logging import emit
from forge.models.dit import DiT

WORKER = "trainer"
VAL_SEED = 1234


def _one_line(text: str, limit: int = 300) -> str:
    """Collapse whitespace so an exception message survives emit()'s no-newline
    rule. Without this the error event itself raises and the crash goes unlogged.
    """
    return " ".join(str(text).split())[:limit]


class FlowMatchingLoss:
    """Conditional flow matching with OT paths (Lipman 2023)."""

    def __init__(self, sigma_min: float = 1e-3):
        self.sigma_min = sigma_min

    def __call__(self, model, x1: torch.Tensor, sdf: torch.Tensor, scalars: torch.Tensor):
        t = torch.rand(x1.shape[0], device=x1.device)
        x0 = torch.randn_like(x1)
        tb = t.view(-1, 1, 1, 1)
        x_t = (1 - (1 - self.sigma_min) * tb) * x0 + tb * x1
        u_t = x1 - (1 - self.sigma_min) * x0
        pred = model(torch.cat([x_t, sdf], dim=1), t, scalars)
        loss = F.mse_loss(pred, u_t)
        return loss, {"loss": loss.item()}


def _infinite(loader: DataLoader):
    """Cycle a loader forever, but refuse to spin on an empty one.

    Without the guard an empty loader makes the run consume a core in silence,
    emitting neither an error nor a complete event, so the watchdog only ever
    sees a worker that never appeared.
    """
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise ValueError("train_loader produced no batches")


@torch.no_grad()
def _validate(model, loss_fn: FlowMatchingLoss, val_loader: DataLoader, device) -> float:
    """Mean flow-matching loss over the validation split.

    t and x_0 are drawn from a fixed seed so successive validations differ
    because the model changed, not because the sampled noise did.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(VAL_SEED)
    was_training = model.training
    model.eval()
    try:
        total, n = 0.0, 0
        for batch in val_loader:
            x1 = batch["x"].to(device)
            loss, _ = loss_fn(model, x1, batch["sdf"].to(device), batch["scalars"].to(device))
            total += loss.item() * x1.shape[0]
            n += x1.shape[0]
        if n == 0:
            # 0.0 would read as a perfect score for a split that has no data
            raise ValueError("val_loader produced no batches")
        return total / n
    finally:
        model.train(was_training)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def train_one_run(
    run_id: str,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: DiT,
    loss_fn: FlowMatchingLoss,
    checkpoint_dir: Path,
    log_path: Path,
) -> dict:
    """Flow-matching training loop. Returns the final metrics dict."""
    tcfg, lcfg = config["training"], config["logging"]
    device = next(model.parameters()).device
    if device.type == "cuda":
        # scope the reported peak to this run on this device, not to whatever
        # else the process allocated first
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    warmup = tcfg["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) if warmup > 0 else 1.0
    )
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tcfg["ema_decay"]))

    total_steps = tcfg["total_steps"]
    recent: list[float] = []
    last_val = None
    started = time.time()

    def save(tag: str, step: int):
        torch.save(
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "config": config,
            },
            checkpoint_dir / f"{tag}.pt",
        )

    emit(log_path, WORKER, run_id, "train", "progress", step=0, total=total_steps,
         params=sum(p.numel() for p in model.parameters()))
    model.train()
    step = 0
    try:
        for batch in _infinite(train_loader):
            loss, _ = loss_fn(
                model,
                batch["x"].to(device, non_blocking=True),
                batch["sdf"].to(device, non_blocking=True),
                batch["scalars"].to(device, non_blocking=True),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
            opt.step()
            sched.step()
            ema.update_parameters(model)

            step += 1
            recent.append(loss.item())
            recent[:] = recent[-100:]

            if step % lcfg["log_every"] == 0:
                emit(log_path, WORKER, run_id, "train", "progress", step=step,
                     loss=f"{recent[-1]:.6f}", loss_avg100=f"{sum(recent) / len(recent):.6f}",
                     lr=f"{sched.get_last_lr()[0]:.3e}")
            if step % lcfg["progress_every"] == 0:
                emit(log_path, WORKER, run_id, "train", "heartbeat", step=step,
                     elapsed_sec=int(time.time() - started))
                print(f"step {step}/{total_steps} loss {recent[-1]:.6f} "
                      f"elapsed {int(time.time() - started)}s", flush=True)
            if step % lcfg["val_every"] == 0:
                last_val = _validate(model, loss_fn, val_loader, device)
                emit(log_path, WORKER, run_id, "val", "progress", step=step,
                     val_loss=f"{last_val:.6f}")
            if step % lcfg["checkpoint_every"] == 0:
                save(f"step_{step}", step)
                emit(log_path, WORKER, run_id, "train", "progress", step=step,
                     checkpoint=f"step_{step}.pt")
            if step >= total_steps:
                break
    except BaseException as exc:  # includes KeyboardInterrupt / CUDA OOM
        emit(log_path, WORKER, run_id, "train", "error", step=step,
             exc_type=type(exc).__name__, msg=_one_line(exc))
        raise

    if last_val is None or step % lcfg["val_every"]:
        last_val = _validate(model, loss_fn, val_loader, device)
    save("final", step)

    metrics = {
        "steps": step,
        "train_loss_last100": sum(recent) / len(recent) if recent else None,
        "val_loss": last_val,
        "elapsed_sec": time.time() - started,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    emit(log_path, WORKER, run_id, "train", "complete", step=step,
         train_loss=f"{metrics['train_loss_last100']:.6f}",
         val_loss=f"{last_val:.6f}", elapsed_sec=int(metrics["elapsed_sec"]))
    return metrics

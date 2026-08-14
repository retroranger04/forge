"""Runs training as configured in configs/run_01.yaml.
All progress emitted to configured log path with run_id from config.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # forge is not installed; it resolves from the repo root

from forge.data.dataset import ForgeSplitDataset  # noqa: E402
from forge.logging import emit  # noqa: E402
from forge.models.dit import DiT  # noqa: E402
from forge.train.flow_matching import FlowMatchingLoss, train_one_run  # noqa: E402

# 27.0 ms/step measured, against 36.4 at 0 workers and 33.0 at 8
NUM_WORKERS = 4


def main():
    config = yaml.safe_load((ROOT / "configs/run_01.yaml").read_text(encoding="utf-8"))
    config["run_started_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = config["run_id"]
    dcfg, pcfg = config["data"], config["paths"]
    log_path = ROOT / pcfg["log_path"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; run_01 is a GPU run")
    device = torch.device("cuda")

    stats_path = ROOT / dcfg["stats_path"]
    train_ds = ForgeSplitDataset(ROOT / dcfg["train_dir"], True, stats_path)
    val_full = ForgeSplitDataset(ROOT / dcfg["val_dir"], True, stats_path)
    val_ds = Subset(val_full, range(dcfg["val_subset_size"]))

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    # 100 samples: worker startup would cost more than the loading it saves
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0, pin_memory=True)

    model = DiT(**config["model"]).to(device)
    loss_fn = FlowMatchingLoss(config["flow_matching"]["sigma_min"])

    emit(log_path, "trainer", run_id, "train", "progress",
         launched_at=config["run_started_at"], device=torch.cuda.get_device_name(0),
         train_samples=len(train_ds), val_samples=len(val_ds),
         batch_size=batch_size, num_workers=NUM_WORKERS,
         total_steps=config["training"]["total_steps"])

    metrics = train_one_run(
        run_id=run_id,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        checkpoint_dir=ROOT / pcfg["checkpoint_dir"],
        log_path=log_path,
    )

    final_path = ROOT / pcfg["final_model_path"]
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": metrics["steps"], "model": model.state_dict(),
                "config": config, "metrics": metrics}, final_path)
    print(f"done: {metrics}", flush=True)


if __name__ == "__main__":
    # required on Windows: DataLoader workers spawn and re-import this module
    main()

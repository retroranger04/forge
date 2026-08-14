"""Split conformal calibration on ID plane-stress data and coverage
measurement on the ID and OOD splits. Inference only; the model is frozen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge.data.dataset import ForgeSplitDataset  # noqa: E402
from forge.eval.conformal import calibrate_quantile  # noqa: E402
from forge.eval.coverage import evaluate_split_coverage, generate_split_scores  # noqa: E402
from forge.logging import emit  # noqa: E402
from forge.models.dit import DiT  # noqa: E402

WORKER = "coverage_eval"
SCORE = "sample_mean_absolute_residual"
ID_COVERAGE_TOLERANCE = 0.05  # |empirical - nominal| above this = calibration broken
Q90_REFERENCE = 0.002333      # Task 4 small-scale q_hat at 0.90, anomaly reference only


def load_ema_model(ckpt_path: Path, device) -> tuple[DiT, dict]:
    """DiT with the checkpoint's EMA weights. Strict load: a partial one would
    leave randomly initialised tensors in place and publish their coverage."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DiT(**ckpt["config"]["model"]).to(device)
    ema_state = {k[len("module."):]: v for k, v in ckpt["ema"].items() if k.startswith("module.")}
    if not ema_state:
        raise RuntimeError("checkpoint contains no EMA model parameters")
    model.load_state_dict(ema_state, strict=True)
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pass: 4 conditions per split, M=50")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/eval.yaml").read_text(encoding="utf-8"))
    run_id, dcfg, scfg = cfg["run_id"], cfg["data"], cfg["sampling"]
    levels = cfg["calibration"]["nominal_levels"]
    n_cal = cfg["calibration"]["calibration_split_size"]
    excl = cfg["calibration"]["excluded_head_size"]
    log_path = ROOT / cfg["paths"]["eval_log_path"]
    (ROOT / cfg["paths"]["results_path"]).parent.mkdir(parents=True, exist_ok=True)
    if cfg["calibration"]["score"] != SCORE:
        # the field is descriptive, not a switch; an edited value that silently
        # changed nothing is the drift this check exists to stop
        raise SystemExit(f"only score {SCORE!r} is implemented, config asks for "
                         f"{cfg['calibration']['score']!r}")

    m = 50 if args.smoke else scfg["num_samples_per_condition"]
    n_each = 4 if args.smoke else None  # None = whole split
    sample_kw = dict(num_samples_per_condition=m, num_fm_steps=scfg["num_fm_steps"],
                     batch_size=scfg["batch_size"], log_path=log_path, run_id=run_id,
                     worker_id=WORKER)

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model, ckpt = load_ema_model(ROOT / cfg["model_checkpoint"], device)
    stats_path = ROOT / dcfg["train_stats_path"]
    vm_std = json.loads(stats_path.read_text(encoding="utf-8"))["von_mises"]["std"]
    print(f"checkpoint step {ckpt['step']}, M={m}, "
          f"{scfg['num_fm_steps']} FM steps, batch {scfg['batch_size']}")

    def loader(split_dir: str, index_range: range | None):
        ds = ForgeSplitDataset(ROOT / split_dir, True, stats_path)
        if index_range is not None:
            ds = Subset(ds, index_range)
        return DataLoader(ds, batch_size=scfg["batch_size"])

    def split_len(split_dir: str) -> int:
        return len(ForgeSplitDataset(ROOT / split_dir, True, stats_path))

    # id_test partition is derived, never hardcoded: an excluded head, then
    # calibration, then evaluation over the remainder. Changing either size in
    # the config cannot silently drop or double-count samples.
    id_len = split_len(dcfg["id_test_dir"])
    cal_count = n_each if n_each else n_cal
    if excl + cal_count + (n_each or 1) > id_len:
        raise SystemExit(
            f"id_test has {id_len} samples; excluding {excl} and calibrating on "
            f"{cal_count} leaves none to evaluate"
        )
    cal_idx = range(excl, excl + cal_count)
    id_eval_idx = (range(cal_idx.stop, cal_idx.stop + n_each) if n_each
                   else range(excl + n_cal, id_len))

    timings = {}

    # --- calibration: head of id_test ----------------------------------------
    t0 = time.time()
    cal_loader = loader(dcfg["id_test_dir"], cal_idx)
    cal_scores, cal_masks = generate_split_scores(
        model, cal_loader, split_name="calibration", progress_every=50, seed=0, **sample_kw)
    timings["calibration"] = time.time() - t0

    q_hats = {lv: calibrate_quantile(cal_scores, cal_masks, 1.0 - lv) for lv in levels}
    for lv in levels:
        if not q_hats[lv] > 0:
            raise SystemExit(f"q_hat at {lv} is not > 0: {q_hats[lv]}")
    ordered = [q_hats[lv] for lv in sorted(levels)]
    if ordered != sorted(ordered):
        raise SystemExit(f"q_hat not increasing with nominal level: {q_hats}")
    print(f"q_hat (normalized units): " +
          ", ".join(f"{lv:.2f}->{q_hats[lv]:.6f}" for lv in sorted(levels)))
    # not a stop condition: a large move from the small-scale reference is a
    # signal to investigate after the full results are in, not before
    ratio = q_hats[0.90] / Q90_REFERENCE
    q90_anomaly = ratio > 5.0 or ratio < 0.2
    if q90_anomaly:
        print(f"ANOMALY: q_hat@0.90 is {ratio:.2f}x the small-scale reference "
              f"{Q90_REFERENCE:.6f}; continuing")
        emit(log_path, WORKER, run_id, "calibration", "warning",
             msg="q90_far_from_smallscale_reference", ratio=f"{ratio:.3f}")
    emit(log_path, WORKER, run_id, "calibration", "complete",
         n=int(cal_scores.shape[0]), elapsed_sec=int(timings["calibration"]),
         **{f"q{int(lv * 100)}": f"{q_hats[lv]:.6f}" for lv in sorted(levels)})

    results = {
        "checkpoint_step": int(ckpt["step"]),
        "smoke": args.smoke,
        "num_samples_per_condition": m,
        "num_fm_steps": scfg["num_fm_steps"],
        "n_calibration": int(cal_scores.shape[0]),
        "score": SCORE,
        "coverage_definition": "pooled empirical per-pixel coverage; "
                               "no finite-sample split-conformal guarantee claimed",
        "id_test_partition": {
            # excluded head was run_01's training validation subset
            "excluded_head": [0, excl],
            "calibration": [cal_idx.start, cal_idx.stop],
            "evaluation": [id_eval_idx.start, id_eval_idx.stop],
        },
        # q_hat and widths are residuals, so only the scale factor applies:
        # physical_width = normalized_width * von_mises_std. Coverage itself is
        # unit-invariant (the normalisation is a positive affine map).
        "units": "normalized; physical residual = normalized * von_mises_std",
        "von_mises_std": vm_std,
        "q_hat": {f"{lv:.2f}": q_hats[lv] for lv in sorted(levels)},
        "q90_anomaly_vs_smallscale": q90_anomaly,
        "splits": {},
    }
    # calibration scores are large; free before the eval splits allocate theirs
    del cal_scores, cal_masks

    # --- evaluation splits ---------------------------------------------------
    eval_splits = [
        ("id_test", dcfg["id_test_dir"], id_eval_idx),
        ("ood_fidelity", dcfg["ood_fidelity_dir"], range(n_each) if n_each else None),
        ("ood_geometry", dcfg["ood_geometry_dir"], range(n_each) if n_each else None),
    ]
    for name, split_dir, idx in eval_splits:
        t0 = time.time()
        row = evaluate_split_coverage(
            model, loader(split_dir, idx), q_hats, split_name=name, progress_every=100,
            seed=1000, **sample_kw)
        timings[name] = time.time() - t0
        results["splits"][name] = row
        print(f"{name}: n={row['n_samples']} " + " ".join(
            f"cov@{int(lv * 100)}={row[f'coverage_at_{int(lv * 100)}']:.4f}"
            for lv in sorted(levels))
            + f" mean|resid|={row['mean_absolute_residual']:.6f}"
            + f"  [{timings[name] / 60:.1f} min]")

        # ID coverage far from nominal means calibration is broken; stop before
        # spending two hours measuring OOD against a threshold we cannot trust
        if name == "id_test" and not args.smoke:
            off = {lv: row[f"coverage_at_{int(lv * 100)}"] - lv for lv in sorted(levels)}
            if any(abs(d) > ID_COVERAGE_TOLERANCE for d in off.values()):
                results["splits"][name]["halted"] = "ID coverage off nominal"
                out = ROOT / cfg["paths"]["results_path"]
                out.write_text(json.dumps(results, indent=2), encoding="utf-8")
                emit(log_path, WORKER, run_id, "id_test", "error",
                     msg="ID_coverage_off_nominal",
                     **{f"d{int(lv * 100)}": f"{off[lv]:+.4f}" for lv in sorted(levels)})
                raise SystemExit(
                    "STOP: ID empirical coverage deviates from nominal by more than "
                    f"{ID_COVERAGE_TOLERANCE} ("
                    + ", ".join(f"{lv:.2f}: {off[lv]:+.4f}" for lv in sorted(levels))
                    + f"). Calibration is not trustworthy; wrote {out}"
                )

    # --- coverage gap vs ID --------------------------------------------------
    id_row = results["splits"]["id_test"]
    for name in ("ood_fidelity", "ood_geometry"):
        results["splits"][name]["coverage_gap_vs_id"] = {
            f"{lv:.2f}": id_row[f"coverage_at_{int(lv * 100)}"]
            - results["splits"][name][f"coverage_at_{int(lv * 100)}"]
            for lv in sorted(levels)
        }

    results["timings_sec"] = timings
    results["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated(device))
    out = ROOT / cfg["paths"]["results_path"]
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"peak VRAM {results['peak_vram_bytes'] / 2**30:.2f} GiB -> wrote {out}")


if __name__ == "__main__":
    main()

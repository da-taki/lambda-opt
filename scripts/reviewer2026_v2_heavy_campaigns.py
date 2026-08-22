from __future__ import annotations

import gc
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_reviewer2026_v2 as runner

base = runner.base
ROOT = runner.ROOT
OUT = runner.OUT
H = runner.H
SCALE_CONFIGS = runner.SCALE_CONFIGS
SCALE_SEEDS = runner.SCALE_SEEDS
LONG_SEEDS = runner.LONG_SEEDS


def protocol_hash_ok(out_dir: Path) -> dict[str, Any]:
    protocol = out_dir / "FROZEN_PROTOCOL.json"
    recorded = out_dir / "FROZEN_PROTOCOL.sha256.json"
    if not protocol.exists() or not recorded.exists():
        raise FileNotFoundError(f"missing frozen protocol files in {out_dir}")
    actual = runner.sha256(protocol)
    expected = json.loads(recorded.read_text(encoding="utf-8"))["sha256"]
    if actual != expected:
        raise RuntimeError(f"frozen protocol hash mismatch for {protocol}: {actual} != {expected}")
    return {"path": str(protocol.relative_to(ROOT)), "expected_sha256": expected, "actual_sha256": actual, "status": "PASS"}


def campaign_key(row: dict[str, Any]) -> str:
    return "|".join(str(row[k]) for k in ["model_config", "seed", "checkpoint_age", "rewrite_family"])


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def train_checkpoints(
    *,
    seed: int,
    cfg: dict[str, Any],
    out_dir: Path,
    training_steps: int,
    checkpoint_ages: list[int],
    stale_steps: list[int],
    scheduler_step: int,
    model_label: str,
) -> dict[str, Any]:
    ckpt_dir = out_dir / "checkpoints" / model_label / f"seed_{seed}"
    required = [ckpt_dir / f"checkpoint_step_{age}.pt" for age in checkpoint_ages]
    required += [ckpt_dir / f"optimizer_state_step_{step}.pt" for step in stale_steps]
    if all(p.exists() for p in required):
        payload = base.load_checkpoint(ckpt_dir / f"checkpoint_step_{checkpoint_ages[-1]}.pt")
        model = base.make_model(cfg, int(payload["vocab"]))
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
        return {
            "seed": seed,
            "model_config": cfg["name"],
            "training_steps": training_steps,
            "n_params": nparam,
            "vocab": int(payload["vocab"]),
            "dataset": payload["dataset"],
            "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)),
            "scheduler_step_size": scheduler_step,
            "reused_existing_checkpoints": True,
        }

    old_sched_step = base.SCHED_STEP
    base.SCHED_STEP = scheduler_step
    try:
        base.set_seed(seed)
        batches, vocab, dataset = base.load_batches(seed, training_steps + H + 50)
        model = base.make_model(cfg, vocab).to(base.DEVICE)
        nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt = base.make_optimizer(model)
        sched = base.make_scheduler(opt)
        losses: list[float] = []
        step_times: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        for step in range(1, training_steps + 1):
            t0 = time.perf_counter()
            losses.append(base.train_one(model, opt, sched, batches[(step - 1) % len(batches)]))
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
            if step in stale_steps:
                path = ckpt_dir / f"optimizer_state_step_{step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"optimizer_state": base.cpu_clone(opt.state_dict()), "training_step": step, "seed": seed}, path)
            if step in checkpoint_ages:
                base.save_checkpoint(
                    ckpt_dir / f"checkpoint_step_{step}.pt",
                    base.checkpoint_payload(model, opt, sched, seed=seed, step=step, cfg=cfg, vocab=vocab, dataset=dataset),
                )
        torch.cuda.synchronize()
        meta = {
            "seed": seed,
            "model_config": cfg["name"],
            "training_steps": training_steps,
            "n_params": nparam,
            "vocab": vocab,
            "dataset": dataset,
            "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)),
            "train_loss_final": losses[-1],
            "mean_step_time_seconds": float(np.mean(step_times)),
            "tokens_per_second": float(base.SEQ_LEN / np.mean(step_times)),
            "total_training_wall_seconds": time.perf_counter() - t_start,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "scheduler_step_size": scheduler_step,
            "reused_existing_checkpoints": False,
        }
        del model, opt, sched
        torch.cuda.empty_cache()
        return meta
    finally:
        base.SCHED_STEP = old_sched_step


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    for value in sorted({r[key] for r in rows}, key=str):
        rs = [r for r in rows if r[key] == value]
        tvals = [r["T_tau"] for r in rs if r["T_tau"] is not None]
        out.append({
            key: value,
            "rows": len(rs),
            "crossed_h100": len(tvals),
            "task_nonrecovered_h100": sum(bool(r["task_nonrecovered_h100"]) for r in rs),
            "median_T_tau": float(np.median(tvals)) if tvals else "",
            "max_validation_loss_gap_mean": float(np.mean([float(r["max_validation_loss_gap"]) for r in rs])) if rs else "",
            "max_parameter_divergence_mean": float(np.mean([float(r["max_parameter_divergence"]) for r in rs])) if rs else "",
        })
    return out


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "max_parameter_divergence",
        "max_normalized_parameter_divergence",
        "max_clean_update_normalized_divergence",
        "max_layer_relative_displacement",
        "max_calibration_kl",
        "max_calibration_rms_logit",
        "max_validation_loss_gap",
    ]
    out: dict[str, Any] = {"rows": len(rows)}
    for metric in metrics:
        vals = [float(r[metric]) for r in rows if r.get(metric) not in ("", None)]
        out[metric] = {
            "min": min(vals) if vals else "",
            "median": float(np.median(vals)) if vals else "",
            "mean": float(np.mean(vals)) if vals else "",
            "max": max(vals) if vals else "",
        }
    return out


def summarize_campaign(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    expected_rows: int,
    required_sizes: set[str],
    required_seeds: set[int],
    required_ages: set[int],
    protocol_check: dict[str, Any],
    kind: str,
) -> None:
    keys = [campaign_key(r) for r in rows]
    errors = []
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    if len(set(keys)) != len(keys):
        errors.append("duplicate campaign keys")
    if {r["model_config"] for r in rows} != required_sizes:
        errors.append("model-config set mismatch")
    if {int(r["seed"]) for r in rows} != required_seeds:
        errors.append("seed set mismatch")
    if {int(r["checkpoint_age"]) for r in rows} != required_ages:
        errors.append("checkpoint-age set mismatch")
    if {r["rewrite_family"] for r in rows} != set(base.REWRITE_FAMILIES):
        errors.append("rewrite-family set mismatch")
    if len(step_rows) != expected_rows * (H + 1):
        errors.append(f"expected {expected_rows * (H + 1)} trajectory rows, found {len(step_rows)}")
    if protocol_check.get("status") != "PASS":
        errors.append("frozen protocol hash check did not pass")
    if errors:
        raise RuntimeError("; ".join(errors))

    hist = Counter(str(r["T_tau"]) for r in rows if r["T_tau"] is not None)
    runner.write_json(out_dir / "latency_histogram.json", {str(i): hist.get(str(i), 0) for i in range(0, H + 1)})
    runner.write_json(out_dir / "metric_summary.json", metric_summary(rows))
    runner.write_csv(out_dir / "per_seed_summary.csv", group_summary(rows, "seed"))
    runner.write_csv(out_dir / "per_family_summary.csv", group_summary(rows, "rewrite_family"))
    if kind == "controlled_scale":
        runner.write_csv(out_dir / "per_size_summary.csv", group_summary(rows, "model_config"))
    else:
        runner.write_csv(out_dir / "per_checkpoint_summary.csv", group_summary(rows, "checkpoint_age"))

    summary = {
        "status": "COMPLETE",
        "kind": kind,
        "rows": len(rows),
        "trajectory_rows": len(step_rows),
        "model_configs": sorted(required_sizes),
        "seeds": sorted(required_seeds),
        "checkpoint_ages": sorted(required_ages),
        "rewrite_families": base.REWRITE_FAMILIES,
        "horizon": H,
        "crossed_h100": sum(r["T_tau"] is not None for r in rows),
        "task_nonrecovered_h100": sum(bool(r["task_nonrecovered_h100"]) for r in rows),
        "protocol_hash": protocol_check,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    runner.write_json(out_dir / "summary.json", summary)
    runner.write_md(out_dir / "STATUS.md", f"""# Status

COMPLETE.

- Rows: {summary['rows']}
- Trajectory rows: {summary['trajectory_rows']}
- Frozen protocol hash: {protocol_check['status']}
""")
    runner.write_md(out_dir / "RUN_LOG.md", f"""# Run Log

- Campaign: {kind}
- Completed UTC: {summary['completed_utc']}
- Rows written: {summary['rows']}
- SHA-256 protocol verification: {protocol_check['actual_sha256']}
""")
    blocker = out_dir / "BLOCKER.json"
    if blocker.exists():
        blocker.unlink()


def run_campaign_grid(
    *,
    out_dir: Path,
    configs: list[dict[str, Any]],
    seeds: list[int],
    training_steps: int,
    checkpoint_ages: list[int],
    stale_age: int,
    scheduler_step: int,
    kind: str,
) -> None:
    runner.set_horizon()
    protocol_check = protocol_hash_ok(out_dir)
    rows = load_rows(out_dir / "rows.json")
    step_rows: list[dict[str, Any]] = runner.read_csv(out_dir / "trajectory_steps.csv") if (out_dir / "trajectory_steps.csv").exists() else []
    done = {campaign_key(r) for r in rows}
    metas: list[dict[str, Any]] = json.loads((out_dir / "training_metadata.json").read_text(encoding="utf-8")) if (out_dir / "training_metadata.json").exists() else []
    meta_done = {(m["model_config"], int(m["seed"])) for m in metas}
    stale_steps = [age - stale_age for age in checkpoint_ages]

    old_sched_step = base.SCHED_STEP
    base.SCHED_STEP = scheduler_step
    try:
        for cfg in configs:
            for seed in seeds:
                model_label = cfg["name"] if kind == "controlled_scale" else "candidate_B_19p48m"
                if (cfg["name"], seed) not in meta_done:
                    meta = train_checkpoints(
                        seed=seed,
                        cfg=cfg,
                        out_dir=out_dir,
                        training_steps=training_steps,
                        checkpoint_ages=checkpoint_ages,
                        stale_steps=stale_steps,
                        scheduler_step=scheduler_step,
                        model_label=model_label,
                    )
                    metas.append(meta)
                    meta_done.add((cfg["name"], seed))
                    runner.write_json(out_dir / "training_metadata.json", metas)
                batches, vocab, dataset = base.load_batches(seed, training_steps + H + 50)
                val_batches = batches[training_steps + 1 : training_steps + 4]
                calibration_batch = batches[training_steps + 4]
                ckpt_dir = out_dir / "checkpoints" / model_label / f"seed_{seed}"
                for age in checkpoint_ages:
                    base_file = ckpt_dir / f"checkpoint_step_{age}.pt"
                    stale_file = ckpt_dir / f"optimizer_state_step_{age - stale_age}.pt"
                    payload = base.load_checkpoint(base_file)
                    stale = base.load_checkpoint(stale_file)["optimizer_state"] if stale_file.exists() else None
                    for fam in base.REWRITE_FAMILIES:
                        row_id = f"{kind}_{cfg['name']}_seed{seed}_ckpt{age}_{fam}"
                        row_key = "|".join([cfg["name"], str(seed), str(age), fam])
                        if row_key in done:
                            continue
                        t0 = time.perf_counter()
                        summary, steps, _ = runner.run_pair(payload, fam, stale, batches, val_batches, calibration_batch)
                        summary.update({
                            "row_id": row_id,
                            "experiment": f"reviewer2026_v2_{kind}",
                            "model_config": cfg["name"],
                            "seed": seed,
                            "checkpoint_age": age,
                            "training_steps": training_steps,
                            "rewrite_family": fam,
                            "dataset": dataset,
                            "checkpoint_file": str(base_file.relative_to(ROOT)),
                            "stale_source_file": str(stale_file.relative_to(ROOT)) if stale_file.exists() and "stale" in fam else "",
                            "runtime_seconds": time.perf_counter() - t0,
                        })
                        rows.append(summary)
                        done.add(row_key)
                        for step_row in steps:
                            step_rows.append({
                                "row_id": row_id,
                                "model_config": cfg["name"],
                                "seed": seed,
                                "checkpoint_age": age,
                                "rewrite_family": fam,
                                **step_row,
                            })
                        runner.write_csv(out_dir / "rows.csv", rows)
                        runner.write_json(out_dir / "rows.json", rows)
                        runner.write_csv(out_dir / "trajectory_steps.csv", step_rows)
                        runner.write_json(out_dir / "completed_keys.json", sorted(campaign_key(r) for r in rows))
                        gc.collect()
    finally:
        base.SCHED_STEP = old_sched_step

    summarize_campaign(
        out_dir=out_dir,
        rows=rows,
        step_rows=step_rows,
        expected_rows=len(configs) * len(seeds) * len(checkpoint_ages) * len(base.REWRITE_FAMILIES),
        required_sizes={cfg["name"] for cfg in configs},
        required_seeds=set(seeds),
        required_ages=set(checkpoint_ages),
        protocol_check=protocol_check,
        kind=kind,
    )


def run_controlled_scale() -> None:
    run_campaign_grid(
        out_dir=OUT / "controlled_scale",
        configs=SCALE_CONFIGS,
        seeds=SCALE_SEEDS,
        training_steps=300,
        checkpoint_ages=[75, 150, 300],
        stale_age=25,
        scheduler_step=100,
        kind="controlled_scale",
    )


def run_long_training() -> None:
    run_campaign_grid(
        out_dir=OUT / "long_training",
        configs=[base.MODEL_CFG],
        seeds=LONG_SEEDS,
        training_steps=3000,
        checkpoint_ages=[750, 1500, 3000],
        stale_age=25,
        scheduler_step=1000,
        kind="long_training",
    )
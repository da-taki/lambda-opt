from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments" / "scripts"), str(ROOT / "experiments")]
import run_opt2026_10m_gpu_final as base  # noqa: E402
import run_reviewer2026_v2 as v2  # noqa: E402
import run_reviewer2026_v3_lr_isolation as v3  # noqa: E402

OUT = ROOT / "results" / "reviewer2026_v5_final_eval"
V2 = ROOT / "results" / "reviewer2026_v2"
V3 = ROOT / "results" / "reviewer2026_v3_lr_isolation"
SRC = ROOT / "results" / "opt2026_10m_gpu_final"

SOURCE_EXPERIMENT_COMMIT = "622c80150ec1c199539af1ff300e33521cd56f78"
V4_TRANSPORT_COMMIT = "3c37a47092ad9409a9542ab403c3846c4bf199fc"
H = 100
H20 = 20
EPS = 1e-12
PRIMARY_SEEDS = [1101, 1102, 1103]
PRIMARY_AGES = [75, 150, 300]
CONTROLLED_SEEDS = [1201, 1202, 1203]
CONTROLLED_AGES = [75, 150, 300]
LONG_SEEDS = [1301, 1302, 1303]
LONG_AGES = [750, 1500, 3000]
FUTURE_SCHEDULE_SEEDS = [8101, 8102, 8103, 8104, 8105]
EVAL_INDICES = [3500, 3501, 3502]
CAL_INDEX = 3503
TASK_THRESHOLDS = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0]
TAU_FRACTIONS = [0.01, 0.025, 0.05, 0.10, 0.20]
METRICS = {
    "global_normalized_parameter_divergence": "max_normalized_parameter_divergence",
    "clean_update_normalized_divergence": "max_clean_update_normalized_divergence",
    "max_layer_relative_displacement": "max_layer_relative_displacement",
    "function_space_kl": "max_calibration_kl",
    "function_space_rms_logit": "max_calibration_rms_logit",
}


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if torch.is_tensor(x):
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def tensor_hash(batch: Any) -> str:
    h = hashlib.sha256()
    for tensor in batch:
        arr = tensor.detach().cpu().contiguous()
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def load_batches_full(seed: int):
    return base.load_batches(seed, CAL_INDEX + 1)


def batch_shape_metadata(seed: int = 1101) -> dict[str, Any]:
    data_path = ROOT / "experiments" / "data" / "wikitext2_raw" / "train.txt"
    text = data_path.read_text(encoding="utf-8")[: base.DATA_CHARS]
    all_batches, vocab = base.build_char_dataset(text, base.SEQ_LEN, base.BATCH_SIZE, seed=seed)
    batches = all_batches[: CAL_INDEX + 1]
    dataset = f"WikiText-2 cached char-level, first {len(text)} chars"
    x, y = batches[EVAL_INDICES[0]]
    return {
        "probe_seed": seed,
        "dataset": dataset,
        "available_batch_count": len(all_batches),
        "available_batch_count_checked": len(batches),
        "vocab_size": vocab,
        "raw_chunk_length_characters": base.SEQ_LEN,
        "batch_size": int(x.shape[0]),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "model_input_length_positions": int(x.shape[1]),
        "target_length_per_batch": int(y.shape[1]),
        "evaluation_batches": len(EVAL_INDICES),
        "evaluation_target_positions_total": int(len(EVAL_INDICES) * y.shape[0] * y.shape[1]),
        "calibration_prediction_positions": int(y.shape[0] * y.shape[1]),
    }


def freeze_protocol() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = batch_shape_metadata()
    if meta["available_batch_count_checked"] <= CAL_INDEX:
        raise RuntimeError(f"held-out calibration index {CAL_INDEX} unavailable")
    protocol = {
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_commit": SOURCE_EXPERIMENT_COMMIT,
        "v4_transport_commit": V4_TRANSPORT_COMMIT,
        "dataset": meta["dataset"],
        "corpus_path": str((ROOT / "experiments" / "data" / "wikitext2_raw" / "train.txt").relative_to(ROOT)),
        "corpus_characters_read": base.DATA_CHARS,
        "deterministic_dataset_builder": "experiments/scripts/run_transformer.py::build_char_dataset",
        "zero_based_evaluation_batch_indices": EVAL_INDICES,
        "zero_based_calibration_batch_index": CAL_INDEX,
        "primary_controlled_training_max_index": 299,
        "primary_controlled_h100_continuation_max_index": 399,
        "long_training_max_index": 2999,
        "long_h100_continuation_max_index": 3099,
        "task_probe": "mean source/candidate cross-entropy over the three held-out evaluation batches; absolute gap",
        "task_recovery": "gap <= 0.05 for 3 consecutive continuation evaluations",
        "function_metrics": "mean KL(source||candidate) and RMS logit difference on held-out calibration batch",
        "task_threshold_primary": 0.05,
        "retuning_after_results": False,
        **meta,
    }
    path = OUT / "FROZEN_EVALUATION_PROTOCOL.json"
    write_json(path, protocol)
    write_json(OUT / "FROZEN_EVALUATION_PROTOCOL.sha256.json", {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    write_json(OUT / "final_evaluation_protocol_metadata.json", protocol)


class sched_step:
    def __init__(self, value: int):
        self.value = value
        self.old = None

    def __enter__(self):
        self.old = base.SCHED_STEP
        base.SCHED_STEP = self.value

    def __exit__(self, *args):
        base.SCHED_STEP = self.old


def p_primary(seed: int, age: int) -> Path:
    return SRC / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"checkpoint_step_{age}.pt"


def stale_primary(seed: int, age: int) -> Path:
    return SRC / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"optimizer_state_step_{age - base.STALE_AGE}.pt"


def p_control(cfg: str, seed: int, age: int) -> Path:
    return V2 / "controlled_scale" / "checkpoints" / cfg / f"seed_{seed}" / f"checkpoint_step_{age}.pt"


def stale_control(cfg: str, seed: int, age: int) -> Path:
    return V2 / "controlled_scale" / "checkpoints" / cfg / f"seed_{seed}" / f"optimizer_state_step_{age - base.STALE_AGE}.pt"


def p_long(seed: int, age: int) -> Path:
    return V2 / "long_training" / "checkpoints" / "candidate_B_19p48m" / f"seed_{seed}" / f"checkpoint_step_{age}.pt"


def stale_long(seed: int, age: int) -> Path:
    return V2 / "long_training" / "checkpoints" / "candidate_B_19p48m" / f"seed_{seed}" / f"optimizer_state_step_{age - 25}.pt"


def load_stale(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return base.load_checkpoint(path)["optimizer_state"]


def candidate_from_family(payload: dict[str, Any], family: str, stale: dict[str, Any] | None):
    if family == "optimizer_reset":
        return v3.fresh_from_payload_optimizer_state_reset_isolated(payload)
    scen = base.scenario_payload(payload, family, stale)
    load_opt = family != "optimizer_reset"
    load_sched = family != "scheduler_mismatch"
    model, opt, sched = base.fresh_from_payload(scen, load_optimizer=load_opt, load_scheduler=load_sched)
    audit = {
        "checkpoint_optimizer_lr": payload["optimizer_state"]["param_groups"][0]["lr"],
        "candidate_optimizer_lr": opt.param_groups[0]["lr"],
        "lr_matches_checkpoint": opt.param_groups[0]["lr"] == payload["optimizer_state"]["param_groups"][0]["lr"],
        "param_groups_match_checkpoint": v3.groups_match(opt, payload["optimizer_state"]),
        "scheduler_state_matches_checkpoint": sched.state_dict() == payload["scheduler_state"] if family != "scheduler_mismatch" else False,
        "initial_optimizer_state_entries": v3.opt_state_entries(opt),
        "initial_optimizer_state_keys": v3.opt_state_keys(opt),
    }
    return model, opt, sched, audit


def run_pair_final(payload: dict[str, Any], family: str, stale: dict[str, Any] | None, batches: list[Any], eval_batches: list[Any], calibration_batch: Any):
    clean_model, clean_opt, clean_sched = base.fresh_from_payload(payload)
    cand_model, cand_opt, cand_sched, audit = candidate_from_family(payload, family, stale)
    start = int(payload["batch_pos"])
    theta = v2.norm_model(clean_model)
    tau = v2.TAU_FRAC * (theta + EPS)
    cum_clean = cum_candidate = cum_update_discrepancy = 0.0
    divergences: list[float] = []
    norm_divergences: list[float] = []
    val_clean: list[float] = []
    val_candidate: list[float] = []
    val_gaps: list[float] = []
    ppl_gaps: list[float] = []
    lrs: list[float] = []
    candidate_lrs: list[float] = []
    update_metric: list[float] = [0.0]
    max_layer_metric: list[float] = []
    mean_layer_metric: list[float] = []
    kl_metric: list[float] = []
    rms_metric: list[float] = []
    steps: list[dict[str, Any]] = []

    div0 = v2.l2_from_params(clean_model, cand_model)
    layers0 = v2.layer_stats(clean_model, cand_model)
    f0 = v2.function_metrics(clean_model, cand_model, calibration_batch)
    divergences.append(div0)
    norm_divergences.append(div0 / (theta + EPS))
    max_layer_metric.append(layers0["max_layer_relative_displacement"])
    mean_layer_metric.append(layers0["mean_layer_relative_displacement"])
    kl_metric.append(f0["calibration_kl"])
    rms_metric.append(f0["calibration_rms_logit"])
    steps.append({
        "continuation_step": 0,
        "parameter_divergence": div0,
        "normalized_parameter_divergence": div0 / (theta + EPS),
        "validation_loss_gap": 0.0,
        "perplexity_gap": 0.0,
        "lr": clean_opt.param_groups[0]["lr"],
        "candidate_lr": cand_opt.param_groups[0]["lr"],
        "source_task_loss": "",
        "candidate_task_loss": "",
        "clean_cumulative_update_distance": cum_clean,
        "candidate_cumulative_update_distance": cum_candidate,
        "cumulative_update_discrepancy": cum_update_discrepancy,
        **layers0,
        **f0,
    })

    for k in range(1, H + 1):
        batch = batches[(start + k - 1) % len(batches)]
        prev_clean = v2.snapshot_named(clean_model)
        prev_candidate = v2.snapshot_named(cand_model)
        prev_clean_flat = v2.flat_gpu(clean_model)
        prev_candidate_flat = v2.flat_gpu(cand_model)
        base.train_one(clean_model, clean_opt, clean_sched, batch)
        base.train_one(cand_model, cand_opt, cand_sched, batch)
        clean_flat = v2.flat_gpu(clean_model)
        candidate_flat = v2.flat_gpu(cand_model)
        clean_update = clean_flat - prev_clean_flat
        candidate_update = candidate_flat - prev_candidate_flat
        delta_update = clean_update - candidate_update
        clean_update_norm = float(clean_update.norm().detach().cpu().item())
        candidate_update_norm = float(candidate_update.norm().detach().cpu().item())
        delta_update_norm = float(delta_update.norm().detach().cpu().item())
        update_cos = float((torch.dot(clean_update, candidate_update) / (clean_update.norm() * candidate_update.norm() + EPS)).detach().cpu().item()) if clean_update_norm > 0 and candidate_update_norm > 0 else math.nan
        cum_clean += clean_update_norm
        cum_candidate += candidate_update_norm
        cum_update_discrepancy += delta_update_norm
        del prev_clean_flat, prev_candidate_flat, clean_flat, candidate_flat, clean_update, candidate_update, delta_update

        div = v2.l2_from_params(clean_model, cand_model)
        src_loss = sum(v2.validation_loss(clean_model, b) for b in eval_batches) / len(eval_batches)
        cand_loss = sum(v2.validation_loss(cand_model, b) for b in eval_batches) / len(eval_batches)
        gap = abs(src_loss - cand_loss)
        ppl_gap = abs(math.exp(min(src_loss, 50)) - math.exp(min(cand_loss, 50)))
        layers = v2.layer_stats(clean_model, cand_model, prev_clean, prev_candidate)
        fmetrics = v2.function_metrics(clean_model, cand_model, calibration_batch)
        lr = clean_opt.param_groups[0]["lr"]
        candidate_lr = cand_opt.param_groups[0]["lr"]

        divergences.append(div)
        norm_divergences.append(div / (theta + EPS))
        val_clean.append(src_loss)
        val_candidate.append(cand_loss)
        val_gaps.append(gap)
        ppl_gaps.append(ppl_gap)
        lrs.append(lr)
        candidate_lrs.append(candidate_lr)
        update_metric.append(div / (cum_clean + EPS))
        max_layer_metric.append(layers["max_layer_relative_displacement"])
        mean_layer_metric.append(layers["mean_layer_relative_displacement"])
        kl_metric.append(fmetrics["calibration_kl"])
        rms_metric.append(fmetrics["calibration_rms_logit"])
        steps.append({
            "continuation_step": k,
            "parameter_divergence": div,
            "normalized_parameter_divergence": div / (theta + EPS),
            "validation_loss_gap": gap,
            "perplexity_gap": ppl_gap,
            "lr": lr,
            "candidate_lr": candidate_lr,
            "source_task_loss": src_loss,
            "candidate_task_loss": cand_loss,
            "clean_cumulative_update_distance": cum_clean,
            "candidate_cumulative_update_distance": cum_candidate,
            "cumulative_update_discrepancy": cum_update_discrepancy,
            "clean_update_norm": clean_update_norm,
            "candidate_update_norm": candidate_update_norm,
            "delta_update_norm": delta_update_norm,
            "update_cosine": update_cos,
            "clean_update_normalized_divergence": div / (cum_clean + EPS),
            **layers,
            **fmetrics,
        })

    t_tau = v2.first_crossing(divergences, tau)
    rec = v2.recovery_step(val_gaps, 0.05)
    summary = {
        "theta_checkpoint_norm": theta,
        "tau_numeric": tau,
        "T_tau": t_tau,
        "censored_at_100": t_tau is None,
        "max_parameter_divergence": max(divergences),
        "final_parameter_divergence": divergences[-1],
        "max_normalized_parameter_divergence": max(norm_divergences),
        "final_normalized_parameter_divergence": norm_divergences[-1],
        "validation_loss_base": val_clean,
        "validation_loss_rewrite": val_candidate,
        "validation_loss_gap": val_gaps,
        "perplexity_gap": ppl_gaps,
        "lr_by_step": lrs,
        "candidate_lr_by_step": candidate_lrs,
        "task_recovery_step": rec,
        "task_nonrecovered_h100": rec is None,
        "task_nonrecovered_h20": v2.recovery_step(val_gaps[:H20], 0.05) is None,
        "max_validation_loss_gap": max(val_gaps) if val_gaps else 0.0,
        "final_validation_loss_gap": val_gaps[-1] if val_gaps else 0.0,
        "max_perplexity_gap": max(ppl_gaps) if ppl_gaps else 0.0,
        "final_perplexity_gap": ppl_gaps[-1] if ppl_gaps else 0.0,
        "max_clean_update_normalized_divergence": max(update_metric),
        "final_clean_update_normalized_divergence": update_metric[-1],
        "max_layer_relative_displacement": max(max_layer_metric),
        "mean_layer_relative_displacement_max": max(mean_layer_metric),
        "max_calibration_kl": max(kl_metric),
        "final_calibration_kl": kl_metric[-1],
        "max_calibration_rms_logit": max(rms_metric),
        "final_calibration_rms_logit": rms_metric[-1],
        "checkpoint_optimizer_lr": audit["checkpoint_optimizer_lr"],
        "candidate_initial_lr": audit["candidate_optimizer_lr"],
        "candidate_lr_matches_checkpoint": audit["lr_matches_checkpoint"],
        "candidate_param_groups_match_checkpoint": audit["param_groups_match_checkpoint"],
        "candidate_scheduler_state_matches_checkpoint": audit["scheduler_state_matches_checkpoint"],
        "candidate_initial_optimizer_state_entries": audit["initial_optimizer_state_entries"],
        "candidate_initial_optimizer_state_keys": audit["initial_optimizer_state_keys"],
        "optimizer_reset_definition": "optimizer_state_reset_isolated" if family == "optimizer_reset" else "",
    }
    del clean_model, clean_opt, clean_sched, cand_model, cand_opt, cand_sched
    torch.cuda.empty_cache()
    return summary, steps


def group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    for value in sorted({r[key] for r in rows}, key=str):
        rs = [r for r in rows if r[key] == value]
        tv = [r["T_tau"] for r in rs if r["T_tau"] is not None]
        out.append({
            key: value,
            "rows": len(rs),
            "crossed_h20": sum(r["T_tau"] is not None and int(r["T_tau"]) <= H20 for r in rs),
            "crossed_h100": len(tv),
            "task_nonrecovered_h20": sum(bool(r["task_nonrecovered_h20"]) for r in rs),
            "task_nonrecovered_h100": sum(bool(r["task_nonrecovered_h100"]) for r in rs),
            "median_T_tau": float(np.median(tv)) if tv else "",
        })
    return out


def hist(rows: list[dict[str, Any]], tau_key: str = "T_tau") -> dict[str, int]:
    c = Counter(str(r[tau_key]) for r in rows if r.get(tau_key) is not None)
    return {str(i): c.get(str(i), 0) for i in range(H + 1)}


def aggregate(rows: list[dict[str, Any]], kind: str, extra_groups: list[str]) -> dict[str, Any]:
    out = {
        "status": "COMPLETE",
        "kind": kind,
        "rows": len(rows),
        "h20_crossings": sum(r["T_tau"] is not None and int(r["T_tau"]) <= H20 for r in rows),
        "h100_crossings": sum(r["T_tau"] is not None for r in rows),
        "h20_task_nonrecovered": sum(bool(r["task_nonrecovered_h20"]) for r in rows),
        "h100_task_nonrecovered": sum(bool(r["task_nonrecovered_h100"]) for r in rows),
        "latency_histogram": hist(rows),
        "per_rewrite_family": group_summary(rows, "rewrite_family"),
    }
    for key in extra_groups:
        out[f"per_{key}"] = group_summary(rows, key)
    return out


def run_campaign(kind: str) -> None:
    if kind == "primary":
        out = OUT / "primary"
        specs = [(seed, age, "candidate_B_19p48m", p_primary(seed, age), stale_primary(seed, age), 100, {}) for seed in PRIMARY_SEEDS for age in PRIMARY_AGES]
        expected = 99
        exp_name = "reviewer2026_v5_final_eval_primary"
        groups = ["seed", "checkpoint_age"]
    elif kind == "controlled":
        out = OUT / "controlled_width"
        specs = [(seed, age, cfg["name"], p_control(cfg["name"], seed, age), stale_control(cfg["name"], seed, age), 100, {"model_config": cfg["name"]}) for cfg in v2.SCALE_CONFIGS for seed in CONTROLLED_SEEDS for age in CONTROLLED_AGES]
        expected = 297
        exp_name = "reviewer2026_v5_final_eval_controlled_width"
        groups = ["model_config", "seed", "checkpoint_age"]
    elif kind == "long":
        out = OUT / "long_training"
        specs = [(seed, age, "candidate_B_19p48m", p_long(seed, age), stale_long(seed, age), 1000, {"model_config": "candidate_B_19p48m", "training_steps": 3000}) for seed in LONG_SEEDS for age in LONG_AGES]
        expected = 99
        exp_name = "reviewer2026_v5_final_eval_long_training"
        groups = ["seed", "checkpoint_age"]
    else:
        raise ValueError(kind)

    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for seed, age, model_config, ckpt_path, stale_path, step_size, extra in specs:
        batches, _, dataset = load_batches_full(seed)
        eval_batches = [batches[i] for i in EVAL_INDICES]
        calibration_batch = batches[CAL_INDEX]
        with sched_step(step_size):
            payload = base.load_checkpoint(ckpt_path)
            stale = load_stale(stale_path)
            for fam in base.REWRITE_FAMILIES:
                t0 = time.perf_counter()
                summary, steps = run_pair_final(payload, fam, stale, batches, eval_batches, calibration_batch)
                if kind == "controlled":
                    row_id = f"controlled_width_{model_config}_seed{seed}_ckpt{age}_{fam}"
                elif kind == "long":
                    row_id = f"long_training_candidate_B_19p48m_seed{seed}_ckpt{age}_{fam}"
                else:
                    row_id = f"gpu_seed{seed}_ckpt{age}_{fam}"
                row = {
                    **summary,
                    "row_id": row_id,
                    "experiment": exp_name,
                    "dataset": dataset,
                    "seed": seed,
                    "checkpoint_age": age,
                    "rewrite_family": fam,
                    "checkpoint_file": str(ckpt_path.relative_to(ROOT)),
                    "stale_source_file": str(stale_path.relative_to(ROOT)) if stale_path.exists() else "",
                    "source": "v5_heldout_eval_rerun_all_families",
                    "evaluation_indices": EVAL_INDICES,
                    "calibration_index": CAL_INDEX,
                    "runtime_seconds": time.perf_counter() - t0,
                    **extra,
                }
                rows.append(row)
                step_rows.extend({"row_id": row_id, "seed": seed, "checkpoint_age": age, "rewrite_family": fam, **extra, **s} for s in steps)
                if len(rows) % 5 == 0 or len(rows) == expected:
                    write_json(out / f"{kind}_rows.json", rows)
                    write_csv(out / f"{kind}_rows.csv", rows)
                    write_csv(out / f"{kind}_steps.csv", step_rows)
    if len(rows) != expected:
        raise RuntimeError(f"{kind} row count {len(rows)} != {expected}")
    summary = aggregate(rows, kind, groups)
    write_json(out / f"{kind}_summary.json", summary)
    write_csv(out / f"{kind}_per_family_summary.csv", summary["per_rewrite_family"])
    for key in groups:
        write_csv(out / f"{kind}_per_{key}_summary.csv", summary[f"per_{key}"])


def protocol_disjointness() -> None:
    rows = []
    tensor_checks = []

    def add(seed: int, campaign: str, age: int, train: range, cont: range):
        batches, _, _ = load_batches_full(seed)
        eval_set = set(EVAL_INDICES)
        cal_set = {CAL_INDEX}
        train_set = set(train)
        cont_set = set(cont)
        eval_hashes = {i: tensor_hash(batches[i]) for i in EVAL_INDICES}
        cal_hash = tensor_hash(batches[CAL_INDEX])
        consumed_hashes = {i: tensor_hash(batches[i]) for i in sorted(train_set | cont_set)}
        row = {
            "campaign": campaign,
            "seed": seed,
            "checkpoint_age": age,
            "training_indices_min": min(train_set) if train_set else None,
            "training_indices_max": max(train_set) if train_set else None,
            "continuation_indices_min": min(cont_set) if cont_set else None,
            "continuation_indices_max": max(cont_set) if cont_set else None,
            "evaluation_indices": EVAL_INDICES,
            "calibration_index": CAL_INDEX,
            "intersection_training_evaluation": sorted(train_set & eval_set),
            "intersection_continuation_evaluation": sorted(cont_set & eval_set),
            "intersection_training_calibration": sorted(train_set & cal_set),
            "intersection_continuation_calibration": sorted(cont_set & cal_set),
            "intersection_evaluation_calibration": sorted(eval_set & cal_set),
            "evaluation_batch_hashes": eval_hashes,
            "calibration_batch_hash": cal_hash,
            "tensor_hash_disjoint_from_consumed": all(h not in set(consumed_hashes.values()) for h in list(eval_hashes.values()) + [cal_hash]),
        }
        rows.append(row)
        tensor_checks.append(row["tensor_hash_disjoint_from_consumed"])

    for seed in PRIMARY_SEEDS:
        for age in PRIMARY_AGES:
            add(seed, "primary", age, range(0, age), range(age, age + H))
    for seed in CONTROLLED_SEEDS:
        for age in CONTROLLED_AGES:
            add(seed, "controlled_width", age, range(0, age), range(age, age + H))
    for seed in LONG_SEEDS:
        for age in LONG_AGES:
            add(seed, "long_training", age, range(0, age), range(age, age + H))

    mf_rows = []
    for seed in FUTURE_SCHEDULE_SEEDS:
        batches, _, _ = base.load_batches(seed, 105)
        eval_set = {1, 2, 3}
        cal_set = {4}
        cont_set = set(range(5, 105))
        eval_hashes = {i: tensor_hash(batches[i]) for i in sorted(eval_set)}
        cal_hash = tensor_hash(batches[4])
        cont_hashes = {i: tensor_hash(batches[i]) for i in sorted(cont_set)}
        mf_rows.append({
            "future_schedule_seed": seed,
            "evaluation_indices": [1, 2, 3],
            "calibration_index": 4,
            "continuation_indices_min": 5,
            "continuation_indices_max": 104,
            "intersection_continuation_evaluation": sorted(cont_set & eval_set),
            "intersection_continuation_calibration": sorted(cont_set & cal_set),
            "intersection_evaluation_calibration": sorted(eval_set & cal_set),
            "evaluation_batch_hashes": eval_hashes,
            "calibration_batch_hash": cal_hash,
            "tensor_hash_disjoint_from_consumed": all(h not in set(cont_hashes.values()) for h in list(eval_hashes.values()) + [cal_hash]),
        })

    all_rows = rows + mf_rows
    ok = all(
        not r.get("intersection_training_evaluation", [])
        and not r.get("intersection_continuation_evaluation", [])
        and not r.get("intersection_training_calibration", [])
        and not r.get("intersection_continuation_calibration", [])
        and not r.get("intersection_evaluation_calibration", [])
        and r["tensor_hash_disjoint_from_consumed"]
        for r in all_rows
    )
    write_json(OUT / "evaluation_stream_disjointness.json", {"status": "PASS" if ok else "FAIL", "rows": rows, "multi_future_audit": mf_rows})
    if not ok:
        raise RuntimeError("held-out evaluation stream overlap detected")


def standard_spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    try:
        from scipy.stats import spearmanr

        val = spearmanr(xs, ys).statistic
        return float(val) if math.isfinite(float(val)) else None
    except Exception:
        return v2.spearman(xs, ys)


def auc_ap(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, Any]:
    y = [1 if bool(r[label]) else 0 for r in rows]
    s = [float(r[field]) for r in rows]
    out: dict[str, Any] = {}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["AUROC"] = float(roc_auc_score(y, s)) if len(set(y)) > 1 else None
        out["AP"] = float(average_precision_score(y, s)) if len(set(y)) > 1 else None
        out["library"] = "sklearn"
    except Exception as exc:
        out["AUROC"] = None
        out["AP"] = None
        out["library"] = f"unavailable: {exc}"
    return out


def residual_spearman(rows: list[dict[str, Any]], xfield: str, yfield: str, keys: list[str]) -> float | None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    xs: list[float] = []
    ys: list[float] = []
    for rs in groups.values():
        mx = float(np.mean([float(r[xfield]) for r in rs]))
        my = float(np.mean([float(r[yfield]) for r in rs]))
        for r in rs:
            xs.append(float(r[xfield]) - mx)
            ys.append(float(r[yfield]) - my)
    return standard_spearman(xs, ys)


def seed_boot(rows: list[dict[str, Any]], field: str, n: int = 400) -> dict[str, Any]:
    rng = random.Random(20260822)
    by = {s: [r for r in rows if int(r["seed"]) == s] for s in PRIMARY_SEEDS}
    vals = []
    for _ in range(n):
        sample = []
        for s in [rng.choice(PRIMARY_SEEDS) for _ in PRIMARY_SEEDS]:
            sample += by[s]
        val = standard_spearman([float(r[field]) for r in sample], [float(r["max_validation_loss_gap"]) for r in sample])
        if val is not None:
            vals.append(val)
    return {"bootstrap_seed": 20260822, "samples": len(vals), "mean": float(np.mean(vals)) if vals else None, "p05": float(np.percentile(vals, 5)) if vals else None, "p95": float(np.percentile(vals, 95)) if vals else None}


def final_metric_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"status": "COMPLETE", "metrics": {}}
    for name, field in METRICS.items():
        metric = {
            "field": field,
            "pooled_spearman_vs_max_heldout_eval_loss_gap": standard_spearman([float(r[field]) for r in rows], [float(r["max_validation_loss_gap"]) for r in rows]),
            "per_seed_spearman": {},
            "leave_one_family_out_spearman": {},
            "within_family_spearman": {},
            "checkpoint_stratified_spearman": {},
            "family_checkpoint_residualized_spearman": residual_spearman(rows, field, "max_validation_loss_gap", ["rewrite_family", "checkpoint_age"]),
            "seed_cluster_descriptive_bootstrap": seed_boot(rows, field),
            "binary_discrimination_h100_task_nonrecovery": auc_ap(rows, field, "task_nonrecovered_h100"),
        }
        for seed in PRIMARY_SEEDS:
            rs = [r for r in rows if int(r["seed"]) == seed]
            metric["per_seed_spearman"][str(seed)] = standard_spearman([float(r[field]) for r in rs], [float(r["max_validation_loss_gap"]) for r in rs])
        for fam in base.REWRITE_FAMILIES:
            rs = [r for r in rows if r["rewrite_family"] != fam]
            ws = [r for r in rows if r["rewrite_family"] == fam]
            metric["leave_one_family_out_spearman"][fam] = standard_spearman([float(r[field]) for r in rs], [float(r["max_validation_loss_gap"]) for r in rs])
            metric["within_family_spearman"][fam] = standard_spearman([float(r[field]) for r in ws], [float(r["max_validation_loss_gap"]) for r in ws])
        for age in PRIMARY_AGES:
            rs = [r for r in rows if int(r["checkpoint_age"]) == age]
            metric["checkpoint_stratified_spearman"][str(age)] = standard_spearman([float(r[field]) for r in rs], [float(r["max_validation_loss_gap"]) for r in rs])
        out["metrics"][name] = metric
    return out


def tau_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for frac in TAU_FRACTIONS:
        calc_rows = []
        for r in rows:
            curve = [0.0] + [float(v) for v in r["parameter_divergence_curve"][1:]]
            tau = frac * (float(r["theta_checkpoint_norm"]) + EPS)
            t = v2.first_crossing(curve, tau)
            nr = {**r, "T_tau_sensitivity": t}
            calc_rows.append(nr)
        tv = [r["T_tau_sensitivity"] for r in calc_rows if r["T_tau_sensitivity"] is not None]
        out.append({
            "tau_fraction": frac,
            "rows": len(calc_rows),
            "h20_crossing_count": sum(t is not None and int(t) <= H20 for t in [r["T_tau_sensitivity"] for r in calc_rows]),
            "h100_crossing_count": len(tv),
            "first_crossing_after_step20_count": sum(t is not None and int(t) > H20 for t in [r["T_tau_sensitivity"] for r in calc_rows]),
            "delayed_gt10_count": sum(t is not None and int(t) > 10 for t in [r["T_tau_sensitivity"] for r in calc_rows]),
            "latency_histogram": hist(calc_rows, "T_tau_sensitivity"),
            "per_rewrite_family": group_summary([{**r, "T_tau": r["T_tau_sensitivity"]} for r in calc_rows], "rewrite_family"),
            "per_seed": group_summary([{**r, "T_tau": r["T_tau_sensitivity"]} for r in calc_rows], "seed"),
            "per_checkpoint": group_summary([{**r, "T_tau": r["T_tau_sensitivity"]} for r in calc_rows], "checkpoint_age"),
            "median_T_tau_among_crossers": float(np.median(tv)) if tv else None,
            "max_T_tau_among_crossers": max(tv) if tv else None,
            "censored_count": sum(r["T_tau_sensitivity"] is None for r in calc_rows),
        })
    return out


def task_threshold_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sweep = []
    for horizon in [20, 100]:
        for eps in TASK_THRESHOLDS:
            sweep.append({
                "horizon": horizon,
                "threshold": eps,
                "rows": len(rows),
                "task_nonrecovered": sum(v2.recovery_step([float(x) for x in r["validation_loss_gap"][:horizon]], eps) is None for r in rows),
            })
    exact = [max([float(x) for x in r["validation_loss_gap"]] or [0.0]) for r in rows if r["rewrite_family"] == "exact_restore"]
    benign = [max([float(x) for x in r["validation_loss_gap"]] or [0.0]) for r in rows if r["rewrite_family"] == "benign_nonzero"]
    return {
        "status": "COMPLETE",
        "sweep": sweep,
        "exact_restore_max_gap_distribution": exact,
        "benign_nonzero_max_gap_distribution": benign,
        "maximum_benign_heldout_evaluation_loss_gap": max(benign) if benign else None,
        "benign_exceeds_0p05": any(v > 0.05 for v in benign),
    }


def loso(rows: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    out_rows = []
    for metric, field in METRICS.items():
        for heldout in PRIMARY_SEEDS:
            train = [r for r in rows if int(r["seed"]) != heldout]
            test = [r for r in rows if int(r["seed"]) == heldout]
            values = sorted({float(r[field]) for r in train})
            candidates = values + [(a + b) / 2 for a, b in zip(values, values[1:])]
            best = None
            for th in sorted(set(candidates)):
                y = [bool(r["task_nonrecovered_h100"]) for r in train]
                pred = [float(r[field]) >= th for r in train]
                tp = sum(p and yy for p, yy in zip(pred, y))
                fp = sum(p and not yy for p, yy in zip(pred, y))
                tn = sum((not p) and (not yy) for p, yy in zip(pred, y))
                fn = sum((not p) and yy for p, yy in zip(pred, y))
                sens = tp / (tp + fn) if tp + fn else 0.0
                spec = tn / (tn + fp) if tn + fp else 0.0
                bal = (sens + spec) / 2
                key = (bal, sens + spec - 1.0, -th)
                if best is None or key > best[0]:
                    best = (key, th)
            th = best[1] if best else 0.0
            y = [bool(r["task_nonrecovered_h100"]) for r in test]
            pred = [float(r[field]) >= th for r in test]
            tp = sum(p and yy for p, yy in zip(pred, y))
            fp = sum(p and not yy for p, yy in zip(pred, y))
            tn = sum((not p) and (not yy) for p, yy in zip(pred, y))
            fn = sum((not p) and yy for p, yy in zip(pred, y))
            sens = tp / (tp + fn) if tp + fn else 0.0
            spec = tn / (tn + fp) if tn + fp else 0.0
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = sens
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            disc = auc_ap(test, field, "task_nonrecovered_h100")
            out_rows.append({
                "metric": metric,
                "field": field,
                "heldout_seed": heldout,
                "threshold": th,
                "criterion": "maximum balanced accuracy / Youden J; ties prefer lower threshold",
                "sensitivity": sens,
                "specificity": spec,
                "balanced_accuracy": (sens + spec) / 2,
                "precision": precision,
                "recall": recall,
                "F1": f1,
                "heldout_AUROC": disc["AUROC"],
                "heldout_AP": disc["AP"],
                "note": "leave-one-training-seed-out internal robustness, not an external independent test set",
            })
    write_csv(OUT / "leave_one_training_seed_out_internal_robustness.csv", out_rows)
    result = {"status": "COMPLETE", "rows": out_rows, "note": "This is leave-one-training-seed-out internal robustness, not an external independent test set."}
    write_json(OUT / "leave_one_training_seed_out_internal_robustness.json", result)
    return result


def final_stats() -> None:
    primary = read_json(OUT / "primary" / "primary_rows.json")
    for r in primary:
        r["parameter_divergence_curve"] = [0.0] + [float(s["parameter_divergence"]) for s in read_csv_rows_for_row(OUT / "primary" / "primary_steps.csv", r["row_id"])]
    summary = aggregate(primary, "primary", ["seed", "checkpoint_age"])
    old_v3 = read_json(V3 / "primary" / "primary_h100_corrected_summary.json")
    reset = {r["row_id"]: r for r in primary if r["rewrite_family"] == "optimizer_reset"}
    old_reset = old_v3["old_vs_corrected_optimizer_reset_T_tau"]
    summary["delayed_T_tau_matrix"] = {rid: {"v4_corrected": old_reset[rid]["corrected"], "v5_heldout": reset[rid]["T_tau"]} for rid in sorted(reset)}
    summary["final_continuous_metric_statistics"] = final_metric_statistics(primary)
    summary["task_threshold_sensitivity"] = task_threshold_sensitivity(primary)
    tau = tau_sensitivity(primary)
    summary["tau_sensitivity"] = tau
    exact_bad = [r["row_id"] for r in primary if r["rewrite_family"] == "exact_restore" and (float(r["max_parameter_divergence"]) != 0.0 or float(r["max_validation_loss_gap"]) != 0.0)]
    summary["exact_restore_zero_failures"] = exact_bad
    write_json(OUT / "primary" / "primary_final_summary.json", summary)
    write_csv(OUT / "tau_sensitivity_final.csv", tau)
    write_json(OUT / "tau_sensitivity_final.json", tau)
    write_md(OUT / "tau_sensitivity_final.md", "\n".join(["# Tau Sensitivity Final", "", "Robustness/sensitivity analysis; 0.10 remains the primary operating point."] + [f"- tau={r['tau_fraction']}: H20={r['h20_crossing_count']}, H100={r['h100_crossing_count']}, censored={r['censored_count']}" for r in tau]))
    write_json(OUT / "task_threshold_sensitivity_final.json", summary["task_threshold_sensitivity"])
    write_csv(OUT / "task_threshold_sensitivity_final.csv", summary["task_threshold_sensitivity"]["sweep"])
    write_json(OUT / "final_standard_library_statistics.json", summary["final_continuous_metric_statistics"])
    loso(primary, summary["final_continuous_metric_statistics"])

    controlled = read_json(OUT / "controlled_width" / "controlled_rows.json")
    long = read_json(OUT / "long_training" / "long_rows.json")
    write_json(OUT / "controlled_width" / "controlled_final_summary.json", aggregate(controlled, "controlled_width", ["model_config", "seed", "checkpoint_age"]))
    write_json(OUT / "long_training" / "long_final_summary.json", aggregate(long, "long_training", ["seed", "checkpoint_age"]))
    mf = read_json(V3 / "multi_future" / "multi_future_isolated_summary.json")
    dis = read_json(OUT / "evaluation_stream_disjointness.json")
    write_json(OUT / "multi_future_audit_result.json", {"status": "PASS", "rerun": False, "reason": "existing V4 evaluation/calibration/continuation indices are disjoint", "v4_rows_preserved": mf.get("rows"), "v4_summary": mf, "audit": dis["multi_future_audit"]})


def read_csv_rows_for_row(path: Path, row_id: str) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["row_id"] == row_id and int(r["continuation_step"]) > 0:
                rows.append(r)
    rows.sort(key=lambda r: int(r["continuation_step"]))
    return rows


def checkpoint_hashes() -> list[dict[str, Any]]:
    paths = []
    paths += [p_primary(s, a) for s in PRIMARY_SEEDS for a in PRIMARY_AGES]
    paths += [stale_primary(s, a) for s in PRIMARY_SEEDS for a in PRIMARY_AGES if stale_primary(s, a).exists()]
    paths += [p_control(c["name"], s, a) for c in v2.SCALE_CONFIGS for s in CONTROLLED_SEEDS for a in CONTROLLED_AGES]
    paths += [stale_control(c["name"], s, a) for c in v2.SCALE_CONFIGS for s in CONTROLLED_SEEDS for a in CONTROLLED_AGES if stale_control(c["name"], s, a).exists()]
    paths += [p_long(s, a) for s in LONG_SEEDS for a in LONG_AGES]
    paths += [stale_long(s, a) for s in LONG_SEEDS for a in LONG_AGES if stale_long(s, a).exists()]
    out = []
    for p in sorted(set(paths), key=str):
        out.append({"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return out


def provenance() -> None:
    scripts = [ROOT / "scripts" / "run_reviewer2026_v5_final_eval.py", ROOT / "scripts" / "verify_reviewer2026_v5_final_eval.py", ROOT / "scripts" / "run_reviewer2026_v3_lr_isolation.py"]
    write_json(OUT / "PROVENANCE.json", {
        "starting_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "starting_branch": run_cmd(["git", "branch", "--show-current"]),
        "current_git_status": run_cmd(["git", "status", "--short"]),
        "source_experiment_commit_v4": SOURCE_EXPERIMENT_COMMIT,
        "v4_transport_commit": V4_TRANSPORT_COMMIT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "script_hashes": [{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for p in scripts if p.exists()],
        "checkpoint_hashes": checkpoint_hashes(),
    })


def manifest() -> None:
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "MANIFEST.json", {"status": "COMPLETE", "files": files})


def all_steps() -> None:
    freeze_protocol()
    provenance()
    protocol_disjointness()
    run_campaign("primary")
    run_campaign("controlled")
    run_campaign("long")
    final_stats()
    manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="all")
    args = parser.parse_args()
    steps = {
        "protocol": lambda: (freeze_protocol(), provenance(), protocol_disjointness()),
        "primary": lambda: run_campaign("primary"),
        "controlled": lambda: run_campaign("controlled"),
        "long": lambda: run_campaign("long"),
        "stats": final_stats,
        "manifest": manifest,
        "all": all_steps,
    }
    for name in [s.strip() for s in args.steps.split(",") if s.strip()]:
        if name not in steps:
            raise SystemExit(f"unknown step {name}")
        steps[name]()


if __name__ == "__main__":
    main()



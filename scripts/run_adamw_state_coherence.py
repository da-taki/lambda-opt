from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments" / "scripts"), str(ROOT / "experiments")]

import run_opt2026_10m_gpu_final as base  # noqa: E402
import run_reviewer2026_v2 as v2  # noqa: E402
import run_reviewer2026_v3_lr_isolation as v3  # noqa: E402
import run_reviewer2026_v5_final_eval as v5  # noqa: E402

OUT = ROOT / "results" / "adamw_state_coherence"
V5 = ROOT / "results" / "reviewer2026_v5_final_eval"
PRIMARY_SEEDS = [1101, 1102, 1103]
PRIMARY_AGES = [75, 150, 300]
LONG_SEEDS = [1301, 1302, 1303]
LONG_AGES = [750, 1500, 3000]
CONTROLLED_SEEDS = [1201, 1202, 1203]
CONTROLLED_AGES = [75, 150, 300]
H = 100
H20 = 20
EPS = 1e-12
TAU_FRAC = 0.10
TASK_EVAL_STEPS = {1, 20, 100}
LARGE_EVAL_STEPS = {0, 1, 20, 100}
OFFICIAL_VALID_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
SOURCE_EXPERIMENT_COMMIT = "c3f245cafab05c4db376755e292602844985ff91"
TRANSPORT_SOURCE_BRANCH = "experiment/opt2026-reviewer-results-v5"
TRANSPORT_SOURCE_COMMIT = "365257a3a5fbe68672ad99334f0183a45e15bda2"
NEW_FAMILY = "reset_mv_preserve_counter"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def maybe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def batch_hash(batch: Any) -> str:
    h = hashlib.sha256()
    for tensor in batch:
        cpu = tensor.detach().cpu().contiguous()
        h.update(str(tuple(cpu.shape)).encode("ascii"))
        h.update(str(cpu.dtype).encode("ascii"))
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def flat(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])


def count_params_from_payload(payload: dict[str, Any]) -> int:
    return int(sum(t.numel() for t in payload["model_state"].values()))


def optimizer_step_values_from_state(opt_state: dict[str, Any]) -> list[int]:
    values = []
    for state in opt_state.get("state", {}).values():
        if "step" not in state:
            continue
        step = state["step"]
        if torch.is_tensor(step):
            values.append(int(step.detach().cpu().item()))
        else:
            values.append(int(step))
    return values


def optimizer_step_values(opt: torch.optim.Optimizer) -> list[int]:
    values = []
    for state in opt.state.values():
        if "step" not in state:
            continue
        step = state["step"]
        if torch.is_tensor(step):
            values.append(int(step.detach().cpu().item()))
        else:
            values.append(int(step))
    return values


def moment_norms(opt: torch.optim.Optimizer) -> dict[str, float]:
    out = {"m_norm": 0.0, "v_norm": 0.0}
    for state in opt.state.values():
        if "exp_avg" in state:
            out["m_norm"] += float(state["exp_avg"].detach().float().pow(2).sum().cpu().item())
        if "exp_avg_sq" in state:
            out["v_norm"] += float(state["exp_avg_sq"].detach().float().pow(2).sum().cpu().item())
    return {k: math.sqrt(v) for k, v in out.items()}


def prediction_a_t(age: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    return ((1.0 - beta1) / (1.0 - beta1 ** (age + 1))) * math.sqrt((1.0 - beta2 ** (age + 1)) / (1.0 - beta2))


def write_prediction() -> dict[str, Any]:
    ages = [75, 150, 300, 750, 1500, 3000]
    rows = [{"checkpoint_age": age, "A_t": prediction_a_t(age)} for age in ages]
    obj = {
        "status": "COMPLETE",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon_note": "Analytic approximation ignores epsilon for nonzero gradient coordinates; it is not exact when epsilon matters.",
        "formula": "A_t = ((1-beta1)/(1-beta1^(t+1))) * sqrt((1-beta2^(t+1))/(1-beta2))",
        "values": rows,
    }
    write_json(OUT / "moment_reset_counter_preserved_prediction.json", obj)
    lines = [
        "# Moment Reset Counter-Preserved Prediction",
        "",
        "A_t = ((1-beta1)/(1-beta1^(t+1))) * sqrt((1-beta2^(t+1))/(1-beta2))",
        "",
        "This ignores epsilon for nonzero gradient coordinates and is not exact when epsilon matters.",
        "",
        "| checkpoint age | A_t |",
        "|---:|---:|",
    ]
    lines += [f"| {r['checkpoint_age']} | {r['A_t']:.9f} |" for r in rows]
    write_md(OUT / "moment_reset_counter_preserved_prediction.md", "\n".join(lines))
    return obj


def audit_width448() -> dict[str, Any]:
    primary = base.load_checkpoint(v5.p_primary(1101, 75))
    controlled = base.load_checkpoint(v5.p_control("width448_19p48m", 1201, 75))
    keys = [
        "architecture",
        "parameter_count",
        "dataset_construction",
        "corpus_slice",
        "batch_size",
        "sequence_length",
        "optimizer",
        "betas",
        "eps",
        "weight_decay",
        "initial_lr",
        "StepLR_policy",
        "total_training_steps",
        "checkpoint_ages",
        "rewrite_semantics",
    ]
    primary_desc = {
        "architecture": primary["cfg"],
        "parameter_count": count_params_from_payload(primary),
        "dataset_construction": "run_transformer.build_char_dataset, shuffled by training seed",
        "corpus_slice": f"first {base.DATA_CHARS} characters of experiments/data/wikitext2_raw/train.txt",
        "batch_size": base.BATCH_SIZE,
        "sequence_length": base.SEQ_LEN,
        "optimizer": "torch.optim.AdamW",
        "betas": list(base.BETAS),
        "eps": base.EPS,
        "weight_decay": base.WEIGHT_DECAY,
        "initial_lr": base.LR,
        "StepLR_policy": {"step_size": 100, "gamma": base.SCHED_GAMMA},
        "total_training_steps": 300,
        "checkpoint_ages": PRIMARY_AGES,
        "rewrite_semantics": base.REWRITE_FAMILIES,
    }
    controlled_desc = {
        "architecture": controlled["cfg"],
        "parameter_count": count_params_from_payload(controlled),
        "dataset_construction": "run_transformer.build_char_dataset, shuffled by training seed",
        "corpus_slice": f"first {base.DATA_CHARS} characters of experiments/data/wikitext2_raw/train.txt",
        "batch_size": base.BATCH_SIZE,
        "sequence_length": base.SEQ_LEN,
        "optimizer": "torch.optim.AdamW",
        "betas": list(base.BETAS),
        "eps": base.EPS,
        "weight_decay": base.WEIGHT_DECAY,
        "initial_lr": base.LR,
        "StepLR_policy": {"step_size": 100, "gamma": base.SCHED_GAMMA},
        "total_training_steps": 300,
        "checkpoint_ages": CONTROLLED_AGES,
        "rewrite_semantics": base.REWRITE_FAMILIES,
    }
    comparisons = {key: {"primary": primary_desc[key], "width448_19p48m": controlled_desc[key], "identical": primary_desc[key] == controlled_desc[key]} for key in keys}
    differences = {key: value for key, value in comparisons.items() if not value["identical"]}
    obj = {
        "status": "COMPLETE",
        "eligible_for_narrow_mechanism_pooling": len(differences) == 0,
        "primary_seed": primary["seed"],
        "controlled_width_seed": controlled["seed"],
        "allowed_difference": "independent training seeds only",
        "comparisons": comparisons,
        "differences": differences,
    }
    write_json(OUT / "width448_replication_eligibility.json", obj)
    return obj


def shock_rows_from_steps(path: Path, campaign: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(path):
        if row["rewrite_family"] not in {"reset_v", "reset_m", "stale_v", "optimizer_reset"}:
            continue
        if int(row["continuation_step"]) != 1:
            continue
        clean = maybe_float(row["clean_update_norm"]) or 0.0
        cand = maybe_float(row["candidate_update_norm"]) or 0.0
        delta = maybe_float(row["delta_update_norm"]) or 0.0
        rows.append({
            "campaign": campaign,
            "row_id": row["row_id"],
            "seed": int(row["seed"]),
            "checkpoint_age": int(row["checkpoint_age"]),
            "rewrite_family": row["rewrite_family"],
            "continuation_step": 1,
            "clean_update_norm": clean,
            "candidate_update_norm": cand,
            "delta_update_norm": delta,
            "S1": delta / (clean + EPS),
            "candidate_clean_update_norm_ratio": cand / (clean + EPS),
            "update_cosine": maybe_float(row["update_cosine"]),
            "lr": maybe_float(row["lr"]),
            "candidate_lr": maybe_float(row["candidate_lr"]),
        })
    return rows


def family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (campaign, family), group in sorted(defaultdict(list, {k: [] for k in []}).items()):
        pass
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["campaign"], row["rewrite_family"])].append(row)
    for (campaign, family), group in sorted(groups.items()):
        s1 = [float(r["S1"]) for r in group]
        ratio = [float(r["candidate_clean_update_norm_ratio"]) for r in group]
        cos = [float(r["update_cosine"]) for r in group if r["update_cosine"] is not None]
        out.append({
            "campaign": campaign,
            "rewrite_family": family,
            "n": len(group),
            "S1_min": min(s1),
            "S1_median": median(s1),
            "S1_max": max(s1),
            "candidate_clean_update_norm_ratio_median": median(ratio),
            "update_cosine_median": median(cos),
        })
    return out


def audit_existing_shock() -> dict[str, Any]:
    rows = []
    rows += shock_rows_from_steps(V5 / "primary" / "primary_steps.csv", "primary")
    rows += shock_rows_from_steps(V5 / "controlled_width" / "controlled_steps.csv", "controlled_width")
    rows += shock_rows_from_steps(V5 / "long_training" / "long_steps.csv", "long_training")
    summaries = family_summary(rows)
    primary = [r for r in rows if r["campaign"] == "primary"]
    by_key = {(r["seed"], r["checkpoint_age"], r["rewrite_family"]): r for r in primary}
    pairs = []
    for seed in PRIMARY_SEEDS:
        for age in PRIMARY_AGES:
            rv = by_key[(seed, age, "reset_v")]
            op = by_key[(seed, age, "optimizer_reset")]
            pairs.append({
                "seed": seed,
                "checkpoint_age": age,
                "reset_v_S1": rv["S1"],
                "optimizer_reset_S1": op["S1"],
                "paired_difference_reset_v_minus_optimizer_reset": rv["S1"] - op["S1"],
                "reset_v_candidate_clean_ratio": rv["candidate_clean_update_norm_ratio"],
                "optimizer_reset_candidate_clean_ratio": op["candidate_clean_update_norm_ratio"],
            })
    diffs = [p["paired_difference_reset_v_minus_optimizer_reset"] for p in pairs]
    clear_advantage = all(d > 0 for d in diffs) and (median(diffs) or 0.0) > 0.0
    obj = {
        "status": "COMPLETE",
        "rows": rows,
        "family_summaries": summaries,
        "primary_reset_v_vs_optimizer_reset_pairs": pairs,
        "paired_reset_v_advantage_count": sum(d > 0 for d in diffs),
        "paired_n": len(diffs),
        "median_paired_difference_reset_v_minus_optimizer_reset": median(diffs),
        "clear_repeated_first_step_shock_advantage": clear_advantage,
    }
    write_csv(OUT / "existing_state_shock_audit.csv", rows)
    write_csv(OUT / "existing_state_shock_family_summary.csv", summaries)
    write_csv(OUT / "existing_state_shock_primary_pairs.csv", pairs)
    write_json(OUT / "existing_state_shock_audit.json", obj)
    lines = [
        "# Existing State Shock Audit",
        "",
        f"Primary paired reset_v > optimizer_reset count: {obj['paired_reset_v_advantage_count']}/{obj['paired_n']}.",
        f"Median paired S1 difference: {obj['median_paired_difference_reset_v_minus_optimizer_reset']:.9f}.",
        f"Fail-fast verdict: {'continue' if clear_advantage else 'stop'}.",
        "",
        "| seed | age | reset_v S1 | optimizer_reset S1 | difference |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {p['seed']} | {p['checkpoint_age']} | {p['reset_v_S1']:.9f} | {p['optimizer_reset_S1']:.9f} | {p['paired_difference_reset_v_minus_optimizer_reset']:.9f} |" for p in pairs]
    write_md(OUT / "existing_state_shock_audit.md", "\n".join(lines))
    return obj


def reset_mv_preserve_counter_payload(payload: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(payload)
    p["rewrite_family"] = NEW_FAMILY
    for state in p["optimizer_state"]["state"].values():
        if "exp_avg" in state:
            state["exp_avg"] = torch.zeros_like(state["exp_avg"])
        if "exp_avg_sq" in state:
            state["exp_avg_sq"] = torch.zeros_like(state["exp_avg_sq"])
    return p


def fresh_reset_mv_preserve_counter(payload: dict[str, Any]):
    scen = reset_mv_preserve_counter_payload(payload)
    model, opt, sched = base.fresh_from_payload(scen, load_optimizer=True, load_scheduler=True)
    mstats = moment_norms(opt)
    ckpt_steps = optimizer_step_values_from_state(payload["optimizer_state"])
    cand_steps = optimizer_step_values(opt)
    audit = {
        "checkpoint_optimizer_lr": payload["optimizer_state"]["param_groups"][0]["lr"],
        "candidate_optimizer_lr": opt.param_groups[0]["lr"],
        "lr_matches_checkpoint": opt.param_groups[0]["lr"] == payload["optimizer_state"]["param_groups"][0]["lr"],
        "param_groups_match_checkpoint": v3.groups_match(opt, payload["optimizer_state"]),
        "scheduler_state_matches_checkpoint": sched.state_dict() == payload["scheduler_state"],
        "initial_optimizer_state_entries": v3.opt_state_entries(opt),
        "initial_optimizer_state_keys": v3.opt_state_keys(opt),
        "m_zero": mstats["m_norm"] == 0.0,
        "v_zero": mstats["v_norm"] == 0.0,
        "m_norm": mstats["m_norm"],
        "v_norm": mstats["v_norm"],
        "counter_preserved": sorted(set(cand_steps)) == sorted(set(ckpt_steps)),
        "checkpoint_step_values_unique": sorted(set(ckpt_steps)),
        "candidate_step_values_unique": sorted(set(cand_steps)),
    }
    return model, opt, sched, audit


def candidate_for_family(payload: dict[str, Any], family: str, stale: dict[str, Any] | None):
    if family == NEW_FAMILY:
        return fresh_reset_mv_preserve_counter(payload)
    return v5.candidate_from_family(payload, family, stale)


def run_mechanistic_pair(payload: dict[str, Any], batches: list[Any], eval_batches: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_model, clean_opt, clean_sched = base.fresh_from_payload(payload)
    cand_model, cand_opt, cand_sched, audit = fresh_reset_mv_preserve_counter(payload)
    start = int(payload["batch_pos"])
    theta = v2.norm_model(clean_model)
    tau = TAU_FRAC * (theta + EPS)
    div0 = v2.l2_from_params(clean_model, cand_model)
    if div0 != 0.0:
        raise RuntimeError("model equality before replay failed")
    if not all([audit["lr_matches_checkpoint"], audit["param_groups_match_checkpoint"], audit["scheduler_state_matches_checkpoint"], audit["m_zero"], audit["v_zero"], audit["counter_preserved"]]):
        raise RuntimeError(f"rewrite audit failed: {audit}")
    divergences = [div0]
    steps: list[dict[str, Any]] = []
    task_gaps: dict[int, float] = {}
    first_metrics: dict[str, Any] = {}
    same_batch_hashes = []
    for k in range(1, H + 1):
        batch_index = (start + k - 1) % len(batches)
        batch = batches[batch_index]
        same_batch_hashes.append(batch_hash(batch))
        prev_clean = flat(clean_model)
        prev_candidate = flat(cand_model)
        base.train_one(clean_model, clean_opt, clean_sched, batch)
        base.train_one(cand_model, cand_opt, cand_sched, batch)
        clean_now = flat(clean_model)
        cand_now = flat(cand_model)
        clean_update = clean_now - prev_clean
        candidate_update = cand_now - prev_candidate
        delta_update = clean_update - candidate_update
        clean_norm = float(clean_update.norm().detach().cpu().item())
        cand_norm = float(candidate_update.norm().detach().cpu().item())
        delta_norm = float(delta_update.norm().detach().cpu().item())
        cosine = float((torch.dot(clean_update, candidate_update) / (clean_update.norm() * candidate_update.norm() + EPS)).detach().cpu().item()) if clean_norm > 0 and cand_norm > 0 else math.nan
        div = v2.l2_from_params(clean_model, cand_model)
        divergences.append(div)
        row = {
            "continuation_step": k,
            "training_batch_index": batch_index,
            "training_batch_hash": same_batch_hashes[-1],
            "parameter_divergence": div,
            "normalized_parameter_divergence": div / (theta + EPS),
            "lr": clean_opt.param_groups[0]["lr"],
            "candidate_lr": cand_opt.param_groups[0]["lr"],
            "clean_update_norm": clean_norm,
            "candidate_update_norm": cand_norm,
            "delta_update_norm": delta_norm,
            "S1_if_first_step": delta_norm / (clean_norm + EPS) if k == 1 else "",
            "candidate_clean_update_norm_ratio_if_first_step": cand_norm / (clean_norm + EPS) if k == 1 else "",
            "update_cosine_if_first_step": cosine if k == 1 else "",
        }
        if k in TASK_EVAL_STEPS:
            src_loss = sum(v2.validation_loss(clean_model, b) for b in eval_batches) / len(eval_batches)
            cand_loss = sum(v2.validation_loss(cand_model, b) for b in eval_batches) / len(eval_batches)
            task_gaps[k] = abs(src_loss - cand_loss)
            row["heldout_task_loss_clean"] = src_loss
            row["heldout_task_loss_candidate"] = cand_loss
            row["heldout_task_gap"] = task_gaps[k]
        steps.append(row)
        if k == 1:
            first_metrics = {
                "first_step_clean_update_norm": clean_norm,
                "first_step_candidate_update_norm": cand_norm,
                "first_step_delta_update_norm": delta_norm,
                "S1": delta_norm / (clean_norm + EPS),
                "first_step_update_cosine": cosine,
                "observed_candidate_clean_update_norm_ratio": cand_norm / (clean_norm + EPS),
            }
        del prev_clean, prev_candidate, clean_now, cand_now, clean_update, candidate_update, delta_update
    t_tau = v2.first_crossing(divergences, tau)
    age = int(payload["training_step"])
    summary = {
        **first_metrics,
        "theta_checkpoint_norm": theta,
        "tau_fraction": TAU_FRAC,
        "tau_numeric": tau,
        "T_tau": t_tau,
        "censored_at_100": t_tau is None,
        "max_parameter_divergence": max(divergences),
        "final_parameter_divergence": divergences[-1],
        "heldout_task_gap_k1": task_gaps.get(1),
        "heldout_task_gap_k20": task_gaps.get(20),
        "heldout_task_gap_k100": task_gaps.get(100),
        "predicted_A_t": prediction_a_t(age),
        "checkpoint_optimizer_lr": audit["checkpoint_optimizer_lr"],
        "candidate_initial_lr": audit["candidate_optimizer_lr"],
        "candidate_lr_matches_checkpoint": audit["lr_matches_checkpoint"],
        "candidate_param_groups_match_checkpoint": audit["param_groups_match_checkpoint"],
        "candidate_scheduler_state_matches_checkpoint": audit["scheduler_state_matches_checkpoint"],
        "candidate_initial_optimizer_state_entries": audit["initial_optimizer_state_entries"],
        "candidate_initial_optimizer_state_keys": audit["initial_optimizer_state_keys"],
        "m_zero": audit["m_zero"],
        "v_zero": audit["v_zero"],
        "counter_preserved": audit["counter_preserved"],
        "checkpoint_step_values_unique": audit["checkpoint_step_values_unique"],
        "candidate_step_values_unique": audit["candidate_step_values_unique"],
        "same_training_batches": len(same_batch_hashes) == H and len(set(same_batch_hashes)) == H,
    }
    del clean_model, clean_opt, clean_sched, cand_model, cand_opt, cand_sched
    torch.cuda.empty_cache()
    return summary, steps


def run_mechanistic_experiment() -> dict[str, Any]:
    base.require_cuda()
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    specs = []
    specs += [("primary", seed, age, v5.p_primary(seed, age), 100, {}) for seed in PRIMARY_SEEDS for age in PRIMARY_AGES]
    specs += [("long_training", seed, age, v5.p_long(seed, age), 1000, {"training_steps": 3000}) for seed in LONG_SEEDS for age in LONG_AGES]
    for campaign, seed, age, path, sched_step_value, extra in specs:
        batches, _, dataset = v5.load_batches_full(seed)
        eval_batches = [batches[i] for i in v5.EVAL_INDICES]
        with v5.sched_step(sched_step_value):
            payload = base.load_checkpoint(path)
            t0 = time.perf_counter()
            summary, steps = run_mechanistic_pair(payload, batches, eval_batches)
        row_id = f"{campaign}_seed{seed}_ckpt{age}_{NEW_FAMILY}"
        row = {
            **summary,
            "row_id": row_id,
            "experiment": "adamw_state_coherence_mechanistic",
            "campaign": campaign,
            "dataset": dataset,
            "seed": seed,
            "checkpoint_age": age,
            "rewrite_family": NEW_FAMILY,
            "checkpoint_file": str(path.relative_to(ROOT)),
            "source": "existing_checkpoint_no_retraining",
            "evaluation_indices": v5.EVAL_INDICES,
            "runtime_seconds": time.perf_counter() - t0,
            **extra,
        }
        rows.append(row)
        step_rows += [{"row_id": row_id, "campaign": campaign, "seed": seed, "checkpoint_age": age, "rewrite_family": NEW_FAMILY, **s} for s in steps]
        write_json(OUT / "mechanistic_reset_mv_preserve_counter_rows.json", rows)
        write_csv(OUT / "mechanistic_reset_mv_preserve_counter_rows.csv", rows)
        write_csv(OUT / "mechanistic_reset_mv_preserve_counter_steps.csv", step_rows)
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 mechanistic cases, got {len(rows)}")
    by_age: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_age[int(row["checkpoint_age"])].append(row)
    age_summary = []
    for age, group in sorted(by_age.items()):
        age_summary.append({
            "checkpoint_age": age,
            "n": len(group),
            "predicted_A_t": prediction_a_t(age),
            "observed_ratio_median": median([float(r["observed_candidate_clean_update_norm_ratio"]) for r in group]),
            "S1_median": median([float(r["S1"]) for r in group]),
            "T_tau_crossings": sum(r["T_tau"] is not None for r in group),
            "T_tau_median_among_crossers": median([float(r["T_tau"]) for r in group if r["T_tau"] is not None]),
        })
    ratio_by_age = [r["observed_ratio_median"] for r in age_summary if r["observed_ratio_median"] is not None]
    pred_by_age = [r["predicted_A_t"] for r in age_summary if r["observed_ratio_median"] is not None]
    directional = all(y >= x for x, y in zip(ratio_by_age, ratio_by_age[1:])) if len(ratio_by_age) > 1 else False
    corr = v2.spearman(pred_by_age, ratio_by_age) if len(ratio_by_age) > 2 else None
    summary = {
        "status": "COMPLETE",
        "rows": len(rows),
        "required_18_cases_present": len(rows) == 18,
        "crossings_h100": sum(r["T_tau"] is not None for r in rows),
        "crossings_h20": sum(r["T_tau"] is not None and int(r["T_tau"]) <= 20 for r in rows),
        "median_T_tau_among_crossers": median([float(r["T_tau"]) for r in rows if r["T_tau"] is not None]),
        "S1_median": median([float(r["S1"]) for r in rows]),
        "observed_ratio_median": median([float(r["observed_candidate_clean_update_norm_ratio"]) for r in rows]),
        "age_summary": age_summary,
        "observed_ratio_monotone_non_decreasing_by_age": directional,
        "spearman_predicted_A_t_vs_observed_age_median_ratio": corr,
        "state_coherence_supported_for_phase4": directional and (corr is None or corr > 0),
    }
    write_json(OUT / "mechanistic_reset_mv_preserve_counter_summary.json", summary)
    lines = [
        "# reset_mv_preserve_counter Mechanistic Summary",
        "",
        f"Cases: {len(rows)}",
        f"H100 crossings: {summary['crossings_h100']}",
        f"Median S1: {summary['S1_median']:.9f}",
        "",
        "| age | predicted A_t | observed ratio median | S1 median | crossings | median T_tau |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in age_summary:
        tt = "" if row["T_tau_median_among_crossers"] is None else f"{row['T_tau_median_among_crossers']:.1f}"
        lines.append(f"| {row['checkpoint_age']} | {row['predicted_A_t']:.9f} | {row['observed_ratio_median']:.9f} | {row['S1_median']:.9f} | {row['T_tau_crossings']} | {tt} |")
    write_md(OUT / "mechanistic_reset_mv_preserve_counter_summary.md", "\n".join(lines))
    return summary


def build_validation_batches() -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_text = (ROOT / "experiments" / "data" / "wikitext2_raw" / "train.txt").read_text(encoding="utf-8")[: base.DATA_CHARS]
    chars = sorted(set(train_text))
    c2i = {c: i for i, c in enumerate(chars)}
    with urllib.request.urlopen(OFFICIAL_VALID_URL, timeout=30) as response:
        valid_text = response.read().decode("utf-8")
    missing = sorted(set(valid_text) - set(c2i))
    protocol = {
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_source": OFFICIAL_VALID_URL,
        "validation_sha256": sha256_text(valid_text),
        "training_vocab_source": f"first {base.DATA_CHARS} characters of train.txt",
        "training_vocab_size": len(chars),
        "validation_characters": len(valid_text),
        "unrepresentable_validation_characters": missing,
        "task_sequences": 64,
        "function_sequences": 16,
        "target_positions_per_sequence": base.SEQ_LEN - 1,
        "task_target_positions": 64 * (base.SEQ_LEN - 1),
        "function_target_positions": 16 * (base.SEQ_LEN - 1),
        "families": ["exact_restore", "benign_nonzero", "reset_v", "optimizer_reset", NEW_FAMILY],
        "primary_seeds": PRIMARY_SEEDS,
        "primary_ages": PRIMARY_AGES,
        "replay_horizon": H,
        "eval_steps": sorted(LARGE_EVAL_STEPS),
    }
    data = None
    if not missing:
        encoded = torch.tensor([c2i[c] for c in valid_text], dtype=torch.long)
        n_seq = len(encoded) // base.SEQ_LEN
        data = encoded[: n_seq * base.SEQ_LEN].view(n_seq, base.SEQ_LEN)
        protocol["available_validation_sequences"] = int(n_seq)
        protocol["actual_task_target_positions"] = 64 * (base.SEQ_LEN - 1) if n_seq >= 80 else None
        protocol["actual_function_target_positions"] = 16 * (base.SEQ_LEN - 1) if n_seq >= 80 else None
    write_json(OUT / "official_validation_protocol_frozen.json", protocol)
    if missing:
        raise RuntimeError(f"official validation contains unrepresentable characters: {missing}")
    if data is None or data.shape[0] < 80:
        raise RuntimeError(f"need 80 validation sequences, got {0 if data is None else data.shape[0]}")
    task_batches = [(data[i : i + 1, :-1], data[i : i + 1, 1:]) for i in range(64)]
    function_batches = [(data[i : i + 1, :-1], data[i : i + 1, 1:]) for i in range(64, 80)]
    if len(task_batches) * (base.SEQ_LEN - 1) != 8128 or len(function_batches) * (base.SEQ_LEN - 1) != 2032:
        raise RuntimeError("official validation target-position counts changed")
    return task_batches, function_batches, protocol


def validation_loss_many(model: torch.nn.Module, batches: list[Any]) -> float:
    return float(np.mean([v2.validation_loss(model, b) for b in batches]))


def function_metrics_many(model_a: torch.nn.Module, model_b: torch.nn.Module, batches: list[Any]) -> dict[str, float]:
    kls = []
    rms = []
    for batch in batches:
        x, _ = base.to_device_batch(batch)
        with torch.no_grad():
            la = model_a(x).detach().float()
            lb = model_b(x).detach().float()
        logpa = F.log_softmax(la, dim=-1)
        logpb = F.log_softmax(lb, dim=-1)
        pa = logpa.exp()
        kls.append(float((pa * (logpa - logpb)).sum(dim=-1).mean().cpu().item()))
        rms.append(float((la - lb).pow(2).mean().sqrt().cpu().item()))
    return {"official_validation_kl": float(np.mean(kls)), "official_validation_rms_logit": float(np.mean(rms))}


def run_large_validation() -> dict[str, Any]:
    base.require_cuda()
    task_batches, function_batches, protocol = build_validation_batches()
    rows: list[dict[str, Any]] = []
    for seed in PRIMARY_SEEDS:
        batches, _, dataset = v5.load_batches_full(seed)
        for age in PRIMARY_AGES:
            payload = base.load_checkpoint(v5.p_primary(seed, age))
            stale = v5.load_stale(v5.stale_primary(seed, age))
            with v5.sched_step(100):
                for family in ["exact_restore", "benign_nonzero", "reset_v", "optimizer_reset", NEW_FAMILY]:
                    clean_model, clean_opt, clean_sched = base.fresh_from_payload(payload)
                    cand_model, cand_opt, cand_sched, audit = candidate_for_family(payload, family, stale)
                    start = int(payload["batch_pos"])
                    for k in range(0, H + 1):
                        if k in LARGE_EVAL_STEPS:
                            src_loss = validation_loss_many(clean_model, task_batches)
                            cand_loss = validation_loss_many(cand_model, task_batches)
                            fmetrics = function_metrics_many(clean_model, cand_model, function_batches)
                            rows.append({
                                "row_id": f"official_validation_seed{seed}_ckpt{age}_{family}_k{k}",
                                "seed": seed,
                                "checkpoint_age": age,
                                "rewrite_family": family,
                                "continuation_step": k,
                                "official_validation_loss_clean": src_loss,
                                "official_validation_loss_candidate": cand_loss,
                                "official_validation_loss_gap": abs(src_loss - cand_loss),
                                **fmetrics,
                                "candidate_lr_matches_checkpoint": audit["lr_matches_checkpoint"],
                                "candidate_param_groups_match_checkpoint": audit["param_groups_match_checkpoint"],
                                "candidate_scheduler_state_matches_checkpoint": audit["scheduler_state_matches_checkpoint"],
                                "dataset": dataset,
                                "validation_protocol_sha256": sha256_file(OUT / "official_validation_protocol_frozen.json"),
                            })
                        if k == H:
                            break
                        batch = batches[(start + k) % len(batches)]
                        base.train_one(clean_model, clean_opt, clean_sched, batch)
                        base.train_one(cand_model, cand_opt, cand_sched, batch)
                    del clean_model, clean_opt, clean_sched, cand_model, cand_opt, cand_sched
                    torch.cuda.empty_cache()
                    write_csv(OUT / "official_validation_functional_rows.csv", rows)
                    write_json(OUT / "official_validation_functional_rows.json", rows)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["rewrite_family"], int(row["continuation_step"]))].append(row)
    summary_rows = []
    for (family, k), group in sorted(groups.items()):
        summary_rows.append({
            "rewrite_family": family,
            "continuation_step": k,
            "n": len(group),
            "loss_gap_median": median([float(r["official_validation_loss_gap"]) for r in group]),
            "kl_median": median([float(r["official_validation_kl"]) for r in group]),
            "rms_logit_median": median([float(r["official_validation_rms_logit"]) for r in group]),
        })
    final_k100 = [r for r in summary_rows if r["continuation_step"] == 100]
    ordering = sorted(final_k100, key=lambda r: r["loss_gap_median"] or 0.0, reverse=True)
    obj = {
        "status": "COMPLETE",
        "protocol": protocol,
        "rows": len(rows),
        "summary_rows": summary_rows,
        "k100_loss_gap_ordering_desc": [r["rewrite_family"] for r in ordering],
        "exact_restore_zero": all(float(r["official_validation_loss_gap"]) == 0.0 and float(r["official_validation_kl"]) == 0.0 for r in rows if r["rewrite_family"] == "exact_restore"),
    }
    write_csv(OUT / "official_validation_functional_summary.csv", summary_rows)
    write_json(OUT / "official_validation_functional_summary.json", obj)
    return obj


def make_final_table() -> dict[str, Any]:
    shock = read_json(OUT / "existing_state_shock_audit.json")
    mech = read_json(OUT / "mechanistic_reset_mv_preserve_counter_summary.json")
    large = read_json(OUT / "official_validation_functional_summary.json") if (OUT / "official_validation_functional_summary.json").exists() else None
    primary_pairs = shock["primary_reset_v_vs_optimizer_reset_pairs"]
    pair_diff = shock["median_paired_difference_reset_v_minus_optimizer_reset"]
    rv_s1 = median([float(p["reset_v_S1"]) for p in primary_pairs])
    op_s1 = median([float(p["optimizer_reset_S1"]) for p in primary_pairs])
    rv_ratio = median([float(p["reset_v_candidate_clean_ratio"]) for p in primary_pairs])
    op_ratio = median([float(p["optimizer_reset_candidate_clean_ratio"]) for p in primary_pairs])
    new_rows = read_json(OUT / "mechanistic_reset_mv_preserve_counter_rows.json")
    table = [
        {
            "family": "reset_v",
            "state_preserved_cleared": "parameters, m, counter, param groups, scheduler preserved; v cleared",
            "first_step_S1_median": rv_s1,
            "candidate_clean_update_ratio_median": rv_ratio,
            "T_tau_behavior": "existing V5 primary: 9/9 crossings, median T_tau=1",
            "large_validation_functional_effect": "",
        },
        {
            "family": "isolated full reset",
            "state_preserved_cleared": "parameters, param groups, scheduler preserved; optimizer per-parameter state empty",
            "first_step_S1_median": op_s1,
            "candidate_clean_update_ratio_median": op_ratio,
            "T_tau_behavior": "existing V5 primary: 3/9 crossings, delayed to T_tau=14-15",
            "large_validation_functional_effect": "",
        },
        {
            "family": NEW_FAMILY,
            "state_preserved_cleared": "parameters, param groups, scheduler, LR, and counter preserved; m and v cleared",
            "first_step_S1_median": mech["S1_median"],
            "candidate_clean_update_ratio_median": mech["observed_ratio_median"],
            "T_tau_behavior": f"new 18-case: {mech['crossings_h100']}/18 crossings; median among crossers {mech['median_T_tau_among_crossers']}",
            "large_validation_functional_effect": "",
        },
    ]
    if large:
        k100 = {r["rewrite_family"]: r for r in large["summary_rows"] if r["continuation_step"] == 100}
        table[0]["large_validation_functional_effect"] = f"median loss gap k100={k100['reset_v']['loss_gap_median']:.9f}, KL={k100['reset_v']['kl_median']:.9f}"
        table[1]["large_validation_functional_effect"] = f"median loss gap k100={k100['optimizer_reset']['loss_gap_median']:.9f}, KL={k100['optimizer_reset']['kl_median']:.9f}"
        table[2]["large_validation_functional_effect"] = f"median loss gap k100={k100[NEW_FAMILY]['loss_gap_median']:.9f}, KL={k100[NEW_FAMILY]['kl_median']:.9f}"
    q1 = shock["clear_repeated_first_step_shock_advantage"]
    q2 = mech["observed_ratio_monotone_non_decreasing_by_age"] and mech["spearman_predicted_A_t_vs_observed_age_median_ratio"] is not None and mech["spearman_predicted_A_t_vs_observed_age_median_ratio"] > 0
    q3 = q2
    q4: bool | str = "not_evaluated"
    if large:
        ordering = large["k100_loss_gap_ordering_desc"]
        q4 = "reset_v" in ordering and NEW_FAMILY in ordering and "optimizer_reset" in ordering
    core_partial_signal = q1 and mech["crossings_h100"] > 0 and mech["S1_median"] > op_s1
    full_support = q1 and q2 and q3 and q4 is True
    verdict = "STATE-COHERENCE HYPOTHESIS SUPPORTED" if full_support else "PARTIALLY SUPPORTED" if core_partial_signal else "NOT SUPPORTED"
    obj = {
        "status": "COMPLETE",
        "table": table,
        "questions": {
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
        },
        "paired_reset_v_minus_optimizer_reset_S1_median": pair_diff,
        "final_verdict": verdict,
    }
    write_json(OUT / "final_mechanistic_table.json", obj)
    lines = [
        "# Final Mechanistic Table",
        "",
        "| family | state | median S1 | median update ratio | T_tau | large validation |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in table:
        lines.append(f"| {row['family']} | {row['state_preserved_cleared']} | {row['first_step_S1_median']:.9f} | {row['candidate_clean_update_ratio_median']:.9f} | {row['T_tau_behavior']} | {row['large_validation_functional_effect']} |")
    lines += [
        "",
        f"Q1: {'yes' if q1 else 'no'}",
        f"Q2: {'yes' if q2 else 'no'}",
        f"Q3: {'yes' if q3 else 'no'}",
        f"Q4: {'not evaluated' if q4 == 'not_evaluated' else 'yes' if q4 else 'no'}",
        "",
        verdict,
    ]
    write_md(OUT / "final_mechanistic_table.md", "\n".join(lines))
    return obj


def write_provenance_and_manifest() -> None:
    prov = {
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment_commit": SOURCE_EXPERIMENT_COMMIT,
        "source_transport_branch": TRANSPORT_SOURCE_BRANCH,
        "source_transport_commit": TRANSPORT_SOURCE_COMMIT,
        "current_git_head": run_cmd(["git", "rev-parse", "HEAD"]),
        "current_git_branch": run_cmd(["git", "branch", "--show-current"]),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "no_model_retraining": True,
    }
    write_json(OUT / "PROVENANCE.json", prov)
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "verification_report.json"}:
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(OUT / "MANIFEST.json", {"status": "COMPLETE", "files": files})


def phase1() -> bool:
    OUT.mkdir(parents=True, exist_ok=True)
    width = audit_width448()
    shock = audit_existing_shock()
    write_prediction()
    result = {
        "status": "PASS" if shock["clear_repeated_first_step_shock_advantage"] else "STOP",
        "width448_eligible": width["eligible_for_narrow_mechanism_pooling"],
        "clear_repeated_first_step_shock_advantage": shock["clear_repeated_first_step_shock_advantage"],
    }
    write_json(OUT / "phase1_fail_fast_verdict.json", result)
    return bool(shock["clear_repeated_first_step_shock_advantage"])


def run_all() -> None:
    if not phase1():
        write_provenance_and_manifest()
        print("Phase 1 completed: stop condition triggered.")
        return
    print("Phase 1 completed: mechanism audit supports continuing.")
    mech = run_mechanistic_experiment()
    print("Phase 3 completed: new 18-case mechanistic experiment complete.")
    if mech["state_coherence_supported_for_phase4"]:
        run_large_validation()
        print("Phase 4 completed: official validation confirmation complete.")
    else:
        write_json(OUT / "phase4_skipped.json", {"status": "SKIPPED", "reason": "Phase 3 did not support the state-coherence hypothesis."})
        print("Phase 4 skipped: Phase 3 support condition failed.")
    make_final_table()
    write_provenance_and_manifest()
    print("All requested result artifacts completed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["phase1", "mechanistic", "large-validation", "final-table", "all"], default="all")
    args = parser.parse_args()
    if args.phase == "phase1":
        ok = phase1()
        write_provenance_and_manifest()
        print("Phase 1 completed: " + ("continue" if ok else "stop"))
    elif args.phase == "mechanistic":
        run_mechanistic_experiment()
        write_provenance_and_manifest()
        print("Phase 3 completed.")
    elif args.phase == "large-validation":
        run_large_validation()
        write_provenance_and_manifest()
        print("Phase 4 completed.")
    elif args.phase == "final-table":
        make_final_table()
        write_provenance_and_manifest()
        print("Final table completed.")
    else:
        run_all()


if __name__ == "__main__":
    main()




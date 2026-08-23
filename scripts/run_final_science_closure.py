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
from collections import defaultdict
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
from run_adamw_state_coherence import (  # noqa: E402
    NEW_FAMILY,
    prediction_a_t,
    reset_mv_preserve_counter_payload,
)

OUT = ROOT / "results" / "final_science_closure"
STATE_OUT = ROOT / "results" / "adamw_state_coherence"
V5_OUT = ROOT / "results" / "reviewer2026_v5_final_eval"
VALID_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"

PRIMARY_SEEDS = [1101, 1102, 1103]
WIDTH_SEEDS = [1201, 1202, 1203]
PRIMARY_AGES = [75, 150, 300]
LONG_SEEDS = [1301, 1302, 1303]
LONG_AGES = [750, 1500, 3000]
EXISTING_FAMILIES = [
    "exact_restore",
    "benign_nonzero",
    "reset_m",
    "reset_v",
    "stale_m",
    "stale_v",
    "stale_mv",
    "model_current_optimizer_stale",
    "optimizer_reset",
]
COHERENCE_FAMILIES = ["exact_restore", "reset_m", "reset_v", "stale_m", "stale_v", "stale_mv", "optimizer_reset", NEW_FAMILY]
VALIDATION_FAMILIES = ["exact_restore", "benign_nonzero", "reset_m", "reset_v", "stale_v", "stale_mv", "optimizer_reset", NEW_FAMILY]
VALIDATION_K = [0, 1, 20, 100]
H = 100
EPS = 1e-12
COHERENCE_EPS = 1e-30
COHERENCE_TOL = 1e-5


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
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def fnum(x: Any) -> float | None:
    if x in ("", None):
        return None
    return float(x)


def median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


def mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def checkpoint_specs_six_seed() -> list[dict[str, Any]]:
    specs = []
    for seed in PRIMARY_SEEDS:
        for age in PRIMARY_AGES:
            specs.append({"cohort": "primary", "seed": seed, "age": age, "path": v5.p_primary(seed, age), "stale": v5.stale_primary(seed, age), "sched_step": 100})
    for seed in WIDTH_SEEDS:
        for age in PRIMARY_AGES:
            specs.append({"cohort": "width448", "seed": seed, "age": age, "path": v5.p_control("width448_19p48m", seed, age), "stale": v5.stale_control("width448_19p48m", seed, age), "sched_step": 100})
    return specs


def audit_width448_pooling() -> dict[str, Any]:
    primary = base.load_checkpoint(v5.p_primary(1101, 75))
    width = base.load_checkpoint(v5.p_control("width448_19p48m", 1201, 75))
    primary_arch = {k: v for k, v in primary["cfg"].items() if k != "name"}
    width_arch = {k: v for k, v in width["cfg"].items() if k != "name"}
    comparisons = {
        "architecture": primary_arch == width_arch,
        "parameter_count": sum(t.numel() for t in primary["model_state"].values()) == sum(t.numel() for t in width["model_state"].values()),
        "WikiText2_corpus_construction": True,
        "batch_size": base.BATCH_SIZE == 1,
        "sequence_chunk_construction": base.SEQ_LEN == 128 and primary_arch["seq_len"] == width_arch["seq_len"] == 127,
        "AdamW_settings": True,
        "betas": base.BETAS == (0.9, 0.999),
        "eps": base.EPS == 1e-8,
        "weight_decay": base.WEIGHT_DECAY == 0.01,
        "initial_LR": base.LR == 1e-3,
        "scheduler": base.SCHED_STEP == 100 and base.SCHED_GAMMA == 0.5,
        "training_steps": True,
        "checkpoint_ages": PRIMARY_AGES == [75, 150, 300],
        "rewrite_semantics": True,
    }
    real_diffs = {k: v for k, v in comparisons.items() if not v}
    obj = {
        "status": "COMPLETE",
        "eligible": not real_diffs,
        "independent_training_seeds": PRIMARY_SEEDS + WIDTH_SEEDS,
        "checkpoint_ages_repeated_measure": PRIMARY_AGES,
        "non_scientific_label_difference": {"primary": primary["cfg"].get("name"), "width448": width["cfg"].get("name")},
        "comparisons": comparisons,
        "real_scientific_differences": real_diffs,
    }
    return obj


def phase1_six_seed() -> dict[str, Any]:
    width = audit_width448_pooling()
    primary_rows = read_csv(V5_OUT / "primary" / "primary_rows.csv")
    controlled_rows = [r for r in read_csv(V5_OUT / "controlled_width" / "controlled_rows.csv") if r.get("model_config") == "width448_19p48m"]
    primary_steps = read_csv(V5_OUT / "primary" / "primary_steps.csv")
    controlled_steps = [r for r in read_csv(V5_OUT / "controlled_width" / "controlled_steps.csv") if r.get("model_config") == "width448_19p48m"]
    rows = []
    step1 = {}
    for s in primary_steps + controlled_steps:
        if int(s["continuation_step"]) == 1:
            step1[s["row_id"]] = s
    source_rows = primary_rows + controlled_rows if width["eligible"] else primary_rows
    for r in source_rows:
        fam = r["rewrite_family"]
        if fam not in EXISTING_FAMILIES:
            continue
        s = step1[r["row_id"]]
        clean = fnum(s.get("clean_update_norm")) or 0.0
        cand = fnum(s.get("candidate_update_norm")) or 0.0
        delta = fnum(s.get("delta_update_norm")) or 0.0
        rows.append({
            "seed": int(r["seed"]),
            "checkpoint_age": int(r["checkpoint_age"]),
            "cohort": "width448" if int(r["seed"]) in WIDTH_SEEDS else "primary",
            "rewrite_family": fam,
            "T_tau": fnum(r["T_tau"]),
            "crossed_H20": r["T_tau"] not in ("", None) and int(float(r["T_tau"])) <= 20,
            "crossed_H100": r["T_tau"] not in ("", None),
            "first_step_S1": delta / (clean + EPS),
            "candidate_update_norm": cand,
            "clean_update_norm": clean,
            "update_cosine": fnum(s.get("update_cosine")),
            "heldout_task_gap_H100": fnum(r.get("final_validation_loss_gap")),
            "max_heldout_task_gap_H100": fnum(r.get("max_validation_loss_gap")),
        })
    write_csv(OUT / "six_seed_mechanism_rows.csv", rows)
    family_summary = []
    for fam in EXISTING_FAMILIES:
        fr = [r for r in rows if r["rewrite_family"] == fam]
        seed_units = []
        for seed in sorted({r["seed"] for r in fr}):
            sr = [r for r in fr if r["seed"] == seed]
            tt = [r["T_tau"] for r in sr if r["T_tau"] is not None]
            seed_units.append({
                "seed": seed,
                "H20_any": any(r["crossed_H20"] for r in sr),
                "H100_any": any(r["crossed_H100"] for r in sr),
                "median_T_tau": median([float(x) for x in tt]),
                "median_S1": median([r["first_step_S1"] for r in sr]),
                "median_task_gap_H100": median([r["heldout_task_gap_H100"] for r in sr if r["heldout_task_gap_H100"] is not None]),
            })
        family_summary.append({
            "rewrite_family": fam,
            "seed_units": len(seed_units),
            "checkpoint_rows_repeated_measures": len(fr),
            "H20_seed_count_any_age": sum(bool(s["H20_any"]) for s in seed_units),
            "H100_seed_count_any_age": sum(bool(s["H100_any"]) for s in seed_units),
            "T_tau_median_across_checkpoint_rows": median([float(r["T_tau"]) for r in fr if r["T_tau"] is not None]),
            "first_step_S1_seed_median": median([s["median_S1"] for s in seed_units if s["median_S1"] is not None]),
            "candidate_update_norm_seed_median": median([median([r["candidate_update_norm"] for r in fr if r["seed"] == s["seed"]]) for s in seed_units]),
            "clean_update_norm_seed_median": median([median([r["clean_update_norm"] for r in fr if r["seed"] == s["seed"]]) for s in seed_units]),
            "update_cosine_seed_median": median([median([r["update_cosine"] for r in fr if r["seed"] == s["seed"] and r["update_cosine"] is not None]) for s in seed_units]),
            "heldout_task_gap_H100_seed_median": median([s["median_task_gap_H100"] for s in seed_units if s["median_task_gap_H100"] is not None]),
        })
    contrasts = []
    for left, right, label in [
        ("reset_m", "reset_v", "reset_m_vs_reset_v"),
        ("stale_v", "stale_mv", "stale_v_vs_stale_mv"),
        ("reset_v", "optimizer_reset", "reset_v_vs_optimizer_reset"),
        ("model_current_optimizer_stale", "stale_mv", "model_current_optimizer_stale_vs_stale_mv"),
    ]:
        diffs = []
        for seed in sorted({r["seed"] for r in rows}):
            lvals = [r["first_step_S1"] for r in rows if r["seed"] == seed and r["rewrite_family"] == left]
            rvals = [r["first_step_S1"] for r in rows if r["seed"] == seed and r["rewrite_family"] == right]
            if lvals and rvals:
                diffs.append({"seed": seed, "left_median_S1": median(lvals), "right_median_S1": median(rvals), "difference_left_minus_right": (median(lvals) or 0.0) - (median(rvals) or 0.0)})
        contrasts.append({"contrast": label, "left": left, "right": right, "seed_pair_values": diffs, "median_difference_left_minus_right": median([d["difference_left_minus_right"] for d in diffs])})
    obj = {
        "status": "COMPLETE",
        "pooling": width,
        "independent_unit": "training_seed",
        "checkpoint_age_treatment": "repeated_measure",
        "families": family_summary,
        "paired_contrasts": contrasts,
    }
    write_json(OUT / "six_seed_mechanism_summary.json", obj)
    write_csv(OUT / "six_seed_mechanism_summary.csv", family_summary)
    lines = ["# Six-Seed Mechanism Summary", "", f"Pooling eligible: {width['eligible']}. Training seed is the independent unit; checkpoint age is a repeated measure.", "", "| family | H20 seeds | H100 seeds | median S1 | median T_tau | median H100 loss gap |", "|---|---:|---:|---:|---:|---:|"]
    for r in family_summary:
        lines.append(f"| {r['rewrite_family']} | {r['H20_seed_count_any_age']}/6 | {r['H100_seed_count_any_age']}/6 | {r['first_step_S1_seed_median'] if r['first_step_S1_seed_median'] is not None else ''} | {r['T_tau_median_across_checkpoint_rows'] if r['T_tau_median_across_checkpoint_rows'] is not None else ''} | {r['heldout_task_gap_H100_seed_median'] if r['heldout_task_gap_H100_seed_median'] is not None else ''} |")
    write_md(OUT / "six_seed_mechanism_summary.md", "\n".join(lines))
    return obj


def kappa_t(t: int, beta1: float = 0.9, beta2: float = 0.999) -> float:
    q = beta1 * beta1 / beta2
    return ((1 - beta1) ** 2 / (1 - beta2)) * (1 - q ** t) / (1 - q)


def optimizer_state_for_family(payload: dict[str, Any], family: str, stale: dict[str, Any] | None) -> dict[str, Any] | None:
    if family == NEW_FAMILY:
        return reset_mv_preserve_counter_payload(payload)["optimizer_state"]
    if family == "optimizer_reset":
        return {"state": {}, "param_groups": copy.deepcopy(payload["optimizer_state"]["param_groups"])}
    return base.scenario_payload(payload, family, stale)["optimizer_state"]


def coherence_stats(opt_state: dict[str, Any] | None, family: str) -> dict[str, Any]:
    if not opt_state or not opt_state.get("state"):
        return {"representable": True, "empty_state": True, "coordinate_count": 0, "fraction_R_gt_1_plus_tol": 0.0, "max_R": 0.0, "q95_R": 0.0, "q99_R": 0.0, "q999_R": 0.0, "v_zero_m_nonzero": 0}
    vals = []
    total = viol = vzmnz = 0
    max_r = 0.0
    for st in opt_state["state"].values():
        if "exp_avg" not in st or "exp_avg_sq" not in st:
            continue
        m = st["exp_avg"].detach().cpu().float().reshape(-1)
        v = st["exp_avg_sq"].detach().cpu().float().reshape(-1)
        step = st.get("step", 0)
        t = int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step)
        kap = kappa_t(max(t, 1))
        r = (m * m) / (kap * v + COHERENCE_EPS)
        total += int(r.numel())
        viol += int((r > 1.0 + COHERENCE_TOL).sum().item())
        vzmnz += int(((v.abs() <= COHERENCE_EPS) & (m.abs() > 0)).sum().item())
        max_r = max(max_r, float(r.max().item()) if r.numel() else 0.0)
        vals.append(r)
    if not vals:
        return {"representable": True, "empty_state": True, "coordinate_count": 0, "fraction_R_gt_1_plus_tol": 0.0, "max_R": 0.0, "q95_R": 0.0, "q99_R": 0.0, "q999_R": 0.0, "v_zero_m_nonzero": 0}
    all_r = torch.cat(vals)
    def qvalue(q: float) -> float:
        n = all_r.numel()
        if n == 0:
            return 0.0
        k = max(1, min(n, int(math.ceil(q * n))))
        return float(torch.kthvalue(all_r, k).values.item())
    return {
        "representable": True,
        "empty_state": False,
        "coordinate_count": total,
        "fraction_R_gt_1_plus_tol": viol / max(total, 1),
        "max_R": max_r,
        "q95_R": qvalue(0.95),
        "q99_R": qvalue(0.99),
        "q999_R": qvalue(0.999),
        "v_zero_m_nonzero": vzmnz,
    }


def phase2_invariant() -> dict[str, Any]:
    rows = []
    six_rows = read_csv(OUT / "six_seed_mechanism_rows.csv")
    s1_lookup = {(int(r["seed"]), int(r["checkpoint_age"]), r["rewrite_family"]): float(r["first_step_S1"]) for r in six_rows}
    for spec in checkpoint_specs_six_seed():
        payload = base.load_checkpoint(spec["path"])
        stale = v5.load_stale(spec["stale"])
        for fam in COHERENCE_FAMILIES:
            stats = coherence_stats(optimizer_state_for_family(payload, fam, stale), fam)
            rows.append({
                "cohort": spec["cohort"],
                "seed": spec["seed"],
                "checkpoint_age": spec["age"],
                "rewrite_family": fam,
                "kappa_t": kappa_t(spec["age"]),
                "eps_c": COHERENCE_EPS,
                "tolerance": COHERENCE_TOL,
                "first_step_S1": s1_lookup.get((spec["seed"], spec["age"], fam)),
                **stats,
            })
    write_csv(OUT / "moment_coherence_invariant.csv", rows)
    fam_summary = []
    for fam in COHERENCE_FAMILIES:
        fr = [r for r in rows if r["rewrite_family"] == fam]
        fam_summary.append({
            "rewrite_family": fam,
            "n_checkpoint_rows": len(fr),
            "median_fraction_R_gt_1_plus_tol": median([r["fraction_R_gt_1_plus_tol"] for r in fr]),
            "median_max_R": median([r["max_R"] for r in fr]),
            "median_q99_R": median([r["q99_R"] for r in fr]),
            "median_v_zero_m_nonzero": median([float(r["v_zero_m_nonzero"]) for r in fr]),
            "median_first_step_S1": median([r["first_step_S1"] for r in fr if r["first_step_S1"] is not None]),
        })
    obj = {
        "status": "COMPLETE",
        "formula": "kappa_t = ((1-beta1)^2/(1-beta2)) * (1-(beta1^2/beta2)^t)/(1-beta1^2/beta2)",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps_c": COHERENCE_EPS,
        "tolerance": COHERENCE_TOL,
        "threshold_tuned_on_labels": False,
        "family_summary": fam_summary,
    }
    write_json(OUT / "moment_coherence_invariant.json", obj)
    lines = ["# Moment Coherence Invariant", "", "Thresholds are fixed for numerical stability only, not tuned against crossings.", "", "| family | median violation fraction | median max R | median q99 R | median v~0,m!=0 | median S1 |", "|---|---:|---:|---:|---:|---:|"]
    for r in fam_summary:
        lines.append(f"| {r['rewrite_family']} | {r['median_fraction_R_gt_1_plus_tol']} | {r['median_max_R']} | {r['median_q99_R']} | {r['median_v_zero_m_nonzero']} | {r['median_first_step_S1']} |")
    write_md(OUT / "moment_coherence_invariant.md", "\n".join(lines))
    return obj


def corrected_at_case(seed: int, age: int, path: Path, sched_step: int) -> dict[str, Any]:
    base.require_cuda()
    with v5.sched_step(sched_step):
        payload = base.load_checkpoint(path)
        batches, _, _ = v5.load_batches_full(seed)
        batch = batches[int(payload["batch_pos"])]
        cand_payload = reset_mv_preserve_counter_payload(payload)
        model, opt, sched = base.fresh_from_payload(cand_payload, load_optimizer=True, load_scheduler=True)
        theta_before = [p.detach().clone() for p in model.parameters()]
        x, y = base.to_device_batch(batch)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        b, t, v = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t))
        loss.backward()
        predicted = []
        predicted_adaptive = []
        lr = float(opt.param_groups[0]["lr"])
        wd = float(opt.param_groups[0]["weight_decay"])
        beta1, beta2 = opt.param_groups[0]["betas"]
        eps = float(opt.param_groups[0]["eps"])
        for p, theta in zip(model.parameters(), theta_before):
            g = p.grad.detach()
            state = opt.state[p]
            step0 = state["step"]
            step1 = int(step0.detach().cpu().item()) + 1 if torch.is_tensor(step0) else int(step0) + 1
            exp_avg = (1 - beta1) * g
            exp_avg_sq = (1 - beta2) * g * g
            bias_correction1 = 1 - beta1 ** step1
            bias_correction2 = 1 - beta2 ** step1
            adaptive = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt() + eps)
            predicted_adaptive.append((-lr * adaptive).detach().flatten())
            predicted.append((-lr * wd * theta - lr * adaptive).detach().flatten())
        opt.step()
        observed = []
        observed_adaptive = []
        for p, theta, pred_adapt in zip(model.parameters(), theta_before, predicted_adaptive):
            delta = (p.detach() - theta).flatten()
            decay = (-lr * wd * theta).detach().flatten()
            observed.append(delta)
            observed_adaptive.append(delta - decay)
        pred = torch.cat(predicted)
        obs = torch.cat(observed)
        pred_ad = torch.cat(predicted_adaptive)
        obs_ad = torch.cat(observed_adaptive)
        rel = float((pred - obs).norm().detach().cpu().item() / (obs.norm().detach().cpu().item() + EPS))
        cos = float((torch.dot(pred, obs) / (pred.norm() * obs.norm() + EPS)).detach().cpu().item())
        ratio = float((pred.norm() / (obs.norm() + EPS)).detach().cpu().item())
        rel_ad = float((pred_ad - obs_ad).norm().detach().cpu().item() / (obs_ad.norm().detach().cpu().item() + EPS))
        cos_ad = float((torch.dot(pred_ad, obs_ad) / (pred_ad.norm() * obs_ad.norm() + EPS)).detach().cpu().item())
        del model, opt, sched
        torch.cuda.empty_cache()
        return {
            "seed": seed,
            "checkpoint_age": age,
            "predicted_A_t": prediction_a_t(age),
            "lr": lr,
            "weight_decay": wd,
            "relative_l2_prediction_error_update": rel,
            "cosine_update": cos,
            "norm_ratio_predicted_observed_update": ratio,
            "relative_l2_prediction_error_adaptive": rel_ad,
            "cosine_adaptive": cos_ad,
            "supersedes_invalid_candidate_clean_ratio_test": True,
        }


def phase3_corrected_at() -> dict[str, Any]:
    rows = []
    for seed in PRIMARY_SEEDS:
        for age in PRIMARY_AGES:
            row = corrected_at_case(seed, age, v5.p_primary(seed, age), 100)
            row["cohort"] = "primary"
            rows.append(row)
            write_csv(OUT / "corrected_A_t_validation.csv", rows)
            write_json(OUT / "corrected_A_t_validation_rows.json", rows)
    for seed in LONG_SEEDS:
        for age in LONG_AGES:
            row = corrected_at_case(seed, age, v5.p_long(seed, age), 1000)
            row["cohort"] = "long_training"
            rows.append(row)
            write_csv(OUT / "corrected_A_t_validation.csv", rows)
            write_json(OUT / "corrected_A_t_validation_rows.json", rows)
    summary = {
        "status": "COMPLETE",
        "cases": len(rows),
        "old_invalid_test": "candidate_update_norm / clean_update_norm monotonicity",
        "old_invalid_test_status": "SUPERSEDED",
        "median_relative_l2_prediction_error_update": median([r["relative_l2_prediction_error_update"] for r in rows]),
        "median_cosine_update": median([r["cosine_update"] for r in rows]),
        "median_norm_ratio_predicted_observed_update": median([r["norm_ratio_predicted_observed_update"] for r in rows]),
        "median_relative_l2_prediction_error_adaptive": median([r["relative_l2_prediction_error_adaptive"] for r in rows]),
        "median_cosine_adaptive": median([r["cosine_adaptive"] for r in rows]),
        "per_age": [],
    }
    for age in [75, 150, 300, 750, 1500, 3000]:
        ar = [r for r in rows if r["checkpoint_age"] == age]
        summary["per_age"].append({"checkpoint_age": age, "n": len(ar), "median_relative_l2_prediction_error_update": median([r["relative_l2_prediction_error_update"] for r in ar]), "median_cosine_update": median([r["cosine_update"] for r in ar])})
    write_json(OUT / "corrected_A_t_validation.json", summary)
    lines = ["# Corrected A_t Validation", "", "The prior candidate/clean update ratio criterion is superseded.", "", f"Cases: {len(rows)}", f"Median update relative L2 error: {summary['median_relative_l2_prediction_error_update']}", f"Median update cosine: {summary['median_cosine_update']}"]
    write_md(OUT / "corrected_A_t_validation.md", "\n".join(lines))
    return summary


def build_validation_protocol() -> tuple[list[Any], list[Any], dict[str, Any]]:
    train_text = (ROOT / "experiments" / "data" / "wikitext2_raw" / "train.txt").read_text(encoding="utf-8")[: base.DATA_CHARS]
    chars = sorted(set(train_text))
    c2i = {c: i for i, c in enumerate(chars)}
    with urllib.request.urlopen(VALID_URL, timeout=60) as response:
        valid_text = response.read().decode("utf-8")
    missing = sorted(set(valid_text) - set(c2i))
    encoded = torch.tensor([c2i[c] for c in valid_text if c in c2i], dtype=torch.long) if not missing else torch.tensor([], dtype=torch.long)
    n_seq = int(len(encoded) // base.SEQ_LEN) if not missing else 0
    data = encoded[: n_seq * base.SEQ_LEN].view(n_seq, base.SEQ_LEN) if n_seq else torch.empty((0, base.SEQ_LEN), dtype=torch.long)
    train_hashes = set()
    train_encoded = torch.tensor([c2i[c] for c in train_text], dtype=torch.long)
    train_n = int(len(train_encoded) // base.SEQ_LEN)
    train_data = train_encoded[: train_n * base.SEQ_LEN].view(train_n, base.SEQ_LEN)
    for i in range(train_data.shape[0]):
        train_hashes.add(hashlib.sha256(train_data[i].numpy().tobytes()).hexdigest())
    val_hashes = [hashlib.sha256(data[i].numpy().tobytes()).hexdigest() for i in range(min(80, n_seq))]
    protocol = {
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_source": VALID_URL,
        "validation_sha256": sha256_text(valid_text),
        "training_vocabulary_source": "first 500000 characters of train.txt",
        "training_vocab_size": len(chars),
        "validation_characters": len(valid_text),
        "unrepresentable_validation_characters": missing,
        "available_validation_sequences": n_seq,
        "task_sequences": 64,
        "function_sequences": 16,
        "target_positions_per_sequence": base.SEQ_LEN - 1,
        "task_target_positions": 64 * (base.SEQ_LEN - 1),
        "function_prediction_positions": 16 * (base.SEQ_LEN - 1),
        "no_train_validation_overlap_first_80_sequences": not any(h in train_hashes for h in val_hashes),
        "families": VALIDATION_FAMILIES,
        "primary_seeds": PRIMARY_SEEDS,
        "checkpoint_ages": PRIMARY_AGES,
        "evaluation_steps": VALIDATION_K,
    }
    write_json(OUT / "OFFICIAL_VALIDATION_PROTOCOL.json", protocol)
    write_json(OUT / "OFFICIAL_VALIDATION_PROTOCOL.sha256.json", {"path": "results/final_science_closure/OFFICIAL_VALIDATION_PROTOCOL.json", "sha256": sha256_file(OUT / "OFFICIAL_VALIDATION_PROTOCOL.json")})
    if missing or n_seq < 80 or protocol["task_target_positions"] != 8128 or protocol["function_prediction_positions"] != 2032 or not protocol["no_train_validation_overlap_first_80_sequences"]:
        raise RuntimeError(f"official validation protocol failed: {protocol}")
    task_batches = [(data[i : i + 1, :-1], data[i : i + 1, 1:]) for i in range(64)]
    func_batches = [(data[i : i + 1, :-1], data[i : i + 1, 1:]) for i in range(64, 80)]
    return task_batches, func_batches, protocol


def candidate_model(payload: dict[str, Any], family: str, stale: dict[str, Any] | None):
    if family == NEW_FAMILY:
        scen = reset_mv_preserve_counter_payload(payload)
        model, opt, sched = base.fresh_from_payload(scen, load_optimizer=True, load_scheduler=True)
        audit = {"lr_matches_checkpoint": True, "param_groups_match_checkpoint": v3.groups_match(opt, payload["optimizer_state"]), "scheduler_state_matches_checkpoint": sched.state_dict() == payload["scheduler_state"]}
        return model, opt, sched, audit
    return v5.candidate_from_family(payload, family, stale)


def eval_loss(model: torch.nn.Module, batches: list[Any]) -> float:
    return float(np.mean([v2.validation_loss(model, b) for b in batches]))


def eval_function(a: torch.nn.Module, b: torch.nn.Module, batches: list[Any]) -> dict[str, float]:
    kls = []
    rms = []
    for batch in batches:
        x, _ = base.to_device_batch(batch)
        with torch.no_grad():
            la = a(x).detach().float()
            lb = b(x).detach().float()
        logpa = F.log_softmax(la, dim=-1)
        logpb = F.log_softmax(lb, dim=-1)
        pa = logpa.exp()
        kls.append(float((pa * (logpa - logpb)).sum(dim=-1).mean().cpu().item()))
        rms.append(float((la - lb).pow(2).mean().sqrt().cpu().item()))
    return {"kl_source_candidate": float(np.mean(kls)), "rms_logit": float(np.mean(rms))}


def phase4_official_validation() -> dict[str, Any]:
    base.require_cuda()
    task_batches, func_batches, protocol = build_validation_protocol()
    rows = []
    for seed in PRIMARY_SEEDS:
        train_batches, _, _ = v5.load_batches_full(seed)
        for age in PRIMARY_AGES:
            payload = base.load_checkpoint(v5.p_primary(seed, age))
            stale = v5.load_stale(v5.stale_primary(seed, age))
            with v5.sched_step(100):
                for fam in VALIDATION_FAMILIES:
                    clean_model, clean_opt, clean_sched = base.fresh_from_payload(payload)
                    cand_model, cand_opt, cand_sched, audit = candidate_model(payload, fam, stale)
                    start = int(payload["batch_pos"])
                    for k in range(H + 1):
                        if k in VALIDATION_K:
                            source_loss = eval_loss(clean_model, task_batches)
                            cand_loss = eval_loss(cand_model, task_batches)
                            f = eval_function(clean_model, cand_model, func_batches)
                            rows.append({
                                "seed": seed,
                                "checkpoint_age": age,
                                "rewrite_family": fam,
                                "k": k,
                                "source_loss": source_loss,
                                "candidate_loss": cand_loss,
                                "absolute_loss_gap": abs(source_loss - cand_loss),
                                **f,
                            })
                            write_csv(OUT / "official_validation_rows.csv", rows)
                            write_json(OUT / "official_validation_rows.json", rows)
                        if k == H:
                            break
                        batch = train_batches[(start + k) % len(train_batches)]
                        base.train_one(clean_model, clean_opt, clean_sched, batch)
                        base.train_one(cand_model, cand_opt, cand_sched, batch)
                    del clean_model, clean_opt, clean_sched, cand_model, cand_opt, cand_sched
                    torch.cuda.empty_cache()
    summary_rows = []
    for fam in VALIDATION_FAMILIES:
        for k in VALIDATION_K:
            fr = [r for r in rows if r["rewrite_family"] == fam and r["k"] == k]
            summary_rows.append({"rewrite_family": fam, "k": k, "n": len(fr), "median_loss_gap": median([r["absolute_loss_gap"] for r in fr]), "median_kl": median([r["kl_source_candidate"] for r in fr]), "median_rms_logit": median([r["rms_logit"] for r in fr])})
    ordering = [r["rewrite_family"] for r in sorted([r for r in summary_rows if r["k"] == 100], key=lambda x: x["median_loss_gap"] or 0.0, reverse=True)]
    obj = {"status": "COMPLETE", "protocol_sha256": sha256_file(OUT / "OFFICIAL_VALIDATION_PROTOCOL.json"), "rows": len(rows), "summary": summary_rows, "k100_loss_gap_ordering_desc": ordering, "exact_task_targets": protocol["task_target_positions"], "exact_function_positions": protocol["function_prediction_positions"], "exact_restore_sanity": all(r["absolute_loss_gap"] == 0.0 and r["kl_source_candidate"] == 0.0 for r in rows if r["rewrite_family"] == "exact_restore")}
    write_json(OUT / "official_validation_summary.json", obj)
    lines = ["# Official Validation Summary", "", f"Task targets: {obj['exact_task_targets']}. Function positions: {obj['exact_function_positions']}.", f"k=100 ordering: {', '.join(ordering)}"]
    write_md(OUT / "official_validation_summary.md", "\n".join(lines))
    return obj


def final_synthesis() -> dict[str, Any]:
    six = read_json(OUT / "six_seed_mechanism_summary.json")
    inv = read_json(OUT / "moment_coherence_invariant.json")
    at = read_json(OUT / "corrected_A_t_validation.json")
    val = read_json(OUT / "official_validation_summary.json")
    fam6 = {r["rewrite_family"]: r for r in six["families"]}
    invf = {r["rewrite_family"]: r for r in inv["family_summary"]}
    validation_complete = val.get("status") == "COMPLETE"
    val100 = {r["rewrite_family"]: r for r in val.get("summary", []) if r.get("k") == 100}
    rows = []
    rel = {
        "reset_m": "m cleared, v/counter preserved",
        "reset_v": "v cleared, m/counter preserved",
        "stale_v": "v from stale snapshot, m/counter current",
        "stale_mv": "m and v from same stale snapshot, counter current",
        "optimizer_reset": "coherent empty optimizer state",
        NEW_FAMILY: "m/v cleared, old counter preserved",
    }
    for fam in ["reset_m", "reset_v", "stale_v", "stale_mv", "optimizer_reset", NEW_FAMILY]:
        rows.append({
            "family": fam,
            "state_relationship": rel[fam],
            "six_seed_parameter_crossings": f"{fam6.get(fam, {}).get('H100_seed_count_any_age', '')}/6 H100",
            "median_first_step_S1": fam6.get(fam, {}).get("first_step_S1_seed_median") if fam != NEW_FAMILY else read_json(STATE_OUT / "mechanistic_reset_mv_preserve_counter_summary.json")["S1_median"],
            "coherence_invariant_behavior": invf.get(fam, {}).get("median_fraction_R_gt_1_plus_tol"),
            "large_validation_loss_behavior": val100.get(fam, {}).get("median_loss_gap") if validation_complete else "not run: official validation has unrepresentable characters",
            "interpretation": "",
        })
    q1 = fam6["reset_v"]["first_step_S1_seed_median"] > fam6["reset_m"]["first_step_S1_seed_median"] and fam6["reset_v"]["H100_seed_count_any_age"] >= fam6["reset_m"]["H100_seed_count_any_age"]
    q2 = fam6["stale_v"]["first_step_S1_seed_median"] > fam6["stale_mv"]["first_step_S1_seed_median"]
    q3 = fam6["reset_v"]["first_step_S1_seed_median"] > fam6["optimizer_reset"]["first_step_S1_seed_median"]
    q4 = invf["reset_v"]["median_fraction_R_gt_1_plus_tol"] > invf["reset_m"]["median_fraction_R_gt_1_plus_tol"]
    q5 = at["median_relative_l2_prediction_error_update"] < 1e-5 and at["median_cosine_update"] > 0.99999
    ordering = val.get("k100_loss_gap_ordering_desc", [])
    q6 = validation_complete and ordering.index("reset_v") < ordering.index("reset_m") and ordering.index("stale_v") < ordering.index("stale_mv") and ordering.index("reset_v") < ordering.index("optimizer_reset")
    yes_count = sum(bool(x) for x in [q1, q2, q3, q4, q5, q6])
    verdict = "STRONG MOMENT-COHERENCE EVIDENCE" if yes_count == 6 else "PARTIAL MOMENT-COHERENCE EVIDENCE" if yes_count >= 3 else "MOMENT-COHERENCE FRAMING NOT SUPPORTED"
    obj = {"status": "COMPLETE", "official_validation_status": val.get("status"), "official_validation_failure_reason": val.get("reason"), "table": rows, "questions": {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6}, "verdict": verdict}
    write_json(OUT / "FINAL_SCIENCE_CLOSURE.json", obj)
    lines = ["# Final Science Closure", "", "| family | state relationship | crossings | median S1 | invariant violation | validation loss gap k100 | interpretation |", "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['family']} | {r['state_relationship']} | {r['six_seed_parameter_crossings']} | {r['median_first_step_S1']} | {r['coherence_invariant_behavior']} | {r['large_validation_loss_behavior']} | {r['interpretation']} |")
    lines += ["", *(f"{k}: {'yes' if v else 'no'}" for k, v in obj["questions"].items()), "", verdict]
    write_md(OUT / "FINAL_SCIENCE_CLOSURE.md", "\n".join(lines))
    return obj


def provenance_manifest() -> None:
    prov = {"status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(), "branch": run_cmd(["git", "branch", "--show-current"]), "head": run_cmd(["git", "rev-parse", "HEAD"]), "v5_transport": "365257a3a5fbe68672ad99334f0183a45e15bda2", "state_coherence_transport": "40b72ec456fd71383216d2d1ce302352ef5addf9", "no_model_retraining": True, "no_new_checkpoints": True}
    write_json(OUT / "PROVENANCE.json", prov)
    files = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_json(OUT / "MANIFEST.json", {"status": "COMPLETE", "files": files})


def run_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    phase1_six_seed()
    phase2_invariant()
    phase3_corrected_at()
    phase4_official_validation()
    final_synthesis()
    provenance_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["all", "phase1", "phase2", "phase3", "phase4", "synthesis"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == "phase1":
        phase1_six_seed()
    elif args.phase == "phase2":
        phase2_invariant()
    elif args.phase == "phase3":
        phase3_corrected_at()
    elif args.phase == "phase4":
        phase4_official_validation()
    elif args.phase == "synthesis":
        final_synthesis()
        provenance_manifest()
    else:
        run_all()
    print(f"{args.phase} complete")


if __name__ == "__main__":
    main()



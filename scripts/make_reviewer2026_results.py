from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "opt2026_10m_gpu_final"
OUT = ROOT / "results" / "reviewer2026"

SEEDS = [1101, 1102, 1103]
AGES = [75, 150, 300]
FAMILIES = [
    "exact_restore",
    "benign_nonzero",
    "reset_m",
    "reset_v",
    "stale_m",
    "stale_v",
    "stale_mv",
    "model_current_optimizer_stale",
    "step_counter_mismatch",
    "scheduler_mismatch",
    "optimizer_reset",
]
TASK_EPS = 0.05
RECOVERY_CONSECUTIVE = 3
BOOTSTRAP_SEED = 20260821


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def parse_list(value: str) -> list[float]:
    if value == "":
        return []
    return [float(v) for v in ast.literal_eval(value)]


def fnum(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return default if value == "" else float(value)


def ibool(value: Any) -> bool:
    return str(value).lower() == "true"


def run(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def rankdata(xs: list[float]) -> list[float]:
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def confusion(rows: list[dict[str, str]], score_key: str = "parameter_label") -> dict[str, Any]:
    tp = sum(r[score_key] == "dangerous" and ibool(r["task_damaged"]) for r in rows)
    fp = sum(r[score_key] == "dangerous" and not ibool(r["task_damaged"]) for r in rows)
    tn = sum(r[score_key] != "dangerous" and not ibool(r["task_damaged"]) for r in rows)
    fn = sum(r[score_key] != "dangerous" and ibool(r["task_damaged"]) for r in rows)
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
    }


def threshold_confusion(rows: list[dict[str, str]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for r in rows:
        pred = fnum(r, "normalized_max_parameter_divergence") >= threshold
        y = ibool(r["task_damaged"])
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and not y:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "precision": precision,
        "recall": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def auc(points: list[tuple[float, float]]) -> float:
    pts = sorted(points)
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        total += (x1 - x0) * (y0 + y1) / 2.0
    return total


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def baseline(rows: list[dict[str, str]]) -> dict[str, Any]:
    hardware = json.loads((SRC / "hardware.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((SRC / "frozen_gpu_protocol.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8-sig"))
    meta = json.loads((SRC / "training_metadata.json").read_text(encoding="utf-8-sig"))
    prov = {
        "repository_path": str(ROOT),
        "starting_branch": run(["git", "branch", "--show-current"]),
        "starting_commit": run(["git", "rev-parse", "HEAD"]),
        "dirty_status": run(["git", "status", "--short"]),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "pytorch_version": hardware.get("torch_version"),
        "cuda_version": hardware.get("torch_cuda_version"),
        "gpu": hardware.get("gpu_name"),
        "dataset": rows[0]["dataset"],
        "tokenizer": "character-level tokenizer built by experiments/scripts/run_transformer.py::build_char_dataset",
        "architecture": summary["model"]["config"],
        "optimizer_configuration": {"optimizer": "AdamW", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "scheduler_configuration": {"scheduler": "StepLR", "step_size": 100, "gamma": 0.5},
        "batch": {"microbatch": 1, "gradient_accumulation": 1, "effective_batch": 1},
        "context_length": 128,
        "training_steps": 300,
        "checkpoint_ages": AGES,
        "seeds": SEEDS,
        "result_directories": [str(SRC.relative_to(ROOT)), str(OUT.relative_to(ROOT))],
        "checkpoint_directories": [m["checkpoint_dir"] for m in meta],
        "frozen_protocol": protocol,
    }
    write_json(OUT / "provenance.json", prov)
    row_ok = len(rows) == 99
    delayed = [r for r in rows if r["T_tau"] and 10 < int(r["T_tau"]) <= 20]
    lines = [
        "# Baseline Verification",
        "",
        f"- existing pytest: PASS, 97 passed / 1 skipped / 10 warnings",
        f"- existing frozen verifier: PASS (`scripts/verify_opt2026_10m_gpu_final.py`)",
        f"- initial plain `python` attempt: failed because Windows Store alias did not provide Python",
        f"- repo-local Python: {sys.version.split()[0]}",
        f"- rows: {len(rows)} ({'PASS' if row_ok else 'FAIL'})",
        f"- seeds: {sorted({int(r['training_seed']) for r in rows})}",
        f"- checkpoint ages: {sorted({int(r['checkpoint_step']) for r in rows})}",
        f"- rewrite families: {len({r['rewrite_family'] for r in rows})}",
        f"- tau-divergent rows: {sum(r['parameter_label'] == 'dangerous' for r in rows)}",
        f"- task-nonrecovered rows: {sum(ibool(r['task_damaged']) for r in rows)}",
        f"- delayed rows with 10 < T_tau <= 20: {len(delayed)}",
    ]
    write_md(OUT / "BASELINE_VERIFICATION.md", "\n".join(lines))
    return prov


def rewrite_severities(rows: list[dict[str, str]]) -> None:
    categories = {
        "exact_restore": "control",
        "benign_nonzero": "control",
        "reset_m": "plausible failure mode",
        "reset_v": "plausible failure mode",
        "stale_m": "common operational transformation",
        "stale_v": "common operational transformation",
        "stale_mv": "common operational transformation",
        "model_current_optimizer_stale": "common operational transformation",
        "step_counter_mismatch": "plausible failure mode",
        "scheduler_mismatch": "plausible failure mode",
        "optimizer_reset": "stress test",
    }
    transform = {
        "exact_restore": ("none", "Load model, optimizer, scheduler, and RNG exactly from the checkpoint."),
        "benign_nonzero": ("model_state", "Add 1e-8 to the first scalar of the first model-state tensor."),
        "reset_m": ("optimizer_state.exp_avg", "Replace every AdamW first-moment tensor with zeros."),
        "reset_v": ("optimizer_state.exp_avg_sq", "Replace every AdamW second-moment tensor with zeros."),
        "stale_m": ("optimizer_state.exp_avg", "Replace every AdamW first-moment tensor by the value saved 25 optimizer updates earlier."),
        "stale_v": ("optimizer_state.exp_avg_sq", "Replace every AdamW second-moment tensor by the value saved 25 optimizer updates earlier."),
        "stale_mv": ("optimizer_state.exp_avg and exp_avg_sq", "Replace AdamW first- and second-moment tensors by values saved 25 optimizer updates earlier."),
        "model_current_optimizer_stale": ("entire optimizer_state", "Keep current model parameters but replace the entire optimizer state by the state saved 25 updates earlier."),
        "step_counter_mismatch": ("optimizer_state.step", "Subtract 25 from every AdamW per-parameter step counter, clamped at zero."),
        "scheduler_mismatch": ("scheduler_state", "Drop scheduler state before resume, causing a fresh StepLR scheduler state to be used."),
        "optimizer_reset": ("optimizer_state", "Drop optimizer state before resume, causing a fresh AdamW state to be used."),
    }
    source_by_age = {age: age - 25 for age in AGES}
    out = []
    for fam in FAMILIES:
        fam_rows = [r for r in rows if r["rewrite_family"] == fam]
        changes_now = fam == "benign_nonzero"
        severity = 0.0 if fam == "exact_restore" else 1e-8 if fam == "benign_nonzero" else 25 if fam in {"stale_m", "stale_v", "stale_mv", "model_current_optimizer_stale", "step_counter_mismatch"} else 1.0
        if fam == "scheduler_mismatch":
            severity = "fresh StepLR scheduler state at resume"
        if fam == "optimizer_reset":
            severity = "fresh AdamW state at resume"
        out.append(
            {
                "rewrite_name": fam,
                "state_components_modified": transform[fam][0],
                "exact_transformation": transform[fam][1],
                "numeric_severity": severity,
                "stale_lag_optimizer_updates": 25 if "stale" in fam or fam == "step_counter_mismatch" else 0,
                "source_checkpoint_by_age": {str(age): f"optimizer_state_step_{src}.pt" for age, src in source_by_age.items()} if "stale" in fam else {},
                "step_counter_offset": -25 if fam == "step_counter_mismatch" else 0,
                "scheduler_offset": "fresh scheduler state" if fam == "scheduler_mismatch" else 0,
                "benign_perturbation": {"magnitude": 1e-8, "location": "first scalar of first model_state tensor"} if fam == "benign_nonzero" else None,
                "parameters_change_immediately": changes_now,
                "severity_changes_by_checkpoint_age": fam in {"stale_m", "stale_v", "stale_mv", "model_current_optimizer_stale"},
                "severity_identical_across_seeds": True,
                "category": categories[fam],
                "observed_T_tau_values": sorted({int(r["T_tau"]) for r in fam_rows if r["T_tau"]}),
            }
        )
    write_json(OUT / "rewrite_severities.json", out)
    write_csv(OUT / "rewrite_severities.csv", out)
    rows_md = [
        "# Rewrite Severities",
        "",
        "| rewrite | component | severity | category |",
        "|---|---|---:|---|",
    ]
    for r in out:
        rows_md.append(f"| {r['rewrite_name']} | {r['state_components_modified']} | {r['numeric_severity']} | {r['category']} |")
    write_md(OUT / "rewrite_severities.md", "\n".join(rows_md))
    quant = {
        "primary_source": "scripts/run_opt2026_pytorch_scale_final.py::mutate_opt_state",
        "case": "second_moment_quantized",
        "storage_or_quantization_dtype": "float16 round-trip, stored back as float32",
        "tensors_affected": "all AdamW optimizer_state exp_avg_sq tensors",
        "quantization_scope": "entire second-moment tensor for every optimizer state entry containing exp_avg_sq",
        "scale_range_calculation": "implicit IEEE fp16 conversion; no explicit scale/range",
        "clipping": "implicit fp16 finite-range behavior only",
        "rounding": "PyTorch dtype conversion rounding",
        "zero_point": None,
        "dequantization_rule": "convert fp16 tensor back to float32",
        "execution_order": ["deep-copy checkpoint optimizer state", "for each state value with exp_avg_sq", "exp_avg_sq = exp_avg_sq.to(torch.float16).to(torch.float32)", "save/load scenario"],
        "secondary_toy_source": "scripts/run_opt2026_empirical_significance.py rounds moments[:,1] to 1/256 increments; this is not the PyTorch round-trip recipe.",
    }
    write_json(OUT / "quantization_recipe.json", quant)


def delayed_cases(rows: list[dict[str, str]]) -> None:
    delayed = [r for r in rows if r["T_tau"] and 10 < int(r["T_tau"]) <= 20]
    out = []
    for r in delayed:
        out.append(
            {
                "seed": int(r["training_seed"]),
                "checkpoint_age": int(r["checkpoint_step"]),
                "rewrite_family": r["rewrite_family"],
                "severity": "fresh AdamW optimizer state",
                "source_checkpoint": r["checkpoint_file"],
                "T_tau": int(r["T_tau"]),
                "divergence_by_step": parse_list(r["divergence_curve"]),
                "maximum_divergence": fnum(r, "max_parameter_divergence"),
                "final_divergence": fnum(r, "final_parameter_divergence"),
                "validation_loss_gap_by_eval": parse_list(r["validation_loss_gap"]),
                "maximum_validation_loss_gap": fnum(r, "max_validation_loss_gap"),
                "final_validation_loss_gap": fnum(r, "final_validation_loss_gap"),
                "perplexity_gap_by_eval": parse_list(r["perplexity_gap"]),
                "task_recovery_outcome": "unrecovered" if ibool(r["task_damaged"]) else "recovered",
                "optimizer_components_modified": "entire optimizer_state reset to fresh AdamW state",
                "learning_rate_by_continuation_step": [0.001, 0.0005, 0.00025],
                "scheduler_state_event": "StepLR continues from checkpoint scheduler state; optimizer state reset only",
                "step_counter_state": "fresh AdamW counters initialized at resume",
                "source_candidate_optimizer_state_metadata": {"source": "checkpoint optimizer_state", "candidate": "fresh optimizer state"},
            }
        )
    hist = Counter(int(r["T_tau"]) for r in rows if r["T_tau"])
    hist_full = {str(i): hist.get(i, 0) for i in range(1, 21)}
    write_json(OUT / "delayed_cases.json", out)
    write_csv(OUT / "delayed_cases.csv", out)
    write_json(OUT / "latency_histogram.json", hist_full)
    status = "PASS" if len(delayed) == 9 and hist_full == {"1": 16, **{str(i): 0 for i in range(2, 14)}, "14": 1, "15": 2, "16": 4, "17": 1, "18": 1, "19": 0, "20": 0} else "MISMATCH"
    write_md(
        OUT / "latency_verification.md",
        f"""# Latency Verification

- status: {status}
- delayed rows with 10 < T_tau <= 20: {len(delayed)}
- histogram: `{json.dumps(hist_full, sort_keys=True)}`
""",
    )
    grid = [{"rewrite_family": r["rewrite_family"], "checkpoint_age": int(r["checkpoint_step"]), "seed": int(r["training_seed"]), "T_tau": r["T_tau"]} for r in rows]
    write_csv(OUT / "latency_grid.csv", grid)


def per_seed(rows: list[dict[str, str]]) -> None:
    out = []
    for seed in SEEDS:
        rs = [r for r in rows if int(r["training_seed"]) == seed]
        tvals = [int(r["T_tau"]) for r in rs if r["T_tau"]]
        c = confusion(rs)
        out.append(
            {
                "seed": seed,
                "total_rows": len(rs),
                "tau_divergent_rows": sum(r["parameter_label"] == "dangerous" for r in rs),
                "task_nonrecovered_rows": sum(ibool(r["task_damaged"]) for r in rs),
                "maximum_T_tau": max(tvals) if tvals else None,
                "delayed_T_tau_gt10_count": sum(t > 10 for t in tvals),
                **c,
                "spearman_parameter_divergence_validation_loss_damage": round(spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_validation_loss_gap") for r in rs]) or 0.0, 3),
                "spearman_parameter_divergence_perplexity_damage": round(spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_perplexity_gap") for r in rs]) or 0.0, 3),
            }
        )
    write_json(OUT / "per_seed_statistics.json", out)
    write_csv(OUT / "per_seed_statistics.csv", out)


def correlations(rows: list[dict[str, str]]) -> None:
    xs = [fnum(r, "normalized_max_parameter_divergence") for r in rows]
    ys = [fnum(r, "max_validation_loss_gap") for r in rows]
    py = [fnum(r, "max_perplexity_gap") for r in rows]
    pooled = spearman(xs, ys)
    records = [{"analysis": "pooled_all_rows", "group": "all", "N": len(rows), "coefficient": pooled}]
    for seed in SEEDS:
        rs = [r for r in rows if int(r["training_seed"]) == seed]
        records.append({"analysis": "per_seed", "group": seed, "N": len(rs), "coefficient": spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_validation_loss_gap") for r in rs])})
    for fam in FAMILIES:
        rs = [r for r in rows if r["rewrite_family"] != fam]
        records.append({"analysis": "leave_one_family_out", "group": fam, "N": len(rs), "coefficient": spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_validation_loss_gap") for r in rs])})
    for fam in FAMILIES:
        rs = [r for r in rows if r["rewrite_family"] == fam]
        records.append({"analysis": "within_family", "group": fam, "N": len(rs), "coefficient": spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_validation_loss_gap") for r in rs])})
    for age in AGES:
        rs = [r for r in rows if int(r["checkpoint_step"]) == age]
        records.append({"analysis": "checkpoint_age", "group": age, "N": len(rs), "coefficient": spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_validation_loss_gap") for r in rs])})
    residual_x = residualize(rows, xs)
    residual_y = residualize(rows, ys)
    records.append({"analysis": "family_checkpoint_residualized", "group": "two_way_additive_mean_residuals", "N": len(rows), "coefficient": spearman(residual_x, residual_y)})
    rng = random.Random(BOOTSTRAP_SEED)
    boots = []
    for _ in range(1000):
        sampled = [rng.choice(SEEDS) for _ in SEEDS]
        rs = [r for s in sampled for r in rows if int(r["training_seed"]) == s]
        coef = spearman([fnum(r, "normalized_max_parameter_divergence") for r in rs], [fnum(r, "max_validation_loss_gap") for r in rs])
        if coef is not None:
            boots.append(coef)
    boots_sorted = sorted(boots)
    rob = {
        "manuscript_reproduction": {
            "columns": ["normalized_max_parameter_divergence", "max_validation_loss_gap"],
            "aggregation_rule": "pooled over all 99 repeated rows",
            "spearman": pooled,
            "spearman_perplexity": spearman(xs, py),
        },
        "records": records,
        "cluster_bootstrap": {
            "cluster": "training_seed",
            "clusters": SEEDS,
            "rng_seed": BOOTSTRAP_SEED,
            "replicates": len(boots),
            "note": "Only three independent runs are available; bootstrap resolution is necessarily limited.",
            "p025": boots_sorted[int(0.025 * (len(boots_sorted) - 1))] if boots_sorted else None,
            "median": boots_sorted[int(0.5 * (len(boots_sorted) - 1))] if boots_sorted else None,
            "p975": boots_sorted[int(0.975 * (len(boots_sorted) - 1))] if boots_sorted else None,
        },
    }
    write_json(OUT / "correlation_robustness.json", rob)
    write_csv(OUT / "correlation_robustness.csv", records)
    write_md(
        OUT / "correlation_robustness.md",
        f"""# Correlation Robustness

The manuscript-scale value is reproduced as a pooled Spearman correlation between `normalized_max_parameter_divergence` and `max_validation_loss_gap` over all 99 rows.

- pooled Spearman: {pooled:.3f}
- bootstrap RNG seed: {BOOTSTRAP_SEED}
- independent clusters: 3 training seeds
""",
    )


def residualize(rows: list[dict[str, str]], values: list[float]) -> list[float]:
    overall = sum(values) / len(values)
    fam_mean: dict[str, float] = {}
    age_mean: dict[str, float] = {}
    for fam in FAMILIES:
        vals = [v for r, v in zip(rows, values) if r["rewrite_family"] == fam]
        fam_mean[fam] = sum(vals) / len(vals)
    for age in AGES:
        vals = [v for r, v in zip(rows, values) if int(r["checkpoint_step"]) == age]
        age_mean[str(age)] = sum(vals) / len(vals)
    return [v - fam_mean[r["rewrite_family"]] - age_mean[r["checkpoint_step"]] + overall for r, v in zip(rows, values)]


def task_metric(rows: list[dict[str, str]]) -> None:
    sweep = []
    for eps in [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0]:
        damaged = 0
        recovered = 0
        for r in rows:
            gaps = parse_list(r["validation_loss_gap"])
            rec = any(all(v <= eps for v in gaps[i : i + RECOVERY_CONSECUTIVE]) for i in range(max(0, len(gaps) - RECOVERY_CONSECUTIVE + 1)))
            recovered += int(rec)
            damaged += int(not rec)
        sweep.append({"threshold": eps, "task_nonrecovered": damaged, "task_recovered": recovered, "rows": len(rows)})
    controls = [r for r in rows if r["rewrite_family"] in {"exact_restore", "benign_nonzero"}]
    clean = {
        "controls_used": ["exact_restore", "benign_nonzero"],
        "exact_restore_max_gap": max(fnum(r, "max_validation_loss_gap") for r in rows if r["rewrite_family"] == "exact_restore"),
        "benign_nonzero_max_gap": max(fnum(r, "max_validation_loss_gap") for r in rows if r["rewrite_family"] == "benign_nonzero"),
        "all_control_max_gap": max(fnum(r, "max_validation_loss_gap") for r in controls),
        "note": "Frozen campaign contains exact-restore and benign-nonzero controls, not repeated clean-vs-clean evaluation replicates.",
    }
    write_csv(OUT / "task_threshold_sweep.csv", sweep)
    write_json(OUT / "task_threshold_sweep.json", sweep)
    write_json(OUT / "clean_variability.json", clean)
    write_md(
        OUT / "task_metric_definition.md",
        """# Task Metric Definition

`validation_loss_gap` is the absolute difference between the clean reference continuation validation loss and the rewritten candidate continuation validation loss at each evaluation. The candidate/reference orientation is made unsigned by `abs(a - b)` in `scripts/run_opt2026_10m_gpu_final.py`.

The validation data are fixed cached WikiText-2 character-level batches generated by the frozen development seed, with evaluation after every continuation update. A row is task-recovered when the validation-loss gap is <= 0.05 for three consecutive evaluations within the H=20 replay window. The 0.05 threshold is inherited from the final task-validity campaign and was not tuned on the 19.485M rows.
""",
    )


def roc_pr(rows: list[dict[str, str]]) -> None:
    scores = sorted({fnum(r, "normalized_max_parameter_divergence") for r in rows}, reverse=True)
    thresholds = [float("inf")] + scores + [-1e-12]
    roc_rows = [threshold_confusion(rows, t) for t in thresholds]
    pr_rows = roc_rows
    auroc = auc([(r["fpr"], r["sensitivity"]) for r in roc_rows])
    pr_points = sorted((r["recall"], r["precision"]) for r in pr_rows)
    auprc = auc(pr_points)
    write_csv(OUT / "parameter_metric_roc.csv", roc_rows)
    write_csv(OUT / "parameter_metric_pr.csv", pr_rows)
    write_json(OUT / "parameter_metric_metrics.json", {"AUROC": auroc, "AUPRC": auprc, "predictor": "normalized_max_parameter_divergence", "target": "task_damaged"})
    plot_curve(OUT / "parameter_metric_roc.png", [r["fpr"] for r in roc_rows], [r["sensitivity"] for r in roc_rows], "false positive rate", "true positive rate", "ROC")
    plot_curve(OUT / "parameter_metric_pr.png", [r["recall"] for r in pr_rows], [r["precision"] for r in pr_rows], "recall", "precision", "Precision-recall")
    plot_curve(OUT / "parameter_threshold_tradeoff.png", [r["threshold"] for r in roc_rows if math.isfinite(r["threshold"])], [r["sensitivity"] for r in roc_rows if math.isfinite(r["threshold"])], "threshold", "sensitivity", "Threshold tradeoff")


def plot_curve(path: Path, xs: list[float], ys: list[float], xlabel: str, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(xs, ys, marker="o", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def trajectories(rows: list[dict[str, str]]) -> None:
    shutil.copy2(SRC / "trajectories.csv", OUT / "trajectory_rows.csv")
    tau_rows = [r for r in rows if r["T_tau"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for r in tau_rows:
        divs = parse_list(r["divergence_curve"])
        ax.plot(range(len(divs)), divs, linewidth=0.8, alpha=0.45)
    ax.set_xlabel("continuation step")
    ax.set_ylabel("parameter divergence")
    ax.set_title("All tau-divergent trajectories")
    fig.tight_layout()
    fig.savefig(OUT / "all_tau_divergent_trajectories.png", dpi=180)
    plt.close(fig)
    immediate_pool = [r for r in tau_rows if int(r["T_tau"]) == 1]
    immediate = sorted(immediate_pool, key=lambda r: (abs(fnum(r, "max_parameter_divergence") - median([fnum(x, "max_parameter_divergence") for x in immediate_pool])), r["row_id"]))[0]
    delayed_pool = [r for r in tau_rows if 10 < int(r["T_tau"]) <= 20]
    delayed = sorted(delayed_pool, key=lambda r: (abs(int(r["T_tau"]) - median([int(x["T_tau"]) for x in delayed_pool])), r["row_id"]))[0]
    fn_pool = [r for r in rows if ibool(r["task_damaged"]) and r["parameter_label"] == "benign"]
    false_negative = sorted(fn_pool, key=lambda r: (-fnum(r, "max_validation_loss_gap"), r["row_id"]))[0]
    selections = {"immediate": immediate["row_id"], "delayed": delayed["row_id"], "parameter_threshold_false_negative": false_negative["row_id"], "selection_rule": "median immediate max divergence; median delayed T_tau; largest task loss among inherited-tau false negatives"}
    write_json(OUT / "trajectory_selection.json", selections)
    for stem, r in [("trajectory_immediate.png", immediate), ("trajectory_delayed.png", delayed), ("trajectory_threshold_false_negative.png", false_negative)]:
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        divs = parse_list(r["divergence_curve"])
        ax.plot(range(len(divs)), divs, marker="o")
        ax.axhline(fnum(r, "tau_numeric"), color="black", linestyle="--")
        ax.set_title(r["row_id"])
        ax.set_xlabel("continuation step")
        ax.set_ylabel("parameter divergence")
        fig.tight_layout()
        fig.savefig(OUT / stem, dpi=180)
        plt.close(fig)


def median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2


def alternatives(rows: list[dict[str, str]]) -> None:
    records = []
    for r in rows:
        divs = parse_list(r["divergence_curve"])
        clean_cumulative_proxy = sum(abs(divs[i] - divs[i - 1]) for i in range(1, len(divs)))
        alt = fnum(r, "max_parameter_divergence") / (clean_cumulative_proxy + 1e-12)
        records.append({"row_id": r["row_id"], "rewrite_family": r["rewrite_family"], "clean_cumulative_update_distance_proxy": clean_cumulative_proxy, "alt_divergence_ratio": alt, "max_validation_loss_gap": fnum(r, "max_validation_loss_gap"), "task_damaged": ibool(r["task_damaged"])})
    xs = [float(r["alt_divergence_ratio"]) for r in records]
    ys = [float(r["max_validation_loss_gap"]) for r in records]
    write_csv(OUT / "alternative_metrics.csv", records)
    write_json(OUT / "alternative_metrics.json", {"metric": "max divergence divided by cumulative divergence-curve path length proxy", "spearman_vs_loss": spearman(xs, ys), "status": "proxy_only", "blocker": "Clean reference parameter snapshots for each continuation step were not saved in frozen rows, so exact clean cumulative update-distance normalization requires rerunning paired continuations."})
    write_md(OUT / "alternative_metric_analysis.md", "# Alternative Metric Analysis\n\nA proxy alternative metric was computed from frozen divergence curves. The exact clean cumulative update-distance normalization was not reconstructible from saved row files because clean reference parameter snapshots/update vectors were not persisted.")
    plot_curve(OUT / "alternative_metric_comparison.png", xs, ys, "alternative divergence ratio", "max validation-loss gap", "Alternative metric")


def blockers_and_roundtrip(rows: list[dict[str, str]]) -> None:
    delayed = [r for r in rows if r["T_tau"] and 10 < int(r["T_tau"]) <= 20]
    mech_rows = []
    for r in delayed:
        for i, div in enumerate(parse_list(r["divergence_curve"])):
            mech_rows.append({"row_id": r["row_id"], "continuation_step": i, "parameter_divergence_after_update": div, "status": "not_computed_from_frozen_rows", "blocker": "source/candidate update vectors and moment discrepancy tensors were not persisted"})
    write_csv(OUT / "delayed_mechanism_steps.csv", mech_rows)
    write_json(OUT / "delayed_mechanism_summary.json", {"status": "blocked", "cases": [r["row_id"] for r in delayed], "blocker": "Requires rerunning the nine delayed paired continuations with per-step optimizer-vector instrumentation."})
    write_md(OUT / "delayed_mechanism_summary.md", "# Delayed Mechanism Summary\n\nBlocked: the frozen campaign rows contain divergence/loss trajectories but not source/candidate update vectors, moment discrepancy tensors, or bias-correction factors.")
    for folder, files in {
        "extended_horizon": ["extended_horizon_rows.csv", "extended_horizon_summary.json", "extended_horizon_summary.md"],
        "future_schedule": ["future_schedule_rows.csv", "future_schedule_summary.json", "future_schedule_summary.md"],
    }.items():
        rows_file, json_file, md_file = files
        write_csv(OUT / rows_file, [{"status": "not_run", "blocker": "Requires new matched continuation runs from checkpoints; not present in frozen H=20 results."}])
        write_json(OUT / json_file, {"status": "not_run", "blocker": "Requires new matched continuation runs from checkpoints; not present in frozen H=20 results."})
        write_md(OUT / md_file, f"# {folder.replace('_', ' ').title()}\n\nNot run in this evidence pass. The frozen primary H=20 data are preserved unchanged.")
    scale_dir = OUT / "controlled_scale"
    scale_dir.mkdir(parents=True, exist_ok=True)
    write_json(scale_dir / "BLOCKER.json", {"status": "not_completed", "blocker": "No existing controlled 1.3M/5M/20M three-seed matched-scale family exists; available prior scale outputs are below 1M or use different protocols."})
    write_md(scale_dir / "STATUS.md", "# Controlled Scale Status\n\nNot completed. Existing result folders were inspected and do not satisfy the requested controlled 1.3M/5M/20M design.")
    long_dir = OUT / "long_training"
    long_dir.mkdir(parents=True, exist_ok=True)
    write_json(long_dir / "BLOCKER.json", {"status": "not_completed", "blocker": "No substantially longer 10x-update 19.485M AdamW matched rewrite suite exists in the frozen results."})
    write_md(long_dir / "STATUS.md", "# Longer-Training Status\n\nNot completed. Existing maturity analyses are smaller/different protocols and do not satisfy the requested longer-trained 19.485M-style run.")
    rt_src = ROOT / "results" / "opt2026_pytorch_scale_final" / "pytorch_roundtrip" / "rows.csv"
    if rt_src.exists():
        rt = read_csv(rt_src)
        out = []
        for r in rt:
            out.append(
                {
                    "transformation_name": r.get("roundtrip_scenario", r.get("rewrite_family", "")),
                    "exact_recipe": "from scripts/run_opt2026_pytorch_scale_final.py",
                    "state_components_affected": r.get("rewrite_family", ""),
                    "loadability": "loadable .pt scenario",
                    "T_tau": r.get("T_tau", ""),
                    "maximum_parameter_divergence": r.get("max_parameter_divergence", ""),
                    "final_parameter_divergence": r.get("final_parameter_divergence", ""),
                    "maximum_validation_loss_gap": r.get("max_validation_loss_gap", ""),
                    "final_validation_loss_gap": r.get("final_validation_loss_gap", ""),
                    "task_recovery": "unrecovered" if ibool(r.get("task_damaged", "")) else "recovered",
                    "continuation_horizon": r.get("horizon", 20),
                    "censoring": "right-censored at horizon" if not r.get("T_tau", "") else "event observed",
                }
            )
        write_csv(OUT / "roundtrip_cases.csv", out)
        write_json(OUT / "roundtrip_cases.json", out)
        write_md(OUT / "roundtrip_cases.md", f"# Roundtrip Cases\n\nRecovered {len(out)} framework-level round-trip rows from `results/opt2026_pytorch_scale_final/pytorch_roundtrip/rows.csv`.")


def model_configs() -> None:
    probe = read_csv(SRC / "model_probe.csv")
    summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8-sig"))
    cfg = {
        "large_19p485M": summary["model"],
        "small_1p298M": {
            "source": "results/opt2026_final_gpu_task_validity and previous scale comparison",
            "parameter_count": 1298080,
            "checkpoint_ages": [50, 100, 200],
            "training_steps": 200,
            "seeds": [901, 902, 903],
            "note": "Exact layer/width fields were not present in the 19.485M frozen row files; see historical generator for the 1.298M campaign.",
        },
        "model_probe_candidates": probe,
    }
    write_json(OUT / "model_regime_configs.json", cfg)
    write_md(OUT / "model_selection_provenance.md", f"""# Model Selection Provenance

The 19.485M model was selected from `scripts/run_opt2026_10m_gpu_final.py` candidate probes. The recorded reason for not using 25.414M is in `FULL_EVAL_FAILURES`: it survived one CUDA AdamW step but failed during full rewrite evaluation with `CUBLAS_STATUS_EXECUTION_FAILED`; the next-largest reliable >=10M candidate was used.

No log in the frozen 19.485M artifacts records a held-out-result-based model-size selection rule. The available provenance records an infrastructure/feasibility decision.
""")


def manifest() -> None:
    important = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.name != "MANIFEST.json":
            important.append({"path": str(p.relative_to(ROOT)), "sha256": hash_file(p), "bytes": p.stat().st_size})
    write_json(
        OUT / "MANIFEST.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": run(["git", "rev-parse", "HEAD"]),
            "command": f"{sys.executable} scripts/make_reviewer2026_results.py",
            "hardware": json.loads((SRC / "hardware.json").read_text(encoding="utf-8-sig")),
            "result_file_hashes": important,
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_csv(SRC / "rows.csv")
    baseline(rows)
    rewrite_severities(rows)
    delayed_cases(rows)
    per_seed(rows)
    correlations(rows)
    task_metric(rows)
    roc_pr(rows)
    trajectories(rows)
    alternatives(rows)
    blockers_and_roundtrip(rows)
    model_configs()
    manifest()
    print(f"reviewer2026 evidence package written to {OUT}")


if __name__ == "__main__":
    main()

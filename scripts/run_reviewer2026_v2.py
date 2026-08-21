from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments" / "scripts"), str(ROOT / "experiments")]
import run_opt2026_10m_gpu_final as base  # noqa: E402

OUT = ROOT / "results" / "reviewer2026_v2"
SRC = ROOT / "results" / "opt2026_10m_gpu_final"
V1 = ROOT / "results" / "reviewer2026"
SOURCE_COMMIT = "0a00294f29dd50a913aba2cafc0b678ff9751149"

H = 100
H20 = 20
TAU_FRAC = 0.10
EPS = 1e-12
TASK_THRESHOLDS = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0]
RECOVERY_CONSECUTIVE = 3
ROC_BOOTSTRAP_SEED = 20260822
FUTURE_SCHEDULE_SEEDS = [8101, 8102, 8103, 8104, 8105]
SCALE_SEEDS = [1201, 1202, 1203]
LONG_SEEDS = [1301, 1302, 1303]
SCALE_CONFIGS = [
    {"name": "width128_1p63m", "num_layers": 8, "d_model": 128, "nhead": 8, "dim_feedforward": 512, "seq_len": 127, "dropout": 0.0},
    {"name": "width224_4p93m", "num_layers": 8, "d_model": 224, "nhead": 8, "dim_feedforward": 896, "seq_len": 127, "dropout": 0.0},
    {"name": "width448_19p48m", "num_layers": 8, "d_model": 448, "nhead": 8, "dim_feedforward": 1792, "seq_len": 127, "dropout": 0.0},
]


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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if torch.is_tensor(x):
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_horizon() -> None:
    base.HORIZON = H


def l2_from_params(model_a: torch.nn.Module, model_b: torch.nn.Module) -> float:
    total = torch.zeros((), device=base.DEVICE)
    with torch.no_grad():
        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            total = total + (pa.detach().float() - pb.detach().float()).pow(2).sum()
    return float(torch.sqrt(total).detach().cpu().item())


def norm_model(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=base.DEVICE)
    with torch.no_grad():
        for p in model.parameters():
            total = total + p.detach().float().pow(2).sum()
    return float(torch.sqrt(total).detach().cpu().item())


def flat_gpu(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().reshape(-1) for p in model.parameters()])


def layer_stats(model_a: torch.nn.Module, model_b: torch.nn.Module, prev_a: dict[str, torch.Tensor] | None = None, prev_b: dict[str, torch.Tensor] | None = None) -> dict[str, Any]:
    vals = []
    upd_vals = []
    with torch.no_grad():
        for (name_a, pa), (name_b, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert name_a == name_b
            diff = (pa.detach().float() - pb.detach().float()).norm().item()
            denom = pa.detach().float().norm().item() + EPS
            vals.append(diff / denom)
            if prev_a is not None and prev_b is not None:
                ua = pa.detach().float() - prev_a[name_a].to(pa.device)
                ub = pb.detach().float() - prev_b[name_b].to(pb.device)
                upd_vals.append((ua - ub).norm().item())
    return {
        "mean_layer_relative_displacement": float(np.mean(vals)) if vals else 0.0,
        "max_layer_relative_displacement": float(np.max(vals)) if vals else 0.0,
        "max_layer_update_divergence": float(np.max(upd_vals)) if upd_vals else 0.0,
    }


def snapshot_named(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()}


def validation_loss(model: torch.nn.Module, batch: Any) -> float:
    return base.loss_only(model, batch)


def logits_for(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    x, _ = base.to_device_batch(batch)
    with torch.no_grad():
        return model(x).detach().float()


def function_metrics(model_a: torch.nn.Module, model_b: torch.nn.Module, batch: Any) -> dict[str, float]:
    la = logits_for(model_a, batch)
    lb = logits_for(model_b, batch)
    logpa = F.log_softmax(la, dim=-1)
    logpb = F.log_softmax(lb, dim=-1)
    pa = logpa.exp()
    kl = (pa * (logpa - logpb)).sum(dim=-1).mean()
    rms = (la - lb).pow(2).mean().sqrt()
    return {"calibration_kl": float(kl.cpu().item()), "calibration_rms_logit": float(rms.cpu().item())}


def optimizer_moment_stats(opt_a: torch.optim.Optimizer, opt_b: torch.optim.Optimizer) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, prefix in [("exp_avg", "m"), ("exp_avg_sq", "v")]:
        clean_sq = reset_sq = diff_sq = dot = torch.zeros((), device=base.DEVICE)
        for group_a, group_b in zip(opt_a.param_groups, opt_b.param_groups):
            for pa, pb in zip(group_a["params"], group_b["params"]):
                sa = opt_a.state.get(pa, {})
                sb = opt_b.state.get(pb, {})
                if key not in sa or key not in sb:
                    continue
                a = sa[key].detach().float()
                b = sb[key].detach().float()
                clean_sq = clean_sq + a.pow(2).sum()
                reset_sq = reset_sq + b.pow(2).sum()
                diff_sq = diff_sq + (a - b).pow(2).sum()
                dot = dot + (a * b).sum()
        clean = float(torch.sqrt(clean_sq).cpu().item())
        reset = float(torch.sqrt(reset_sq).cpu().item())
        diff = float(torch.sqrt(diff_sq).cpu().item())
        out[f"{prefix}_clean_norm"] = clean
        out[f"{prefix}_candidate_norm"] = reset
        out[f"{prefix}_difference_norm"] = diff
        out[f"{prefix}_relative_difference"] = diff / (clean + EPS)
        out[f"{prefix}_cosine"] = float((dot / (torch.sqrt(clean_sq) * torch.sqrt(reset_sq) + EPS)).cpu().item()) if clean > 0 and reset > 0 else math.nan
    return out


def optimizer_step_value(opt: torch.optim.Optimizer) -> int:
    vals = []
    for state in opt.state.values():
        if "step" in state:
            step = state["step"]
            vals.append(int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step))
    return max(vals) if vals else 0


def first_crossing(values: list[float], tau: float) -> int | None:
    for i, value in enumerate(values):
        if value >= tau:
            return i
    return None


def recovery_step(gaps: list[float], eps: float = 0.05) -> int | None:
    for i in range(len(gaps) - RECOVERY_CONSECUTIVE + 1):
        if all(v <= eps for v in gaps[i : i + RECOVERY_CONSECUTIVE]):
            return i + 1
    return None


def run_pair(base_payload: dict[str, Any], family: str, stale: dict[str, Any] | None, batches: list[Any], val_batches: list[Any], calibration_batch: Any, future_start: int | None = None, instrument: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scen = base.scenario_payload(base_payload, family, stale)
    load_opt = family != "optimizer_reset"
    load_sched = family != "scheduler_mismatch"
    clean_model, clean_opt, clean_sched = base.fresh_from_payload(base_payload)
    cand_model, cand_opt, cand_sched = base.fresh_from_payload(scen, load_optimizer=load_opt, load_scheduler=load_sched)
    start = int(base_payload["batch_pos"]) if future_start is None else future_start
    theta_norm = norm_model(clean_model)
    tau = TAU_FRAC * (theta_norm + EPS)
    cum_clean = 0.0
    cum_candidate = 0.0
    cum_update_discrepancy = 0.0
    step_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []

    div0 = l2_from_params(clean_model, cand_model)
    layers0 = layer_stats(clean_model, cand_model)
    f0 = function_metrics(clean_model, cand_model, calibration_batch)
    divergences = [div0]
    norm_divergences = [div0 / (theta_norm + EPS)]
    max_update_norm_metric = [0.0]
    max_layer_metric = [layers0["max_layer_relative_displacement"]]
    mean_layer_metric = [layers0["mean_layer_relative_displacement"]]
    kl_metric = [f0["calibration_kl"]]
    rms_metric = [f0["calibration_rms_logit"]]
    val_clean: list[float] = []
    val_candidate: list[float] = []
    val_gaps: list[float] = []
    ppl_gaps: list[float] = []
    lrs: list[float] = []

    step_rows.append({
        "continuation_step": 0,
        "parameter_divergence": div0,
        "normalized_parameter_divergence": div0 / (theta_norm + EPS),
        "validation_loss_gap": 0.0,
        "perplexity_gap": 0.0,
        "lr": clean_opt.param_groups[0]["lr"],
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
        prev_clean = snapshot_named(clean_model)
        prev_candidate = snapshot_named(cand_model)
        prev_clean_flat = flat_gpu(clean_model)
        prev_candidate_flat = flat_gpu(cand_model)
        clean_loss = base.train_one(clean_model, clean_opt, clean_sched, batch)
        candidate_loss = base.train_one(cand_model, cand_opt, cand_sched, batch)
        clean_flat = flat_gpu(clean_model)
        candidate_flat = flat_gpu(cand_model)
        clean_update = clean_flat - prev_clean_flat
        candidate_update = candidate_flat - prev_candidate_flat
        clean_update_norm = float(clean_update.norm().detach().cpu().item())
        candidate_update_norm = float(candidate_update.norm().detach().cpu().item())
        delta_update = clean_update - candidate_update
        delta_update_norm = float(delta_update.norm().detach().cpu().item())
        update_cos = float((torch.dot(clean_update, candidate_update) / (clean_update.norm() * candidate_update.norm() + EPS)).detach().cpu().item()) if clean_update_norm > 0 and candidate_update_norm > 0 else math.nan
        cum_clean += clean_update_norm
        cum_candidate += candidate_update_norm
        cum_update_discrepancy += delta_update_norm
        del prev_clean_flat, prev_candidate_flat, clean_flat, candidate_flat, clean_update, candidate_update, delta_update

        div = l2_from_params(clean_model, cand_model)
        v_clean = sum(validation_loss(clean_model, b) for b in val_batches) / len(val_batches)
        v_candidate = sum(validation_loss(cand_model, b) for b in val_batches) / len(val_batches)
        gap = abs(v_clean - v_candidate)
        ppl_gap = abs(math.exp(min(v_clean, 50)) - math.exp(min(v_candidate, 50)))
        layers = layer_stats(clean_model, cand_model, prev_clean, prev_candidate)
        fmetrics = function_metrics(clean_model, cand_model, calibration_batch)
        lr = clean_opt.param_groups[0]["lr"]

        divergences.append(div)
        norm_divergences.append(div / (theta_norm + EPS))
        val_clean.append(v_clean)
        val_candidate.append(v_candidate)
        val_gaps.append(gap)
        ppl_gaps.append(ppl_gap)
        lrs.append(lr)
        max_update_norm_metric.append(div / (cum_clean + EPS))
        max_layer_metric.append(layers["max_layer_relative_displacement"])
        mean_layer_metric.append(layers["mean_layer_relative_displacement"])
        kl_metric.append(fmetrics["calibration_kl"])
        rms_metric.append(fmetrics["calibration_rms_logit"])

        row = {
            "continuation_step": k,
            "parameter_divergence": div,
            "normalized_parameter_divergence": div / (theta_norm + EPS),
            "validation_loss_gap": gap,
            "perplexity_gap": ppl_gap,
            "lr": lr,
            "source_task_loss": v_clean,
            "candidate_task_loss": v_candidate,
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
        }
        step_rows.append(row)

        if instrument:
            moments = optimizer_moment_stats(clean_opt, cand_opt)
            clean_step = optimizer_step_value(clean_opt)
            cand_step = optimizer_step_value(cand_opt)
            beta1, beta2 = base.BETAS
            mechanism_rows.append({
                **row,
                "theta_clean_norm": norm_model(clean_model),
                "theta_candidate_norm": norm_model(cand_model),
                "train_loss_clean": clean_loss,
                "train_loss_candidate": candidate_loss,
                "clean_optimizer_step_counter": clean_step,
                "candidate_optimizer_step_counter": cand_step,
                "clean_bias_correction1": 1.0 - beta1 ** clean_step,
                "clean_bias_correction2": 1.0 - beta2 ** clean_step,
                "candidate_bias_correction1": 1.0 - beta1 ** cand_step,
                "candidate_bias_correction2": 1.0 - beta2 ** cand_step,
                "scheduler_last_epoch_clean": getattr(clean_sched, "last_epoch", ""),
                "scheduler_last_epoch_candidate": getattr(cand_sched, "last_epoch", ""),
                "scheduler_transition_indicator": int(k in {25, 50, 75, 100}),
                "weight_decay": base.WEIGHT_DECAY,
                **moments,
            })
        gc.collect()

    t_tau = first_crossing(divergences, tau)
    rec = recovery_step(val_gaps, 0.05)
    summary = {
        "theta_checkpoint_norm": theta_norm,
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
        "task_recovery_step": rec,
        "task_nonrecovered_h100": rec is None,
        "task_nonrecovered_h20": recovery_step(val_gaps[:H20], 0.05) is None,
        "max_validation_loss_gap": max(val_gaps) if val_gaps else 0.0,
        "final_validation_loss_gap": val_gaps[-1] if val_gaps else 0.0,
        "max_perplexity_gap": max(ppl_gaps) if ppl_gaps else 0.0,
        "final_perplexity_gap": ppl_gaps[-1] if ppl_gaps else 0.0,
        "max_clean_update_normalized_divergence": max(max_update_norm_metric),
        "final_clean_update_normalized_divergence": max_update_norm_metric[-1],
        "max_layer_relative_displacement": max(max_layer_metric),
        "mean_layer_relative_displacement_max": max(mean_layer_metric),
        "max_calibration_kl": max(kl_metric),
        "final_calibration_kl": kl_metric[-1],
        "max_calibration_rms_logit": max(rms_metric),
        "final_calibration_rms_logit": rms_metric[-1],
    }
    del clean_model, cand_model, clean_opt, cand_opt, clean_sched, cand_sched
    torch.cuda.empty_cache()
    return summary, step_rows, mechanism_rows


def original_rows() -> list[dict[str, str]]:
    return read_csv(SRC / "rows.csv")


def delayed_case_keys() -> list[tuple[int, int]]:
    rows = original_rows()
    return sorted((int(r["training_seed"]), int(r["checkpoint_step"])) for r in rows if r["rewrite_family"] == "optimizer_reset" and r["T_tau"] and 10 < int(r["T_tau"]) <= 20)


def verify_original_delayed() -> dict[str, Any]:
    want = {(1101, 75): 15, (1101, 150): 16, (1101, 300): 18, (1102, 75): 15, (1102, 150): 16, (1102, 300): 17, (1103, 75): 14, (1103, 150): 16, (1103, 300): 16}
    got = {(int(r["training_seed"]), int(r["checkpoint_step"])): int(r["T_tau"]) for r in original_rows() if r["rewrite_family"] == "optimizer_reset"}
    ok = got == want
    write_json(OUT / "original_h20_delayed_verification.json", {"status": "PASS" if ok else "FAIL", "expected": want, "actual": got})
    if not ok:
        raise RuntimeError(f"original H20 delayed matrix mismatch: {got}")
    return {"status": "PASS", "actual": got}


def provenance() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "PROVENANCE.json", {
        "source_commit": SOURCE_COMMIT,
        "starting_branch": run_cmd(["git", "branch", "--show-current"]),
        "starting_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "dirty_status": run_cmd(["git", "status", "--short"]),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "horizon": H,
        "future_schedule_seeds": FUTURE_SCHEDULE_SEEDS,
        "scale_seeds": SCALE_SEEDS,
        "long_training_seeds": LONG_SEEDS,
    })


def validation_protocol() -> None:
    rows = original_rows()
    batches, vocab, dataset = base.load_batches(base.DEV_SEED, base.TRAIN_STEPS + H + 50)
    val_indices = list(range(base.TRAIN_STEPS + 1, base.TRAIN_STEPS + 4))
    protocol = {
        "dataset": dataset,
        "corpus_path": str((ROOT / "experiments" / "data" / "wikitext2_raw" / "train.txt").relative_to(ROOT)),
        "corpus_characters_read": base.DATA_CHARS,
        "tokenization": "character-level vocabulary built by run_transformer.build_char_dataset",
        "vocab_size": vocab,
        "sequence_length_config": base.SEQ_LEN,
        "model_sequence_length": base.MODEL_CFG["seq_len"] + 1,
        "batch_size": base.BATCH_SIZE,
        "validation_batch_indices": val_indices,
        "validation_batches": len(val_indices),
        "validation_sequences": len(val_indices) * base.BATCH_SIZE,
        "validation_tokens": len(val_indices) * base.BATCH_SIZE * base.SEQ_LEN,
        "evaluation_cadence": "after every continuation optimizer update",
        "batches_cached_fixed": True,
        "batch_rng_seed_for_main_h100": "same held-out training seed as row",
        "dev_seed_for_protocol_probe": base.DEV_SEED,
        "candidate_reference_share_evaluation_batch": True,
        "evaluation_deterministic": True,
        "validation_loss_gap_formula": "abs(validation_loss_clean - validation_loss_candidate)",
        "perplexity_gap_formula": "abs(exp(min(loss_clean, 50)) - exp(min(loss_candidate, 50)))",
        "original_rows_checked": len(rows),
    }
    write_json(OUT / "validation_protocol.json", protocol)
    write_md(OUT / "validation_protocol.md", f"""# Validation Protocol

- Corpus: `{protocol['corpus_path']}`
- Characters used: {protocol['corpus_characters_read']}
- Tokenizer: character-level, vocabulary size {vocab}
- Validation batches: indices {val_indices}, {protocol['validation_batches']} batches, {protocol['validation_tokens']} tokens total
- Sequence length: {base.SEQ_LEN}; batch size: {base.BATCH_SIZE}
- Evaluation cadence: after every continuation optimizer update
- Candidate and clean continuations use identical validation batches.
- Validation-loss gap: `abs(validation_loss_clean - validation_loss_candidate)`.
""")


def run_h100() -> None:
    set_horizon()
    rows_out: list[dict[str, Any]] = []
    step_out: list[dict[str, Any]] = []
    old = {r["row_id"]: r for r in original_rows()}
    for seed in base.HELDOUT_SEEDS:
        batches, vocab, dataset = base.load_batches(seed, base.TRAIN_STEPS + H + 50)
        val_batches = batches[base.TRAIN_STEPS + 1 : base.TRAIN_STEPS + 4]
        calibration_batch = batches[base.TRAIN_STEPS + 4]
        for age in base.CHECKPOINT_AGES:
            base_file = SRC / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"checkpoint_step_{age}.pt"
            stale_file = SRC / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"optimizer_state_step_{age - base.STALE_AGE}.pt"
            payload = base.load_checkpoint(base_file)
            stale = base.load_checkpoint(stale_file)["optimizer_state"] if stale_file.exists() else None
            for fam in base.REWRITE_FAMILIES:
                row_id = f"gpu_seed{seed}_ckpt{age}_{fam}"
                t0 = time.perf_counter()
                summary, steps, _ = run_pair(payload, fam, stale, batches, val_batches, calibration_batch)
                summary.update({
                    "row_id": row_id,
                    "experiment": "reviewer2026_v2_h100",
                    "seed": seed,
                    "checkpoint_age": age,
                    "rewrite_family": fam,
                    "dataset": dataset,
                    "checkpoint_file": str(base_file.relative_to(ROOT)),
                    "stale_source_file": str(stale_file.relative_to(ROOT)) if stale_file.exists() and "stale" in fam else "",
                    "runtime_seconds": time.perf_counter() - t0,
                    "original_h20_T_tau": old[row_id]["T_tau"],
                    "original_h20_task_damaged": old[row_id]["task_damaged"],
                    "crossed_after_20": bool(summary["T_tau"] is not None and summary["T_tau"] > 20),
                })
                rows_out.append(summary)
                for s in steps:
                    step_out.append({"row_id": row_id, "seed": seed, "checkpoint_age": age, "rewrite_family": fam, **s})
                write_csv(OUT / "h100_rows.csv", rows_out)
                write_json(OUT / "h100_rows.json", rows_out)
                write_csv(OUT / "h100_trajectory_steps.csv", step_out)
                gc.collect()
    summarize_h100(rows_out)


def summarize_h100(rows: list[dict[str, Any]] | None = None) -> None:
    if rows is None:
        rows = json.loads((OUT / "h100_rows.json").read_text(encoding="utf-8"))
    hist = Counter(str(r["T_tau"]) for r in rows if r["T_tau"] is not None)
    hist_full = {str(i): hist.get(str(i), 0) for i in range(0, H + 1)}
    write_json(OUT / "h100_latency_histogram.json", hist_full)
    family_rows = []
    for fam in base.REWRITE_FAMILIES:
        rs = [r for r in rows if r["rewrite_family"] == fam]
        tvals = [r["T_tau"] for r in rs if r["T_tau"] is not None]
        family_rows.append({
            "rewrite_family": fam,
            "rows": len(rs),
            "crossed_h100": len(tvals),
            "crossed_after_20": sum(bool(r["crossed_after_20"]) for r in rs),
            "task_nonrecovered_h20": sum(bool(r["task_nonrecovered_h20"]) for r in rs),
            "task_nonrecovered_h100": sum(bool(r["task_nonrecovered_h100"]) for r in rs),
            "median_T_tau": float(np.median(tvals)) if tvals else "",
            "max_T_tau": max(tvals) if tvals else "",
        })
    write_csv(OUT / "h100_family_summary.csv", family_rows)
    old_noncross = [r for r in rows if r["original_h20_T_tau"] in ("", None)]
    old_task = [r for r in rows if str(r["original_h20_task_damaged"]).lower() == "true"]
    summary = {
        "rows": len(rows),
        "crossed_h100": sum(r["T_tau"] is not None for r in rows),
        "crossed_after_20_from_h20_noncrossers": sum(r["T_tau"] is not None and r["T_tau"] > 20 for r in old_noncross),
        "h20_noncrossers": len(old_noncross),
        "delayed_families_after_20": sorted({r["rewrite_family"] for r in rows if r["T_tau"] is not None and r["T_tau"] > 20}),
        "optimizer_reset_only_delayed_family": sorted({r["rewrite_family"] for r in rows if r["T_tau"] is not None and r["T_tau"] > 10}) == ["optimizer_reset"],
        "h20_task_nonrecovered_rows": len(old_task),
        "h20_task_nonrecovered_later_recovered_by_h100": sum(not bool(r["task_nonrecovered_h100"]) for r in old_task),
        "h100_task_nonrecovered_rows": sum(bool(r["task_nonrecovered_h100"]) for r in rows),
        "latency_histogram_path": "results/reviewer2026_v2/h100_latency_histogram.json",
    }
    write_json(OUT / "h100_summary.json", summary)
    write_md(OUT / "h100_summary.md", f"""# H=100 Extension Summary

- Rows: {summary['rows']}
- H=100 crossings: {summary['crossed_h100']}
- H=20 noncrossers crossing during steps 21-100: {summary['crossed_after_20_from_h20_noncrossers']} / {summary['h20_noncrossers']}
- Delayed families after step 20: {', '.join(summary['delayed_families_after_20']) or 'none'}
- Optimizer-reset only delayed family beyond step 10: {summary['optimizer_reset_only_delayed_family']}
- H20 task-nonrecovered rows later recovered by H100: {summary['h20_task_nonrecovered_later_recovered_by_h100']} / {summary['h20_task_nonrecovered_rows']}
- H100 task-nonrecovered rows: {summary['h100_task_nonrecovered_rows']} / 99
""")


def run_mechanism() -> None:
    set_horizon()
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for seed, age in delayed_case_keys():
        batches, vocab, dataset = base.load_batches(seed, base.TRAIN_STEPS + H + 50)
        val_batches = batches[base.TRAIN_STEPS + 1 : base.TRAIN_STEPS + 4]
        calibration_batch = batches[base.TRAIN_STEPS + 4]
        base_file = SRC / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"checkpoint_step_{age}.pt"
        payload = base.load_checkpoint(base_file)
        summary, _, mech = run_pair(payload, "optimizer_reset", None, batches, val_batches, calibration_batch, instrument=True)
        case_id = f"gpu_seed{seed}_ckpt{age}_optimizer_reset"
        cases.append({"row_id": case_id, "seed": seed, "checkpoint_age": age, **summary})
        for m in mech:
            rows.append({"row_id": case_id, "seed": seed, "checkpoint_age": age, "rewrite_family": "optimizer_reset", **m})
        plot_case(case_id, mech, plot_dir)
        write_csv(OUT / "delayed_mechanism_steps.csv", rows)
        write_json(OUT / "delayed_mechanism_cases.json", cases)
    write_md(OUT / "delayed_mechanism_summary.md", summarize_mechanism(cases, rows))


def plot_case(case_id: str, rows: list[dict[str, Any]], plot_dir: Path) -> None:
    if not rows:
        return
    xs = [r["continuation_step"] for r in rows]
    fields = ["delta_update_norm", "update_cosine", "m_difference_norm", "v_difference_norm", "parameter_divergence"]
    fig, axes = plt.subplots(len(fields), 1, figsize=(6.5, 8.5), sharex=True)
    for ax, field in zip(axes, fields):
        ax.plot(xs, [r.get(field, math.nan) for r in rows], marker="o", markersize=2, linewidth=1)
        ax.set_ylabel(field)
    axes[-1].set_xlabel("continuation step")
    fig.suptitle(case_id)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{case_id}_mechanism.png", dpi=160)
    plt.close(fig)


def summarize_mechanism(cases: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    by_case = []
    for c in cases:
        rs = [r for r in rows if r["row_id"] == c["row_id"]]
        by_case.append([c["row_id"], c["T_tau"], max(r["v_relative_difference"] for r in rs), max(r["m_relative_difference"] for r in rs), min(r["update_cosine"] for r in rs if math.isfinite(r["update_cosine"]))])
    lines = ["# Delayed Optimizer-Reset Mechanism", "", "| row | T_tau | max v rel diff | max m rel diff | min update cosine |", "|---|---:|---:|---:|---:|"]
    for row in by_case:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.4g} | {row[3]:.4g} | {row[4]:.4g} |")
    lines.append("")
    lines.append("Evidence is per-case; the files retain the full stepwise moment, update-vector, scheduler, and divergence traces.")
    return "\n".join(lines)


def run_multi_future() -> None:
    set_horizon()
    rows: list[dict[str, Any]] = []
    write_json(OUT / "multi_future_frozen_schedule_seeds.json", {"future_schedule_seeds": FUTURE_SCHEDULE_SEEDS, "frozen_before_results": True})
    for seed, age in delayed_case_keys():
        base_file = SRC / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"checkpoint_step_{age}.pt"
        payload = base.load_checkpoint(base_file)
        for future_seed in FUTURE_SCHEDULE_SEEDS:
            batches, vocab, dataset = base.load_batches(future_seed, H + 60)
            val_batches = batches[1:4]
            calibration_batch = batches[4]
            summary, _, _ = run_pair(payload, "optimizer_reset", None, batches, val_batches, calibration_batch, future_start=5, instrument=False)
            rows.append({
                "row_id": f"gpu_seed{seed}_ckpt{age}_optimizer_reset",
                "source_seed": seed,
                "checkpoint_age": age,
                "future_schedule_seed": future_seed,
                "T_tau": summary["T_tau"],
                "censored": summary["T_tau"] is None,
                "task_recovery_step": summary["task_recovery_step"],
                "task_nonrecovered_h20": summary["task_nonrecovered_h20"],
                "task_nonrecovered_h100": summary["task_nonrecovered_h100"],
                "max_validation_loss_gap": summary["max_validation_loss_gap"],
                "final_validation_loss_gap": summary["final_validation_loss_gap"],
                "max_function_kl": summary["max_calibration_kl"],
                "final_function_kl": summary["final_calibration_kl"],
            })
            write_csv(OUT / "multi_future_rows.csv", rows)
    summary = []
    for key in sorted({(r["source_seed"], r["checkpoint_age"]) for r in rows}):
        rs = [r for r in rows if (r["source_seed"], r["checkpoint_age"]) == key]
        tvals = [r["T_tau"] for r in rs if r["T_tau"] is not None]
        summary.append({
            "source_seed": key[0],
            "checkpoint_age": key[1],
            "future_schedules": len(rs),
            "P_T_le_10": sum(t is not None and t <= 10 for t in [r["T_tau"] for r in rs]) / len(rs),
            "P_T_le_20": sum(t is not None and t <= 20 for t in [r["T_tau"] for r in rs]) / len(rs),
            "P_T_le_50": sum(t is not None and t <= 50 for t in [r["T_tau"] for r in rs]) / len(rs),
            "P_T_le_100": sum(t is not None and t <= 100 for t in [r["T_tau"] for r in rs]) / len(rs),
            "median_T_tau": float(np.median(tvals)) if tvals else None,
            "min_T_tau": min(tvals) if tvals else None,
            "max_T_tau": max(tvals) if tvals else None,
            "IQR_T_tau": float(np.percentile(tvals, 75) - np.percentile(tvals, 25)) if len(tvals) >= 2 else None,
            "proportion_task_nonrecovered_H20": sum(r["task_nonrecovered_h20"] for r in rs) / len(rs),
            "proportion_task_nonrecovered_H100": sum(r["task_nonrecovered_h100"] for r in rs) / len(rs),
        })
    write_json(OUT / "multi_future_summary.json", summary)
    write_md(OUT / "multi_future_summary.md", "# Multi-Future Summary\n\n" + "\n".join(f"- seed {r['source_seed']} age {r['checkpoint_age']}: P(T<=20)={r['P_T_le_20']:.2f}, median T={r['median_T_tau']}" for r in summary))


def rankdata(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx == 0 or vy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(rx, ry)) / math.sqrt(vx * vy)


def auc(points: list[tuple[float, float]]) -> float:
    pts = sorted(points)
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))


def roc_pr(rows: list[dict[str, Any]], score: str, label: str = "task_nonrecovered_h100") -> dict[str, Any]:
    thresholds = [float("inf")] + sorted({float(r[score]) for r in rows}, reverse=True) + [-1e-12]
    roc = []
    pr = []
    for th in thresholds:
        tp = fp = tn = fn = 0
        for r in rows:
            pred = float(r[score]) >= th
            y = bool(r[label])
            if pred and y:
                tp += 1
            elif pred and not y:
                fp += 1
            elif not pred and not y:
                tn += 1
            else:
                fn += 1
        tpr = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tpr
        roc.append((fpr, tpr))
        pr.append((recall, precision))
    return {"AUROC": auc(roc), "AUPRC": auc(pr)}


def post_metrics() -> None:
    if not (OUT / "h100_rows.json").exists():
        return
    rows = json.loads((OUT / "h100_rows.json").read_text(encoding="utf-8"))
    metrics = {
        "original_global_parameter_divergence": "max_normalized_parameter_divergence",
        "clean_update_normalized_divergence": "max_clean_update_normalized_divergence",
        "max_layer_relative_displacement": "max_layer_relative_displacement",
        "function_space_kl": "max_calibration_kl",
        "function_space_rms_logit": "max_calibration_rms_logit",
    }
    alt_rows = []
    summary: dict[str, Any] = {}
    for name, field in metrics.items():
        xs = [float(r[field]) for r in rows]
        ys = [float(r["max_validation_loss_gap"]) for r in rows]
        summary[name] = {
            "spearman_vs_max_validation_loss_gap": spearman(xs, ys),
            "roc_pr_h100": roc_pr(rows, field, "task_nonrecovered_h100"),
            "roc_pr_h20": roc_pr(rows, field, "task_nonrecovered_h20"),
            "per_seed": {},
        }
        for seed in base.HELDOUT_SEEDS:
            rs = [r for r in rows if int(r["seed"]) == seed]
            summary[name]["per_seed"][str(seed)] = {
                "spearman": spearman([float(r[field]) for r in rs], [float(r["max_validation_loss_gap"]) for r in rs]),
                "h100": roc_pr(rs, field, "task_nonrecovered_h100"),
                "h20": roc_pr(rs, field, "task_nonrecovered_h20"),
            }
        for r in rows:
            alt_rows.append({"row_id": r["row_id"], "seed": r["seed"], "checkpoint_age": r["checkpoint_age"], "rewrite_family": r["rewrite_family"], "metric": name, "value": r[field], "max_validation_loss_gap": r["max_validation_loss_gap"], "task_nonrecovered_h100": r["task_nonrecovered_h100"]})
    write_csv(OUT / "alternative_metrics_v2.csv", alt_rows)
    write_json(OUT / "alternative_metrics_v2.json", summary)
    write_md(OUT / "alternative_metrics_v2.md", "# Alternative Metrics V2\n\nComputed from persisted H=100 clean/candidate trajectories: clean-update-normalized divergence, layerwise relative displacement, and function-space KL/RMS logit metrics.")
    write_json(OUT / "roc_pr_robustness.json", {"bootstrap_seed": ROC_BOOTSTRAP_SEED, "metrics": summary, "note": "Per-seed AUROC/AUPRC are reported; only three independent training seeds are available."})
    task_sweep(rows)
    comparison_summary(rows, summary)


def task_sweep(rows: list[dict[str, Any]]) -> None:
    out = []
    for horizon in [20, 100]:
        for eps in TASK_THRESHOLDS:
            damaged = 0
            for r in rows:
                gaps = r["validation_loss_gap"][:horizon]
                damaged += int(recovery_step(gaps, eps) is None)
            out.append({"horizon": horizon, "threshold": eps, "task_nonrecovered": damaged, "rows": len(rows)})
    controls = [r for r in rows if r["rewrite_family"] in {"exact_restore", "benign_nonzero"}]
    write_csv(OUT / "task_threshold_sweep_v2.csv", out)
    write_json(OUT / "task_threshold_sweep_v2.json", out)
    write_json(OUT / "control_distributions_v2.json", {
        "control_families": ["exact_restore", "benign_nonzero"],
        "max_validation_loss_gap": [r["max_validation_loss_gap"] for r in controls],
        "max_parameter_divergence": [r["max_parameter_divergence"] for r in controls],
        "max_calibration_kl": [r["max_calibration_kl"] for r in controls],
    })


def comparison_summary(rows: list[dict[str, Any]], metric_summary: dict[str, Any]) -> None:
    h100 = json.loads((OUT / "h100_summary.json").read_text(encoding="utf-8")) if (OUT / "h100_summary.json").exists() else {}
    mf = json.loads((OUT / "multi_future_summary.json").read_text(encoding="utf-8")) if (OUT / "multi_future_summary.json").exists() else []
    lines = [
        "# Reviewer Experiment Completion V2",
        "",
        "## Original H20 Campaign",
        "",
        "- rows: 99",
        "- tau crossings: 25",
        "- task-nonrecovered: 43",
        "- delayed rows: 9",
        "",
        "## H100 Extension",
        "",
        f"- crossings after step 20: {h100.get('crossed_after_20_from_h20_noncrossers', 'not run')}",
        f"- H100 task-nonrecovered rows: {h100.get('h100_task_nonrecovered_rows', 'not run')}",
        "",
        "## Multi-Future",
        "",
        f"- completed case summaries: {len(mf) if isinstance(mf, list) else 'not run'}",
        "",
        "## Mechanism",
        "",
        "- see delayed_mechanism_steps.csv and delayed_mechanism_summary.md",
        "",
        "## Controlled Scale",
        "",
        "- not completed unless controlled_scale/summary.json is present",
        "",
        "## Longer Training",
        "",
        "- not completed unless long_training/summary.json is present",
        "",
        "## Alternative Metrics",
        "",
    ]
    for name, summary in metric_summary.items():
        lines.append(f"- {name}: H100 AUROC={summary['roc_pr_h100']['AUROC']:.3f}, AUPRC={summary['roc_pr_h100']['AUPRC']:.3f}")
    write_md(OUT / "REVIEWER_EXPERIMENT_COMPLETION.md", "\n".join(lines))


def freeze_heavy_protocols() -> None:
    scale_dir = OUT / "controlled_scale"
    long_dir = OUT / "long_training"
    scale_dir.mkdir(parents=True, exist_ok=True)
    long_dir.mkdir(parents=True, exist_ok=True)
    batches, vocab, _ = base.load_batches(base.DEV_SEED, base.TRAIN_STEPS + H + 50)
    scale_cfgs = []
    for cfg in SCALE_CONFIGS:
        model = base.make_model(cfg, vocab)
        scale_cfgs.append({**cfg, "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        del model
    scale_protocol = {
        "status": "FROZEN_BEFORE_RESULTS",
        "seeds": SCALE_SEEDS,
        "model_configs": scale_cfgs,
        "rewrite_families": base.REWRITE_FAMILIES,
        "training_steps": 300,
        "checkpoint_ages": base.CHECKPOINT_AGES,
        "horizon": H,
        "optimizer": {"lr": base.LR, "betas": base.BETAS, "eps": base.EPS, "weight_decay": base.WEIGHT_DECAY},
        "scheduler": {"step_size": base.SCHED_STEP, "gamma": base.SCHED_GAMMA},
        "dataset": "WikiText-2 cached char-level",
    }
    write_json(scale_dir / "FROZEN_PROTOCOL.json", scale_protocol)
    write_json(scale_dir / "FROZEN_PROTOCOL.sha256.json", {"sha256": sha256(scale_dir / "FROZEN_PROTOCOL.json")})
    long_protocol = {
        "status": "FROZEN_BEFORE_RESULTS",
        "seeds": LONG_SEEDS,
        "model_config": base.MODEL_CFG,
        "training_steps": 3000,
        "checkpoint_ages": [750, 1500, 3000],
        "horizon": H,
        "optimizer": {"lr": 1e-3, "betas": base.BETAS, "eps": base.EPS, "weight_decay": base.WEIGHT_DECAY},
        "scheduler": {"step_size": 1000, "gamma": 0.5, "rationale": "preserve approximately original scheduler phase as fraction of training"},
        "stale_age": base.STALE_AGE,
    }
    write_json(long_dir / "FROZEN_PROTOCOL.json", long_protocol)
    write_json(long_dir / "FROZEN_PROTOCOL.sha256.json", {"sha256": sha256(long_dir / "FROZEN_PROTOCOL.json")})


def write_blocker(path: Path, reason: str) -> None:
    write_json(path / "BLOCKER.json", {"status": "NOT_COMPLETED", "technical_failure": reason, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    write_md(path / "STATUS.md", f"# Status\n\nNOT COMPLETED.\n\nTechnical failure/blocker: {reason}")


def manifest() -> None:
    files = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.name != "MANIFEST.json":
            files.append({"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size})
    write_json(OUT / "MANIFEST.json", {"source_commit": SOURCE_COMMIT, "created_utc": datetime.now(timezone.utc).isoformat(), "files": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="provenance,validate,h100,mechanism,multi,post,freeze-heavy")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reviewer2026_v2 reruns.")
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    if "provenance" in steps:
        provenance()
    if "validate" in steps:
        verify_original_delayed()
        validation_protocol()
    if "h100" in steps:
        run_h100()
    if "mechanism" in steps:
        run_mechanism()
    if "multi" in steps:
        run_multi_future()
    if "post" in steps:
        post_metrics()
    if "freeze-heavy" in steps:
        freeze_heavy_protocols()
        if not (OUT / "controlled_scale" / "summary.json").exists():
            write_blocker(OUT / "controlled_scale", "Controlled three-size campaign was frozen but not run in this invocation.")
        if not (OUT / "long_training" / "summary.json").exists():
            write_blocker(OUT / "long_training", "3000-step mature-state campaign was frozen but not run in this invocation.")
    manifest()
    print(f"reviewer2026_v2 steps complete: {steps}")


if __name__ == "__main__":
    main()

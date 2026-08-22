"""Run the OPT 2026 >=10M CUDA Transformer scale replication.

This runner is deliberately narrow: it uses the existing GPTMini family,
the frozen 11 rewrite families, AdamW, the inherited tau, and cached
WikiText-2 character data. It refuses to run the experiment without CUDA.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
sys.path[:0] = [str(EXP / "scripts"), str(EXP), str(ROOT / "scripts")]
from run_transformer import GPTMini, build_char_dataset  # noqa: E402

OUT = ROOT / "results" / "opt2026_10m_gpu_final"
PREV = ROOT / "results" / "opt2026_final_gpu_task_validity"
ARTIFACT = ROOT / "artifact" / "opt2026_final_anonymous"
ZIP_PATH = ROOT / "lambdaopt_OPT2026_FINAL_ANONYMOUS_ARTIFACT.zip"

TAU_FRAC = 0.10
TAU_FORMULA = "tau = 0.10 * (||theta_checkpoint||_2 + 1e-12)"
TAU_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TASK_RECOVERY_EPS = 0.05
RECOVERY_CONSECUTIVE = 3
HORIZON = 20
TRAIN_STEPS = 300
CHECKPOINT_AGES = [75, 150, 300]
HELDOUT_SEEDS = [1101, 1102, 1103]
DEV_SEED = 1051
FORBIDDEN_SEEDS = {201, 202, 203, 221, 222, 223, 231, 232, 233, 801, 901, 902, 903, 951, 1001, 1002, 1003}
STALE_AGE = 25
STEP_OFFSET = 25
DATA_CHARS = 500000
BATCH_SIZE = 1
SEQ_LEN = 128
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
SCHED_STEP = 100
SCHED_GAMMA = 0.5
DEVICE = "cuda"
PRECISION = "fp32"

REWRITE_FAMILIES = [
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

CANDIDATES = [
    dict(name="candidate_A_9p99m", num_layers=8, d_model=320, nhead=8, dim_feedforward=1280, seq_len=127, dropout=0.0),
    dict(name="candidate_B_16p82m", num_layers=8, d_model=416, nhead=8, dim_feedforward=1664, seq_len=127, dropout=0.0),
    dict(name="candidate_B_19p48m", num_layers=8, d_model=448, nhead=8, dim_feedforward=1792, seq_len=127, dropout=0.0),
    dict(name="candidate_C_25p41m", num_layers=8, d_model=512, nhead=8, dim_feedforward=2048, seq_len=127, dropout=0.0),
]
MODEL_CFG = CANDIDATES[2]
FULL_EVAL_FAILURES = [
    {
        "candidate": "candidate_C_25p41m",
        "trainable_params": 25414144,
        "failure": "survived one CUDA AdamW step but failed during full rewrite evaluation with CUBLAS_STATUS_EXECUTION_FAILED",
        "decision": "not used for three-seed replication; next-largest >=10M candidate selected",
    }
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else str(x)
    if torch.is_tensor(x):
        return to_jsonable(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def ps_drives() -> list[dict[str, Any]]:
    script = "Get-PSDrive -PSProvider FileSystem | Select Name,Root,Used,Free | ConvertTo-Json"
    raw = run_cmd(["powershell", "-NoProfile", "-Command", script])
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception:
        return [{"raw": raw}]


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def cpu_model() -> str:
    return run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])


def system_ram() -> int | str:
    raw = run_cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
    try:
        return int(raw)
    except ValueError:
        return raw


def nvidia_query() -> dict[str, Any]:
    raw = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"])
    parts = [p.strip() for p in raw.splitlines()[0].split(",")] if raw and not raw.startswith("ERROR") else []
    if len(parts) >= 5:
        return {
            "gpu_name": parts[0],
            "total_vram_mib": int(parts[1]),
            "used_vram_mib": int(parts[2]),
            "available_vram_mib": int(parts[3]),
            "driver_version": parts[4],
            "raw": raw,
        }
    return {"raw": raw}


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for opt2026_10m_gpu_final; refusing CPU fallback.")


def collect_hardware(free_space_before: list[dict[str, Any]], free_space_after: list[dict[str, Any]]) -> dict[str, Any]:
    require_cuda()
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version.split()[0],
        "python_executable": Path(sys.executable).name,
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "system_ram_bytes": system_ram(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
        "total_vram_bytes": props.total_memory,
        "nvidia_smi": nvidia_query(),
        "free_disk_space_before_setup": free_space_before,
        "free_disk_space_after_setup": free_space_after,
        "venv_gpu_size_bytes": dir_size(ROOT / ".venv-gpu"),
        "pip_cache": run_cmd([sys.executable, "-m", "pip", "cache", "info"]),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def make_model(cfg: dict[str, Any], vocab: int) -> nn.Module:
    model = GPTMini(
        vocab_size=vocab,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        seq_len=cfg["seq_len"],
        dropout=0.0,
    )
    model.eval()
    return model


def make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)


def make_scheduler(opt: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=SCHED_STEP, gamma=SCHED_GAMMA)


def n_params(cfg: dict[str, Any], vocab: int) -> int:
    model = make_model(cfg, vocab)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    return n


def load_batches(seed: int, needed: int):
    path = EXP / "data" / "wikitext2_raw" / "train.txt"
    if not path.exists():
        raise FileNotFoundError(f"missing cached WikiText-2 at {path}; this runner does not use synthetic tokens")
    text = path.read_text(encoding="utf-8")[:DATA_CHARS]
    batches, vocab = build_char_dataset(text, SEQ_LEN, BATCH_SIZE, seed=seed)
    if len(batches) < needed:
        raise RuntimeError(f"need {needed} batches, found {len(batches)}")
    return batches[:needed], vocab, f"WikiText-2 cached char-level, first {len(text)} chars"


def to_device_batch(batch):
    x, y = batch
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)


def loss_only(model: nn.Module, batch) -> float:
    x, y = to_device_batch(batch)
    with torch.no_grad():
        logits = model(x)
        b, t, v = logits.shape
        loss = nn.functional.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t))
    return float(loss.detach().cpu().item())


def train_one(model: nn.Module, opt, sched, batch) -> float:
    x, y = to_device_batch(batch)
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    b, t, v = logits.shape
    loss = nn.functional.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t))
    loss.backward()
    opt.step()
    sched.step()
    return float(loss.detach().cpu().item())


def flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()])


def l2(x: torch.Tensor) -> float:
    return float(x.float().norm(2).item())


def cpu_clone(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_clone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cpu_clone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_clone(v) for v in x)
    return copy.deepcopy(x)


def optimizer_state_to_device(opt: torch.optim.Optimizer) -> None:
    for state in opt.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(DEVICE)


def checkpoint_payload(model, opt, sched, *, seed: int, step: int, cfg: dict[str, Any], vocab: int, dataset: str) -> dict[str, Any]:
    return {
        "format": "opt2026_10m_gpu_final",
        "model_state": cpu_clone(model.state_dict()),
        "optimizer_state": cpu_clone(opt.state_dict()),
        "scheduler_state": copy.deepcopy(sched.state_dict()),
        "training_step": step,
        "batch_pos": step,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "cfg": cfg,
        "vocab": vocab,
        "seed": seed,
        "dataset": dataset,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def fresh_from_payload(payload: dict[str, Any], *, load_optimizer: bool = True, load_scheduler: bool = True):
    model = make_model(payload["cfg"], int(payload["vocab"])).to(DEVICE)
    opt = make_optimizer(model)
    sched = make_scheduler(opt)
    model.load_state_dict(payload["model_state"])
    if load_optimizer and payload.get("optimizer_state") is not None:
        opt.load_state_dict(payload["optimizer_state"])
        optimizer_state_to_device(opt)
    if load_scheduler and payload.get("scheduler_state") is not None:
        sched.load_state_dict(payload["scheduler_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if payload.get("cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
    return model, opt, sched


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def mutate_optimizer_state(opt_state: dict[str, Any] | None, mode: str, stale: dict[str, Any] | None) -> dict[str, Any] | None:
    if mode == "optimizer_reset":
        return None
    opt_state = copy.deepcopy(opt_state)
    if mode == "model_current_optimizer_stale":
        return copy.deepcopy(stale)
    if opt_state is None:
        return None
    if mode in {"stale_m", "stale_v", "stale_mv"} and stale is not None:
        for key, st in opt_state["state"].items():
            old = stale["state"].get(key)
            if old is None:
                continue
            if mode in {"stale_m", "stale_mv"} and "exp_avg" in st:
                st["exp_avg"] = old["exp_avg"].clone()
            if mode in {"stale_v", "stale_mv"} and "exp_avg_sq" in st:
                st["exp_avg_sq"] = old["exp_avg_sq"].clone()
    if mode == "reset_m":
        for st in opt_state["state"].values():
            if "exp_avg" in st:
                st["exp_avg"] = torch.zeros_like(st["exp_avg"])
    if mode == "reset_v":
        for st in opt_state["state"].values():
            if "exp_avg_sq" in st:
                st["exp_avg_sq"] = torch.zeros_like(st["exp_avg_sq"])
    if mode == "step_counter_mismatch":
        for st in opt_state["state"].values():
            if "step" in st:
                step = st["step"]
                st["step"] = torch.clamp(step - STEP_OFFSET, min=0) if torch.is_tensor(step) else max(0, step - STEP_OFFSET)
    return opt_state


def scenario_payload(base: dict[str, Any], family: str, stale: dict[str, Any] | None) -> dict[str, Any]:
    p = copy.deepcopy(base)
    p["rewrite_family"] = family
    if family == "benign_nonzero":
        first = next(iter(p["model_state"]))
        p["model_state"][first] = p["model_state"][first].clone()
        p["model_state"][first].view(-1)[0] += 1e-8
    elif family == "scheduler_mismatch":
        p["scheduler_state"] = None
    elif family != "exact_restore":
        p["optimizer_state"] = mutate_optimizer_state(p["optimizer_state"], family, stale)
    return p


def first_crossing(curve: list[float], tau: float) -> int | None:
    for i, value in enumerate(curve):
        if value >= tau:
            return i
    return None


def recovery_step(gaps: list[float]) -> int | str:
    for i in range(len(gaps) - RECOVERY_CONSECUTIVE + 1):
        if all(v <= TASK_RECOVERY_EPS for v in gaps[i : i + RECOVERY_CONSECUTIVE]):
            return i
    return ""


def continue_model(model, opt, sched, batches, start: int, val_batches):
    out = {"theta": [flat(model)], "train_loss": [], "validation_loss": [], "perplexity": []}
    for j in range(HORIZON):
        out["train_loss"].append(train_one(model, opt, sched, batches[(start + j) % len(batches)]))
        val = sum(loss_only(model, b) for b in val_batches) / len(val_batches)
        out["validation_loss"].append(val)
        out["perplexity"].append(math.exp(min(val, 50)))
        out["theta"].append(flat(model))
    return out


def lopt_project(start: float, growth: float, horizon: int) -> tuple[float, float]:
    if not math.isfinite(growth) or growth <= 0:
        return float("inf"), growth
    return start * (growth ** horizon), growth


def prefix_lopt(divs: list[float], tau: float, prefix: int) -> tuple[str, float, float]:
    if not divs or divs[0] <= 1e-14:
        return "PASS", 0.0, 0.0
    ratios = [max(divs[i] / divs[i - 1], 1e-30) for i in range(1, min(prefix, len(divs) - 1) + 1) if divs[i - 1] > 1e-14]
    growth = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else 1.0
    projected, used = lopt_project(divs[0], growth, HORIZON)
    return ("FLAG" if projected >= tau else "PASS"), projected, used


def rollout_decision(t_tau: int | None, k: int) -> str:
    return "FLAG" if t_tau is not None and t_tau <= k else "PASS"


def eval_row(seed: int, ckpt_step: int, family: str, base_file: Path, base_payload: dict[str, Any], stale: dict[str, Any] | None, batches, val_batches, nparam: int, dataset: str, scenario_file: str) -> dict[str, Any]:
    scen = scenario_payload(base_payload, family, stale)
    load_opt = family != "optimizer_reset"
    load_sched = family != "scheduler_mismatch"
    bm, bo, bs = fresh_from_payload(base_payload)
    rm, ro, rs = fresh_from_payload(scen, load_optimizer=load_opt, load_scheduler=load_sched)
    theta_norm = l2(flat(bm))
    tau = TAU_FRAC * (theta_norm + 1e-12)
    start = int(base_payload["batch_pos"])
    t0 = time.perf_counter()
    base = continue_model(bm, bo, bs, batches, start, val_batches)
    rew = continue_model(rm, ro, rs, batches, start, val_batches)
    torch.cuda.synchronize()
    runtime = time.perf_counter() - t0
    divs = [l2(a - b) for a, b in zip(base["theta"], rew["theta"])]
    val_gaps = [abs(a - b) for a, b in zip(base["validation_loss"], rew["validation_loss"])]
    ppl_gaps = [abs(a - b) for a, b in zip(base["perplexity"], rew["perplexity"])]
    t_tau = first_crossing(divs, tau)
    task_step = recovery_step(val_gaps)
    task_damaged = task_step == ""
    budgeted, bproj, bgrowth = prefix_lopt(divs, tau, 2)
    full, fproj, fgrowth = prefix_lopt(divs, tau, 5)
    row: dict[str, Any] = {
        "row_id": f"gpu_seed{seed}_ckpt{ckpt_step}_{family}",
        "experiment": "opt2026_10m_gpu_final",
        "dataset": dataset,
        "model": "GPTMini",
        "model_config": MODEL_CFG["name"],
        "n_params": nparam,
        "training_seed": seed,
        "checkpoint_step": ckpt_step,
        "training_steps": TRAIN_STEPS,
        "horizon": HORIZON,
        "rewrite_family": family,
        "checkpoint_file": str(base_file.relative_to(ROOT)),
        "scenario_checkpoint_file": scenario_file,
        "torch_save_file_exists": base_file.exists(),
        "torch_load_executed": True,
        "checkpoint_load_success": True,
        "training_resume_success": True,
        "same_future_minibatches": True,
        "same_rng_stream": True,
        "theta_checkpoint_norm": theta_norm,
        "tau_formula": TAU_FORMULA,
        "tau_numeric": tau,
        "tau_provenance": "Inherited fixed threshold from final task-validity campaign; not tuned on GPU rows.",
        "parameter_label": "dangerous" if t_tau is not None else "benign",
        "T_tau": "" if t_tau is None else t_tau,
        "max_parameter_divergence": max(divs),
        "final_parameter_divergence": divs[-1],
        "normalized_max_parameter_divergence": max(divs) / (theta_norm + 1e-12),
        "divergence_curve": divs,
        "validation_loss_base": base["validation_loss"],
        "validation_loss_rewrite": rew["validation_loss"],
        "validation_loss_gap": val_gaps,
        "perplexity_base": base["perplexity"],
        "perplexity_rewrite": rew["perplexity"],
        "perplexity_gap": ppl_gaps,
        "task_recovery_epsilon_loss": TASK_RECOVERY_EPS,
        "task_recovery_step": task_step,
        "task_damaged": task_damaged,
        "task_unrecovered_by_window": task_damaged,
        "final_validation_loss_gap": val_gaps[-1],
        "max_validation_loss_gap": max(val_gaps),
        "final_perplexity_gap": ppl_gaps[-1],
        "max_perplexity_gap": max(ppl_gaps),
        "full_h_replay_decision": "FLAG" if t_tau is not None else "PASS",
        "full_h_replay_runtime_seconds": runtime,
        "budgeted_lopt_decision": budgeted,
        "budgeted_lopt_projected": bproj,
        "budgeted_lopt_growth": bgrowth,
        "full_lopt_decision": full,
        "full_lopt_projected": fproj,
        "full_lopt_growth": fgrowth,
    }
    for mult in TAU_MULTIPLIERS:
        mtau = tau * mult
        mt = first_crossing(divs, mtau)
        row[f"T_tau_x{mult}"] = "" if mt is None else mt
        row[f"parameter_label_x{mult}"] = "dangerous" if mt is not None else "benign"
        for k in [1, 2, 5, 10, 20]:
            row[f"rollout_k{k}_decision_x{mult}"] = rollout_decision(mt, k)
    del bm, bo, bs, rm, ro, rs
    torch.cuda.empty_cache()
    return row


def smoke_test(out: Path, vocab: int) -> dict[str, Any]:
    require_cuda()
    torch.cuda.empty_cache()
    a = torch.ones((1024, 1024), device=DEVICE)
    b = torch.mm(a, a)
    tensor_sum = float(b.sum().detach().cpu().item())
    cfg = dict(name="smoke", num_layers=1, d_model=64, nhead=4, dim_feedforward=128, seq_len=127, dropout=0.0)
    model = make_model(cfg, vocab).to(DEVICE)
    opt = make_optimizer(model)
    sched = make_scheduler(opt)
    x = torch.randint(0, vocab, (1, 127), device=DEVICE)
    y = torch.randint(0, vocab, (1, 127), device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
    loss.backward()
    opt.step()
    sched.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    result = {
        "cuda_tensor_sum": tensor_sum,
        "forward_backward_adamw_step": True,
        "loss": float(loss.detach().cpu().item()),
        "memory_allocated_bytes": torch.cuda.memory_allocated(),
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "step_wall_time_seconds": dt,
    }
    write_json(out / "gpu_smoke_test.json", result)
    del a, b, model, opt, sched, x, y, logits, loss
    torch.cuda.empty_cache()
    return result


def model_probes(out: Path, vocab: int) -> list[dict[str, Any]]:
    probes = []
    for cfg in CANDIDATES:
        torch.cuda.empty_cache()
        status = "completed"
        error = ""
        nparam = 0
        peak = 0
        step_time = 0.0
        tokens_sec = 0.0
        try:
            model = make_model(cfg, vocab).to(DEVICE)
            nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
            opt = make_optimizer(model)
            sched = make_scheduler(opt)
            x = torch.randint(0, vocab, (1, 127), device=DEVICE)
            y = torch.randint(0, vocab, (1, 127), device=DEVICE)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            opt.step()
            sched.step()
            torch.cuda.synchronize()
            step_time = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated()
            tokens_sec = 127 / step_time
            del model, opt, sched, x, y, logits, loss
        except Exception as exc:
            status = "failed"
            error = repr(exc)
        probes.append(
            {
                "name": cfg["name"],
                "num_layers": cfg["num_layers"],
                "d_model": cfg["d_model"],
                "nhead": cfg["nhead"],
                "dim_feedforward": cfg["dim_feedforward"],
                "context": 128,
                "trainable_params": nparam,
                "status": status,
                "error": error,
                "precision": PRECISION,
                "microbatch": 1,
                "gradient_accumulation": 1,
                "peak_vram_bytes": peak,
                "step_wall_time_seconds": step_time,
                "tokens_per_second": tokens_sec,
                "selected_for_replication": cfg["name"] == MODEL_CFG["name"],
            }
        )
        torch.cuda.empty_cache()
    write_csv(out / "model_probe.csv", probes)
    return probes


def pretrain_seed(seed: int, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    set_seed(seed)
    batches, vocab, dataset = load_batches(seed, TRAIN_STEPS + HORIZON + 50)
    model = make_model(cfg, vocab).to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = make_optimizer(model)
    sched = make_scheduler(opt)
    ckpt_dir = out / "gpu_scale" / "checkpoints" / f"seed_{seed}"
    stale_steps = {age - STALE_AGE for age in CHECKPOINT_AGES}
    losses: list[float] = []
    step_times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for step in range(1, TRAIN_STEPS + 1):
        t0 = time.perf_counter()
        losses.append(train_one(model, opt, sched, batches[(step - 1) % len(batches)]))
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if step in stale_steps:
            path = ckpt_dir / f"optimizer_state_step_{step}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"optimizer_state": cpu_clone(opt.state_dict()), "training_step": step, "seed": seed}, path)
        if step in CHECKPOINT_AGES:
            save_checkpoint(ckpt_dir / f"checkpoint_step_{step}.pt", checkpoint_payload(model, opt, sched, seed=seed, step=step, cfg=cfg, vocab=vocab, dataset=dataset))
    torch.cuda.synchronize()
    total = time.perf_counter() - t_start
    peak = torch.cuda.max_memory_allocated()
    meta = {
        "seed": seed,
        "vocab": vocab,
        "dataset": dataset,
        "n_params": nparam,
        "train_loss_final": losses[-1],
        "mean_step_time_seconds": float(np.mean(step_times)),
        "tokens_per_second": float(127 / np.mean(step_times)),
        "total_training_wall_seconds": total,
        "peak_vram_bytes": peak,
        "checkpoint_dir": str(ckpt_dir.relative_to(ROOT)),
    }
    del model, opt, sched
    torch.cuda.empty_cache()
    return meta


def save_representative_scenarios(out: Path, base_file: Path, base_payload: dict[str, Any], stale: dict[str, Any] | None) -> dict[str, str]:
    scen_dir = out / "gpu_scale" / "checkpoints" / "representative_scenarios"
    saved: dict[str, str] = {}
    for fam in ["exact_restore", "reset_v"]:
        payload = scenario_payload(base_payload, fam, stale)
        path = scen_dir / f"{fam}.pt"
        save_checkpoint(path, payload)
        saved[fam] = str(path.relative_to(ROOT))
    return saved


def run_campaign(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    representative: dict[str, str] = {}
    for seed in HELDOUT_SEEDS:
        meta = pretrain_seed(seed, out, MODEL_CFG)
        metas.append(meta)
        batches, vocab, dataset = load_batches(seed, TRAIN_STEPS + HORIZON + 50)
        val_batches = batches[TRAIN_STEPS + 1 : TRAIN_STEPS + 4]
        for age in CHECKPOINT_AGES:
            base_file = out / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"checkpoint_step_{age}.pt"
            stale_file = out / "gpu_scale" / "checkpoints" / f"seed_{seed}" / f"optimizer_state_step_{age - STALE_AGE}.pt"
            base = load_checkpoint(base_file)
            stale = load_checkpoint(stale_file)["optimizer_state"] if stale_file.exists() else None
            if seed == HELDOUT_SEEDS[0] and age == CHECKPOINT_AGES[-1]:
                representative = save_representative_scenarios(out, base_file, base, stale)
            for fam in REWRITE_FAMILIES:
                scenario_file = representative.get(fam, "") if seed == HELDOUT_SEEDS[0] and age == CHECKPOINT_AGES[-1] else ""
                row = eval_row(seed, age, fam, base_file, base, stale, batches, val_batches, int(meta["n_params"]), dataset, scenario_file)
                rows.append(row)
                for i, (div, vg, pg) in enumerate(zip(row["divergence_curve"], [0.0] + row["validation_loss_gap"], [0.0] + row["perplexity_gap"])):
                    trajectories.append({"row_id": row["row_id"], "post_resume_index": i, "parameter_divergence": div, "validation_loss_gap": vg, "perplexity_gap": pg})
                gc.collect()
    return rows, trajectories, metas, representative


def rankdata(vals: list[float]) -> list[float]:
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * len(vals)
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


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = float(np.mean(rx)), float(np.mean(ry))
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den = math.sqrt(sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry))
    return num / den if den else float("nan")


def confusion(rows: list[dict[str, Any]], mult: float = 1.0) -> dict[str, Any]:
    a = b = c = d = 0
    for r in rows:
        danger = r[f"parameter_label_x{mult}"] == "dangerous" if mult != 1.0 else r["parameter_label"] == "dangerous"
        task = bool(r["task_damaged"])
        if not danger and not task:
            a += 1
        elif not danger and task:
            b += 1
        elif danger and not task:
            c += 1
        else:
            d += 1
    return {
        "parameter_benign_task_ok": a,
        "parameter_benign_task_damaged": b,
        "parameter_dangerous_task_ok": c,
        "parameter_dangerous_task_damaged": d,
        "sensitivity": d / (b + d) if b + d else float("nan"),
        "specificity": a / (a + c) if a + c else float("nan"),
        "ppv": d / (c + d) if c + d else float("nan"),
        "npv": a / (a + b) if a + b else float("nan"),
    }


def tau_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for mult in TAU_MULTIPLIERS:
        dangerous = [r for r in rows if r[f"parameter_label_x{mult}"] == "dangerous"]
        benign = [r for r in rows if r[f"parameter_label_x{mult}"] == "benign"]
        tvals = [int(r[f"T_tau_x{mult}"]) for r in dangerous if r[f"T_tau_x{mult}"] != ""]
        conf = confusion(rows, mult)
        out.append(
            {
                "tau_multiplier": mult,
                "parameter_dangerous_count": len(dangerous),
                "parameter_benign_count": len(benign),
                "parameter_benign_task_damaged_count": conf["parameter_benign_task_damaged"],
                "parameter_dangerous_task_damaged_count": conf["parameter_dangerous_task_damaged"],
                "sensitivity": conf["sensitivity"],
                "specificity": conf["specificity"],
                "ppv": conf["ppv"],
                "npv": conf["npv"],
                "rollout_1_misses": sum(int(r[f"T_tau_x{mult}"]) > 1 for r in dangerous if r[f"T_tau_x{mult}"] != ""),
                "rollout_2_misses": sum(int(r[f"T_tau_x{mult}"]) > 2 for r in dangerous if r[f"T_tau_x{mult}"] != ""),
                "rollout_5_misses": sum(int(r[f"T_tau_x{mult}"]) > 5 for r in dangerous if r[f"T_tau_x{mult}"] != ""),
                "rollout_10_misses": sum(int(r[f"T_tau_x{mult}"]) > 10 for r in dangerous if r[f"T_tau_x{mult}"] != ""),
                "rollout_20_misses": sum(int(r[f"T_tau_x{mult}"]) > 20 for r in dangerous if r[f"T_tau_x{mult}"] != ""),
                "median_T_tau": float(np.median(tvals)) if tvals else "",
                "max_T_tau": max(tvals) if tvals else "",
            }
        )
    return out


def grouped(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out = []
    for value in sorted({r[key] for r in rows}):
        rs = [r for r in rows if r[key] == value]
        tvals = [int(r["T_tau"]) for r in rs if r["T_tau"] != ""]
        out.append(
            {
                key: value,
                "rows": len(rs),
                "parameter_dangerous": sum(r["parameter_label"] == "dangerous" for r in rs),
                "task_damaged": sum(bool(r["task_damaged"]) for r in rs),
                "median_T_tau": float(np.median(tvals)) if tvals else "",
                "max_T_tau": max(tvals) if tvals else "",
            }
        )
    return out


def heatmaps(rows: list[dict[str, Any]], out: Path) -> None:
    for field, stem, title in [
        ("parameter_label", "gpu_parameter_danger_heatmap", ">=10M parameter dangerous / 3 seeds"),
        ("task_damaged", "gpu_task_damage_heatmap", ">=10M task damaged / 3 seeds"),
    ]:
        mat = []
        cells = []
        for fam in REWRITE_FAMILIES:
            line = []
            for age in CHECKPOINT_AGES:
                rs = [r for r in rows if r["rewrite_family"] == fam and int(r["checkpoint_step"]) == age]
                count = sum((r[field] == "dangerous") if field == "parameter_label" else bool(r[field]) for r in rs)
                line.append(count)
                cells.append({"rewrite_family": fam, "checkpoint_age": age, "count": count, "seeds": len(rs), "cell": f"{count}/{len(rs)}"})
            mat.append(line)
        write_csv(out / f"{stem}.csv", cells)
        fig, ax = plt.subplots(figsize=(6.3, 5.0))
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=3)
        ax.set_xticks(range(len(CHECKPOINT_AGES)), CHECKPOINT_AGES)
        ax.set_yticks(range(len(REWRITE_FAMILIES)), REWRITE_FAMILIES, fontsize=7)
        ax.set_xlabel("checkpoint age / optimizer steps")
        ax.set_title(title)
        for i, line in enumerate(mat):
            for j, value in enumerate(line):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="count")
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=180)
        fig.savefig(out / f"{stem}.pdf")
        plt.close(fig)


def scatter_plots(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]
    for idx, fam in enumerate(REWRITE_FAMILIES):
        rs = [r for r in rows if r["rewrite_family"] == fam]
        ax.scatter([r["normalized_max_parameter_divergence"] for r in rs], [r["max_validation_loss_gap"] for r in rs], s=30, marker=markers[idx], alpha=0.75, label=fam)
    ax.axvline(0.10, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(TASK_RECOVERY_EPS, color="tab:red", linestyle=":", linewidth=1.0)
    ax.set_xlabel("normalized max parameter divergence")
    ax.set_ylabel("max validation-loss gap")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gpu_parameter_vs_task_damage.png", dpi=180)
    fig.savefig(out / "gpu_parameter_vs_task_damage.pdf")
    plt.close(fig)


def scale_comparison(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    prev_summary = json.loads((PREV / "summary.json").read_text(encoding="utf-8"))
    prev_conf = prev_summary["confusion_by_source"]["final_1p298m_scale_grid"]
    tvals = [int(r["T_tau"]) for r in rows if r["T_tau"] != ""]
    gpu = {
        "actual_parameter_count": rows[0]["n_params"],
        "independent_training_runs": len(HELDOUT_SEEDS),
        "checkpoint_ages": CHECKPOINT_AGES,
        "training_steps": TRAIN_STEPS,
        "total_rewrite_rows": len(rows),
        "parameter_dangerous": sum(r["parameter_label"] == "dangerous" for r in rows),
        "task_damaged": sum(bool(r["task_damaged"]) for r in rows),
        "second_moment_dangerous": sum(r["rewrite_family"] in {"reset_v", "stale_v"} and r["parameter_label"] == "dangerous" for r in rows),
        "benign_but_task_damaged": confusion(rows)["parameter_benign_task_damaged"],
        "median_T_tau": float(np.median(tvals)) if tvals else "",
        "max_T_tau": max(tvals) if tvals else "",
        "rollout_1_misses": tau_sensitivity(rows)[3]["rollout_1_misses"],
        "rollout_5_misses": tau_sensitivity(rows)[3]["rollout_5_misses"],
        "rollout_10_misses": tau_sensitivity(rows)[3]["rollout_10_misses"],
        "spearman_loss": spearman([r["normalized_max_parameter_divergence"] for r in rows], [r["max_validation_loss_gap"] for r in rows]),
    }
    prev = {
        "actual_parameter_count": 1298080,
        "independent_training_runs": 3,
        "checkpoint_ages": [50, 100, 200],
        "training_steps": 200,
        "total_rewrite_rows": 99,
        "parameter_dangerous": prev_conf["parameter_dangerous_task_ok"] + prev_conf["parameter_dangerous_task_damaged"],
        "task_damaged": prev_conf["parameter_benign_task_damaged"] + prev_conf["parameter_dangerous_task_damaged"],
        "second_moment_dangerous": 16,
        "benign_but_task_damaged": prev_conf["parameter_benign_task_damaged"],
        "median_T_tau": 1,
        "max_T_tau": 1,
        "rollout_1_misses": 0,
        "rollout_5_misses": 0,
        "rollout_10_misses": 0,
        "spearman_loss": prev_summary["correlations"]["spearman_norm_divergence_vs_max_validation_loss_gap"],
    }
    labels = ["danger", "task damage", "benign task damage"]
    prev_vals = [prev["parameter_dangerous"] / prev["total_rewrite_rows"], prev["task_damaged"] / prev["total_rewrite_rows"], prev["benign_but_task_damaged"] / prev["total_rewrite_rows"]]
    gpu_vals = [gpu["parameter_dangerous"] / gpu["total_rewrite_rows"], gpu["task_damaged"] / gpu["total_rewrite_rows"], gpu["benign_but_task_damaged"] / gpu["total_rewrite_rows"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x - 0.18, prev_vals, width=0.36, label="1.298M")
    ax.bar(x + 0.18, gpu_vals, width=0.36, label=f"{gpu['actual_parameter_count']/1e6:.1f}M")
    ax.set_xticks(x, labels)
    ax.set_ylabel("fraction of rows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "scale_comparison.png", dpi=180)
    fig.savefig(out / "scale_comparison.pdf")
    plt.close(fig)
    comp = {"scale_1p298m": prev, "gpu_ge_10m": gpu}
    write_json(out / "scale_comparison.json", comp)
    return comp


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text_outputs(out: Path, summary: dict[str, Any]) -> None:
    c = summary["confusion"]
    text = f"""# GPU Scale Manuscript Insert

## >=10M Model Setup

We ran an order-of-magnitude scale replication using the same GPTMini decoder-only Transformer family and cached WikiText-2 character-level data. The replicated CUDA model has {summary['model']['trainable_params']} trainable parameters, {MODEL_CFG['num_layers']} layers, {MODEL_CFG['nhead']} heads, d_model {MODEL_CFG['d_model']}, FFN dimension {MODEL_CFG['dim_feedforward']}, and context length 128.

## Hardware

The experiment used an NVIDIA GeForce GTX 1650 with 4096 MiB VRAM under CUDA PyTorch {summary['hardware']['torch_version']} / CUDA {summary['hardware']['torch_cuda_version']}. Training used fp32, microbatch 1, gradient accumulation 1, and AdamW with the inherited optimizer settings.

## Scale Comparison

The 1.298M campaign had 16/99 parameter-dangerous rows and 20/99 task-damaged rows. The >=10M GPU campaign had {summary['scale_comparison']['gpu_ge_10m']['parameter_dangerous']}/{summary['scale_comparison']['gpu_ge_10m']['total_rewrite_rows']} parameter-dangerous rows and {summary['scale_comparison']['gpu_ge_10m']['task_damaged']}/{summary['scale_comparison']['gpu_ge_10m']['total_rewrite_rows']} task-damaged rows.

## Task-Damage Relationship

At inherited tau, the >=10M 2x2 table is benign/task OK {c['parameter_benign_task_ok']}, benign/task damaged {c['parameter_benign_task_damaged']}, dangerous/task OK {c['parameter_dangerous_task_ok']}, and dangerous/task damaged {c['parameter_dangerous_task_damaged']}. These are empirical descriptive quantities, not certification metrics.

## Limitations

This is an order-of-magnitude scale replication relative to 1.298M parameters. It is not modern LLM-scale evidence.
"""
    (out / "GPU_SCALE_MANUSCRIPT_INSERT.md").write_text(text, encoding="utf-8")
    lines = [
        "% Auto-generated GPU scale tables",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Metric & 1.298M & GPU replication \\\\",
        f"Parameters & 1,298,080 & {summary['model']['trainable_params']} \\\\",
        f"Rows & 99 & {summary['rows']} \\\\",
        f"Dangerous & 16 & {summary['scale_comparison']['gpu_ge_10m']['parameter_dangerous']} \\\\",
        f"Task damaged & 20 & {summary['scale_comparison']['gpu_ge_10m']['task_damaged']} \\\\",
        f"Benign task damaged & 10 & {c['parameter_benign_task_damaged']} \\\\",
        f"Median $T_\\tau$ & 1 & {summary['scale_comparison']['gpu_ge_10m']['median_T_tau']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    (out / "GPU_SCALE_TABLES.tex").write_text("\n".join(lines), encoding="utf-8")
    audit = f"""# Final 10M GPU Audit

- branch: experiment/opt2026-10m-gpu-final
- commit: filled after commit
- Python: {summary['hardware']['python']}
- PyTorch: {summary['hardware']['torch_version']}
- CUDA runtime: {summary['hardware']['torch_cuda_version']}
- driver: {summary['hardware']['nvidia_smi'].get('driver_version')}
- GPU: {summary['hardware']['gpu_name']}
- VRAM bytes: {summary['hardware']['total_vram_bytes']}
- disk cleanup performed: removed disposable failed project `.venv-gpu` of 6372380 bytes; pip cache was 0 bytes
- model configs probed: see model_probe.csv
- exact model chosen: {MODEL_CFG}
- trainable params: {summary['model']['trainable_params']}
- precision: {PRECISION}
- peak VRAM: {summary['runtime']['peak_vram_bytes']}
- throughput tokens/sec: {summary['runtime']['mean_tokens_per_second']}
- mean training step time: {summary['runtime']['mean_step_time_seconds']}
- training steps: {TRAIN_STEPS}
- independent seeds: {HELDOUT_SEEDS}
- checkpoint ages: {CHECKPOINT_AGES}
- rewrite taxonomy: frozen 11-family taxonomy copied from final task-validity campaign
- tau: {TAU_FORMULA}
- tau sweep: see tau_sensitivity.csv
- task-damage table: {c}
- rollout frontier: see tau_sensitivity.csv
- lambda-Opt: budgeted/full columns in rows.csv; no tuning
- 1.298M vs >=10M comparison: see scale_comparison.json
- tests/verifiers/artifact verification: recorded after final verification
- negative results: {summary['negative_results']}
"""
    (out / "FINAL_10M_GPU_AUDIT.md").write_text(audit, encoding="utf-8")


def update_artifact(out: Path, representative_files: dict[str, str]) -> dict[str, Any]:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    dst_results = ARTIFACT / "results" / "opt2026_10m_gpu_final"
    if dst_results.exists():
        shutil.rmtree(dst_results)
    shutil.copytree(out, dst_results, ignore=shutil.ignore_patterns("*.pt"))
    for rel in [representative_files.get("exact_restore", ""), representative_files.get("reset_v", "")]:
        if rel:
            src = ROOT / rel
            dst = ARTIFACT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for rel in ["scripts/run_opt2026_10m_gpu_final.py", "scripts/verify_opt2026_10m_gpu_final.py", "experiments/tests/test_opt2026_10m_gpu_final.py"]:
        src = ROOT / rel
        if src.exists():
            dst = ARTIFACT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in ARTIFACT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(ARTIFACT.parent))
    extract_dir = out / "_artifact_extract_check"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(extract_dir)
    pt = list(ARTIFACT.rglob("results/opt2026_10m_gpu_final/**/*.pt"))
    info = {
        "artifact_dir": str(ARTIFACT.relative_to(ROOT)),
        "zip": str(ZIP_PATH.relative_to(ROOT)),
        "gpu_pt_files_included": [str(p.relative_to(ARTIFACT)) for p in pt],
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "fresh_extraction_dir": str(extract_dir.relative_to(ROOT)),
    }
    write_json(out / "artifact_update.json", info)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    require_cuda()
    set_seed(DEV_SEED)
    free_before = ps_drives()
    batches, vocab, dataset = load_batches(DEV_SEED, TRAIN_STEPS + HORIZON + 50)
    free_after = ps_drives()
    hardware = collect_hardware(free_before, free_after)
    write_json(out / "hardware.json", hardware)
    smoke = smoke_test(out, vocab)
    probes = model_probes(out, vocab)
    selected_probe = next(p for p in probes if p["selected_for_replication"])
    if int(selected_probe["trainable_params"]) < 10_000_000:
        raise RuntimeError("selected replicated model is below 10M parameters")
    if set(HELDOUT_SEEDS) & (FORBIDDEN_SEEDS | {DEV_SEED}):
        raise RuntimeError("seed overlap detected")
    protocol = {
        "development_seed": DEV_SEED,
        "heldout_seeds": HELDOUT_SEEDS,
        "forbidden_seed_overlap": sorted(FORBIDDEN_SEEDS),
        "rewrite_families": REWRITE_FAMILIES,
        "stale_age": STALE_AGE,
        "stale_age_translation": "unchanged from final 1.298M campaign; absolute 25-step stale severity preserved",
        "tau_formula": TAU_FORMULA,
        "task_recovery": f"validation-loss gap <= {TASK_RECOVERY_EPS} for {RECOVERY_CONSECUTIVE} consecutive evaluations within {HORIZON} steps",
    }
    write_json(out / "frozen_gpu_protocol.json", protocol)
    write_json(out / "seed_registry.json", {"development_seed": DEV_SEED, "heldout_seeds": HELDOUT_SEEDS, "no_overlap": True, "forbidden": sorted(FORBIDDEN_SEEDS)})
    rows, trajectories, metas, representative = run_campaign(out)
    write_csv(out / "rows.csv", rows)
    write_csv(out / "trajectories.csv", trajectories)
    write_csv(out / "task_metrics.csv", [{k: r[k] for k in ["row_id", "rewrite_family", "training_seed", "checkpoint_step", "parameter_label", "T_tau", "normalized_max_parameter_divergence", "task_damaged", "task_recovery_step", "max_validation_loss_gap", "final_validation_loss_gap", "max_perplexity_gap", "final_perplexity_gap"]} for r in rows])
    sens = tau_sensitivity(rows)
    write_csv(out / "tau_sensitivity.csv", sens)
    write_csv(out / "per_seed.csv", grouped(rows, "training_seed"))
    write_csv(out / "per_checkpoint.csv", grouped(rows, "checkpoint_step"))
    write_csv(out / "per_family.csv", grouped(rows, "rewrite_family"))
    heatmaps(rows, out)
    scatter_plots(rows, out)
    comp = scale_comparison(rows, out)
    rep_hashes = {fam: {"path": rel, "sha256": sha256(ROOT / rel)} for fam, rel in representative.items()}
    write_json(out / "representative_checkpoints.json", rep_hashes)
    xs = [r["normalized_max_parameter_divergence"] for r in rows]
    summary = {
        "dataset": dataset,
        "rows": len(rows),
        "hardware": hardware,
        "smoke_test": smoke,
        "model": {"config": MODEL_CFG, "trainable_params": int(selected_probe["trainable_params"])},
        "runtime": {
            "mean_step_time_seconds": float(np.mean([m["mean_step_time_seconds"] for m in metas])),
            "mean_tokens_per_second": float(np.mean([m["tokens_per_second"] for m in metas])),
            "peak_vram_bytes": max(m["peak_vram_bytes"] for m in metas),
            "per_seed": metas,
        },
        "training": {"steps": TRAIN_STEPS, "heldout_seeds": HELDOUT_SEEDS, "checkpoint_ages": CHECKPOINT_AGES, "precision": PRECISION, "microbatch": 1, "gradient_accumulation": 1, "effective_batch": 1},
        "confusion": confusion(rows),
        "tau_sensitivity": sens,
        "spearman": {
            "normalized_divergence_vs_max_validation_loss_gap": spearman(xs, [r["max_validation_loss_gap"] for r in rows]),
            "normalized_divergence_vs_max_perplexity_gap": spearman(xs, [r["max_perplexity_gap"] for r in rows]),
        },
        "T_tau_distribution": {str(t): sum(r["T_tau"] == t for r in rows) for t in sorted({r["T_tau"] for r in rows if r["T_tau"] != ""})},
        "scale_comparison": comp,
        "representative_checkpoints": rep_hashes,
        "full_eval_failures": FULL_EVAL_FAILURES,
        "negative_results": [
            "Suggested candidate A was below the required threshold at 9,985,600 parameters and was not used for replication.",
            "Candidate C had 25,414,144 parameters and survived a one-step probe, but failed during full rewrite evaluation with CUBLAS_STATUS_EXECUTION_FAILED.",
            "No BF16 was used; GTX 1650 capability was recorded and fp32 was used because the selected 19.5M model fit reliably.",
            "lambda-Opt is reported unchanged as a secondary baseline; no tuning was performed.",
        ],
    }
    write_json(out / "summary.json", summary)
    write_text_outputs(out, summary)
    artifact_info = update_artifact(out, representative)
    summary["artifact"] = artifact_info
    write_json(out / "summary.json", summary)
    write_json(out / "training_metadata.json", metas)
    print("OPT 2026 >=10M GPU final campaign complete.")


if __name__ == "__main__":
    main()






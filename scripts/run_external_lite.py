from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_lite"
STOP_PATH = OUT / "STOPPED_RUNTIME_GATE.json"
FAMILIES = ["exact_restore", "reset_m", "reset_v", "stale_v", "stale_mv", "optimizer_reset"]
SEEDS = [2601, 2602, 2603]
PARAMETER_COUNT_REQUIRED = 57_387_520
RUNTIME_LIMIT_SECONDS = 10 * 60 * 60


@dataclass(frozen=True)
class Config:
    d_model: int = 512
    layers: int = 10
    heads: int = 8
    ffn: int = 2048
    context: int = 256
    dropout: float = 0.0
    effective_batch: int = 8
    train_steps_total: int = 320
    clean_source_steps: int = 300
    continuation_steps: int = 20
    checkpoint_steps: tuple[int, int] = (150, 300)
    warmup_steps: int = 30
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    validation_target_tokens: int = 8192


CFG = Config()


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


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.numpy().tobytes())
    return h.hexdigest()


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.mlp(self.ln2(x)))


class LiteLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok.weight
        self.register_buffer("mask", torch.triu(torch.ones(cfg.context, cfg.context, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        pos = torch.arange(n, device=x.device)
        h = self.tok(x) + self.pos(pos)[None, :, :]
        mask = self.mask[:n, :n]
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln(h))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_at(step: int) -> float:
    if step <= CFG.warmup_steps:
        return CFG.peak_lr * step / CFG.warmup_steps
    progress = (step - CFG.warmup_steps) / max(1, CFG.train_steps_total - CFG.warmup_steps)
    return CFG.min_lr + 0.5 * (CFG.peak_lr - CFG.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def load_prefix_tokens(tokenizer: GPT2TokenizerFast, min_train_tokens: int, min_valid_tokens: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    sep = tokenizer.encode("\n\n", add_special_tokens=False)

    def encode_until(split: str, needed: int) -> list[int]:
        ids: list[int] = []
        for text in ds[split]["text"]:
            if ids:
                ids.extend(sep)
            if text:
                ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(ids) >= needed:
                break
        return ids

    train_ids = encode_until("train", min_train_tokens)
    valid_ids = encode_until("validation", min_valid_tokens)
    return (
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(valid_ids, dtype=torch.long),
        {
            "dataset": "Salesforce/wikitext/wikitext-103-raw-v1",
            "splits": ["train", "validation"],
            "tokenizer": "gpt2",
            "tokenizer_class": "GPT2TokenizerFast",
            "runtime_gate_prefix_tokenization": True,
            "train_tokens_loaded": len(train_ids),
            "validation_tokens_loaded": len(valid_ids),
        },
    )


def batch(tokens: torch.Tensor, seed: int, index: int, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed * 1_000_003 + index)
    starts = torch.randint(0, tokens.numel() - CFG.context - 1, (batch_size,), generator=g)
    x = torch.stack([tokens[s : s + CFG.context] for s in starts])
    y = torch.stack([tokens[s + 1 : s + CFG.context + 1] for s in starts])
    return x, y


def loss_on(model: LiteLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def build(vocab_size: int, device: torch.device, seed: int) -> tuple[LiteLM, torch.optim.AdamW]:
    set_seed(seed)
    model = LiteLM(vocab_size, CFG).to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.peak_lr,
        betas=(CFG.beta1, CFG.beta2),
        eps=CFG.eps,
        weight_decay=CFG.weight_decay,
    )
    return model, opt


def train_step(model: LiteLM, opt: torch.optim.AdamW, train_tokens: torch.Tensor, device: torch.device, seed: int, step: int) -> None:
    set_lr(opt, lr_at(step))
    opt.zero_grad(set_to_none=True)
    for i in range(CFG.effective_batch):
        x, y = batch(train_tokens, seed, (step - 1) * CFG.effective_batch + i, 1)
        loss = loss_on(model, x.to(device, non_blocking=True), y.to(device, non_blocking=True)) / CFG.effective_batch
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()


def runtime_gate(benchmark_steps: int = 2) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_tokens, valid_tokens, data_meta = load_prefix_tokens(
        tokenizer,
        min_train_tokens=(benchmark_steps * CFG.effective_batch + 64) * (CFG.context + 1),
        min_valid_tokens=CFG.validation_target_tokens + CFG.context + 1,
    )
    model, opt = build(tokenizer.vocab_size, device, 999_001)
    params = parameter_count(model)
    if params != PARAMETER_COUNT_REQUIRED:
        report = {
            "status": "STOPPED_IMPLEMENTATION_MISMATCH",
            "reason": "parameter count differs from frozen protocol",
            "parameter_count": params,
            "required_parameter_count": PARAMETER_COUNT_REQUIRED,
        }
        write_json(STOP_PATH, report)
        raise SystemExit(1)

    # Warm one exact optimizer step, then measure the disposable smoke steps.
    train_step(model, opt, train_tokens, device, 999_001, 1)
    start = time.perf_counter()
    measured = []
    for step in range(2, 2 + benchmark_steps):
        s = time.perf_counter()
        train_step(model, opt, train_tokens, device, 999_001, step)
        measured.append(time.perf_counter() - s)
    elapsed = time.perf_counter() - start
    mean_step = elapsed / benchmark_steps

    scientific_train_steps = len(SEEDS) * CFG.train_steps_total
    # Replay uses clean and candidate one-step updates for each seed/family/continuation step.
    replay_update_equivalent_steps = len(SEEDS) * len(FAMILIES) * CFG.continuation_steps * 2 / CFG.effective_batch
    # Validation is forward-only. Count it conservatively as one update-equivalent per eval point/case.
    validation_equivalent_steps = len(SEEDS) * len(FAMILIES) * 4 / CFG.effective_batch
    total_equivalent_steps = scientific_train_steps + replay_update_equivalent_steps + validation_equivalent_steps
    estimated_seconds = mean_step * total_equivalent_steps
    report = {
        "status": "PASS" if estimated_seconds <= RUNTIME_LIMIT_SECONDS else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_limit_seconds": RUNTIME_LIMIT_SECONDS,
        "estimated_seconds": estimated_seconds,
        "estimated_hours": estimated_seconds / 3600,
        "benchmark_steps": benchmark_steps,
        "benchmark_elapsed_seconds": elapsed,
        "benchmark_step_seconds": measured,
        "mean_optimizer_step_seconds": mean_step,
        "estimate_components": {
            "scientific_train_steps": scientific_train_steps,
            "replay_update_equivalent_steps": replay_update_equivalent_steps,
            "validation_equivalent_steps": validation_equivalent_steps,
            "total_equivalent_optimizer_steps": total_equivalent_steps,
        },
        "parameter_count": params,
        "config": asdict(CFG),
        "study_seeds": SEEDS,
        "families": FAMILIES,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
            "datasets": sys.modules["datasets"].__version__,
        },
        "dataset": data_meta,
        "effective_batch_verified": CFG.effective_batch,
        "validation_example_hashes": {
            "prefix_input": sha256_tensor(valid_tokens[: CFG.context]),
            "prefix_target": sha256_tensor(valid_tokens[1 : CFG.context + 1]),
        },
        "git_head": run_cmd(["git", "rev-parse", "HEAD"]),
    }
    write_json(OUT / "runtime_gate.json", report)
    if report["status"] == "FAIL":
        stop = {
            **report,
            "stopped": True,
            "stop_reason": "Projected runtime exceeded 10-hour hard runtime gate.",
            "scientific_cases_run": False,
            "protocol_altered": False,
            "manuscript_edits": False,
        }
        write_json(STOP_PATH, stop)
    return report


def load_science_tokens(tokenizer: GPT2TokenizerFast) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    sep = tokenizer.encode("\n\n", add_special_tokens=False)

    def encode_until(split: str, needed: int) -> list[int]:
        ids: list[int] = []
        for text in ds[split]["text"]:
            if ids:
                ids.extend(sep)
            if text:
                ids.extend(tokenizer.encode(text, add_special_tokens=False))
            if len(ids) >= needed:
                break
        return ids

    train_ids = encode_until("train", 1_000_000)
    valid_ids = encode_until("validation", CFG.validation_target_tokens + CFG.context + 2048)
    meta = {
        "dataset": "Salesforce/wikitext/wikitext-103-raw-v1",
        "splits": ["train", "validation"],
        "tokenizer": "gpt2",
        "tokenizer_class": "GPT2TokenizerFast",
        "train_tokens_loaded": len(train_ids),
        "validation_tokens_loaded": len(valid_ids),
        "validation_split_untouched": True,
    }
    return torch.tensor(train_ids, dtype=torch.long), torch.tensor(valid_ids, dtype=torch.long), meta


def clone_tensor_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: clone_tensor_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_tensor_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tensor_tree(v) for v in obj)
    return obj


def state_hash(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    def feed(prefix: str, obj: Any) -> None:
        if torch.is_tensor(obj):
            arr = obj.detach().cpu().contiguous()
            h.update(prefix.encode("utf-8")); h.update(str(tuple(arr.shape)).encode("ascii")); h.update(str(arr.dtype).encode("ascii")); h.update(arr.numpy().tobytes())
        elif isinstance(obj, dict):
            for k in sorted(obj, key=lambda x: str(x)):
                feed(prefix + "/" + str(k), obj[k])
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                feed(prefix + f"/{i}", v)
        else:
            h.update((prefix + "=" + repr(obj)).encode("utf-8"))
    feed("state", state)
    return h.hexdigest()


def capture_state(model: LiteLM, opt: torch.optim.AdamW, step: int) -> dict[str, Any]:
    return {"model": clone_tensor_tree(model.state_dict()), "optimizer": clone_tensor_tree(opt.state_dict()), "step": step, "lr": lr_at(step)}


def load_state(vocab_size: int, device: torch.device, seed: int, state: dict[str, Any]) -> tuple[LiteLM, torch.optim.AdamW]:
    model, opt = build(vocab_size, device, seed)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    set_lr(opt, state["lr"])
    return model, opt


def rewrite_optimizer(opt: torch.optim.AdamW, family: str, stale_state: dict[str, Any] | None) -> None:
    if family == "exact_restore":
        return
    if family == "optimizer_reset":
        opt.state.clear()
        return
    if family in {"reset_m", "reset_v"}:
        for st in opt.state.values():
            if family == "reset_m" and "exp_avg" in st:
                st["exp_avg"].zero_()
            if family == "reset_v" and "exp_avg_sq" in st:
                st["exp_avg_sq"].zero_()
        return
    if family in {"stale_v", "stale_mv"}:
        if stale_state is None:
            raise RuntimeError("stale_state required")
        stale_opt = stale_state["optimizer"]
        stale_by_id = stale_opt["state"]
        live_sd = opt.state_dict()
        for pid, st in live_sd["state"].items():
            stale = stale_by_id[pid]
            if family == "stale_mv":
                st["exp_avg"] = stale["exp_avg"].to(st["exp_avg"].device).clone()
            st["exp_avg_sq"] = stale["exp_avg_sq"].to(st["exp_avg_sq"].device).clone()
        opt.load_state_dict(live_sd)
        return
    raise ValueError(family)


def move_optimizer_state(opt: torch.optim.Optimizer, device: torch.device) -> None:
    for state in opt.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def move_pair(model: LiteLM, opt: torch.optim.Optimizer, device: torch.device) -> None:
    model.to(device)
    move_optimizer_state(opt, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().flatten().cpu() for p in model.parameters()])


def optimizer_update(model: LiteLM, opt: torch.optim.AdamW, train_tokens: torch.Tensor, device: torch.device, seed: int, step: int, stream_offset: int) -> torch.Tensor:
    before = flat_params(model)
    set_lr(opt, lr_at(step))
    opt.zero_grad(set_to_none=True)
    for i in range(CFG.effective_batch):
        x, y = batch(train_tokens, seed + 50_000, stream_offset + i, 1)
        loss = loss_on(model, x.to(device, non_blocking=True), y.to(device, non_blocking=True)) / CFG.effective_batch
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return flat_params(model) - before


def freeze_validation(valid_tokens: torch.Tensor) -> dict[str, Any]:
    starts = list(range(0, CFG.validation_target_tokens, CFG.context))
    h = hashlib.sha256()
    for st in starts:
        h.update(valid_tokens[st:st + CFG.context + 1].numpy().tobytes())
    return {"starts": starts, "target_tokens": len(starts) * CFG.context, "sha256": h.hexdigest()}


@torch.no_grad()
def eval_validation(model: LiteLM, valid_tokens: torch.Tensor, validation: dict[str, Any], device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0.0
    seen = 0
    for st in validation["starts"]:
        x = valid_tokens[st:st + CFG.context][None, :].to(device)
        y = valid_tokens[st + 1:st + CFG.context + 1][None, :].to(device)
        loss = loss_on(model, x, y)
        total += float(loss.cpu()) * CFG.context
        seen += CFG.context
    model.train()
    nll = total / seen
    return nll, math.exp(min(20.0, nll))


def train_source_seed(seed: int, train_tokens: torch.Tensor, vocab_size: int, device: torch.device) -> dict[str, Any]:
    model, opt = build(vocab_size, device, seed)
    checkpoints: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    for step in range(1, CFG.clean_source_steps + 1):
        train_step(model, opt, train_tokens, device, seed, step)
        if step in CFG.checkpoint_steps:
            checkpoints[step] = capture_state(model, opt, step)
    return {"seed": seed, "checkpoints": checkpoints, "elapsed_seconds": time.perf_counter() - started}


def run_science() -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_tokens, valid_tokens, data_meta = load_science_tokens(tokenizer)
    model_probe, _ = build(tokenizer.vocab_size, device, 12345)
    params = parameter_count(model_probe)
    del model_probe
    if params != PARAMETER_COUNT_REQUIRED:
        raise RuntimeError(f"parameter count mismatch: {params}")
    validation = freeze_validation(valid_tokens)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_gate": json.loads((OUT / "runtime_gate.json").read_text(encoding="utf-8")),
        "parameter_count": params,
        "config": asdict(CFG),
        "study_seeds": SEEDS,
        "families": FAMILIES,
        "checkpoint_evaluated": 300,
        "stale_source_step": 150,
        "dataset": data_meta,
        "validation_examples": validation,
        "rewrite_semantics": {
            "exact_restore": "unaltered step-300 model/optimizer/scheduler",
            "reset_m": "zero exp_avg only, preserve v/counters/hyperparameters/model",
            "reset_v": "zero exp_avg_sq only, preserve m/counters/hyperparameters/model",
            "stale_v": "replace exp_avg_sq with step-150 value, preserve step-300 exp_avg and counters",
            "stale_mv": "replace exp_avg and exp_avg_sq with step-150 values, preserve step-300 counters",
            "optimizer_reset": "clear per-parameter state, preserve model/current LR/param groups/hyperparameters/scheduler",
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": sys.modules["transformers"].__version__,
            "datasets": sys.modules["datasets"].__version__,
        },
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_head": run_cmd(["git", "rev-parse", "HEAD"]),
        "manuscript_edits": False,
        "classifier_metrics": False,
    }
    write_json(OUT / "protocol.json", protocol)
    replay_device = device
    rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    train_meta = []
    eval_points = {0, 1, 5, 20}
    for seed in SEEDS:
        trained = train_source_seed(seed, train_tokens, tokenizer.vocab_size, device)
        train_meta.append({"seed": seed, "elapsed_seconds": trained["elapsed_seconds"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state150 = trained["checkpoints"][150]
        state300 = trained["checkpoints"][300]
        checkpoint_hashes[f"seed_{seed}_step_150"] = state_hash(state150)
        checkpoint_hashes[f"seed_{seed}_step_300"] = state_hash(state300)
        for family in FAMILIES:
            clean_model, clean_opt = load_state(tokenizer.vocab_size, torch.device("cpu"), seed, state300)
            cand_model, cand_opt = load_state(tokenizer.vocab_size, torch.device("cpu"), seed, state300)
            rewrite_optimizer(cand_opt, family, state150)
            s1 = None; clean_update_norm = None; cand_update_norm = None; update_cosine = None
            for k in range(0, CFG.continuation_steps + 1):
                if k in eval_points:
                    move_pair(clean_model, clean_opt, replay_device)
                    cnll, cppl = eval_validation(clean_model, valid_tokens, validation, replay_device)
                    move_pair(clean_model, clean_opt, torch.device("cpu"))
                    move_pair(cand_model, cand_opt, replay_device)
                    vnll, vppl = eval_validation(cand_model, valid_tokens, validation, replay_device)
                    move_pair(cand_model, cand_opt, torch.device("cpu"))
                    eval_rows.append({"seed": seed, "family": family, "k": k, "clean_validation_nll": cnll, "candidate_validation_nll": vnll, "signed_nll_diff": vnll - cnll, "abs_nll_diff": abs(vnll - cnll), "clean_perplexity": cppl, "candidate_perplexity": vppl, "validation_target_tokens": validation["target_tokens"]})
                if k == CFG.continuation_steps:
                    break
                step = 301 + k
                offset = CFG.clean_source_steps * CFG.effective_batch + k * CFG.effective_batch
                move_pair(clean_model, clean_opt, replay_device)
                cu = optimizer_update(clean_model, clean_opt, train_tokens, replay_device, seed, step, offset)
                move_pair(clean_model, clean_opt, torch.device("cpu"))
                move_pair(cand_model, cand_opt, replay_device)
                vu = optimizer_update(cand_model, cand_opt, train_tokens, replay_device, seed, step, offset)
                move_pair(cand_model, cand_opt, torch.device("cpu"))
                cflat = flat_params(clean_model); vflat = flat_params(cand_model)
                dist = float(torch.linalg.vector_norm(vflat - cflat))
                norm_base = float(torch.linalg.vector_norm(cflat))
                if k == 0:
                    clean_update_norm = float(torch.linalg.vector_norm(cu))
                    cand_update_norm = float(torch.linalg.vector_norm(vu))
                    s1 = float(torch.linalg.vector_norm(vu - cu) / (torch.linalg.vector_norm(cu) + 1e-12))
                    update_cosine = float(F.cosine_similarity(vu, cu, dim=0))
                rows.append({"seed": seed, "checkpoint_step": 300, "stale_source_step": 150, "family": family, "continuation_step": k + 1, "S1": s1, "clean_update_norm": clean_update_norm, "candidate_update_norm": cand_update_norm, "update_cosine": update_cosine, "parameter_l2_divergence": dist, "normalized_parameter_divergence": dist / (norm_base + 1e-12), "clean_lr": lr_at(step), "candidate_lr": lr_at(step), "matched_future_minibatches": True, "H": CFG.continuation_steps})
    import csv
    def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
        keys=[]
        for r in data:
            for k in r:
                if k not in keys: keys.append(k)
        with path.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data)
    write_csv(OUT / "rows.csv", rows)
    write_csv(OUT / "validation_rows.csv", eval_rows)
    first = [r for r in rows if r["continuation_step"] == 1]
    seed_summary=[]
    import statistics
    per_family={}
    for fam in FAMILIES:
        vals=[]
        for seed in SEEDS:
            v=[r["S1"] for r in first if r["seed"] == seed and r["family"] == fam][0]
            vals.append(v)
        per_family[fam]={"median_S1": float(statistics.median(vals)), "values": vals}
    def count_dir(a: str, b: str) -> int:
        c=0
        for i in range(len(SEEDS)):
            if per_family[a]["values"][i] > per_family[b]["values"][i]: c += 1
        return c
    for seed in SEEDS:
        row={"seed": seed}
        for fam in FAMILIES:
            row[f"{fam}_S1"]=[r["S1"] for r in first if r["seed"] == seed and r["family"] == fam][0]
        seed_summary.append(row)
    write_csv(OUT / "seed_summary.csv", seed_summary)
    paired={"A_reset_v_gt_reset_m": count_dir("reset_v","reset_m"), "B_stale_v_gt_stale_mv": count_dir("stale_v","stale_mv"), "C_reset_v_gt_optimizer_reset": count_dir("reset_v","optimizer_reset")}
    replicated = per_family["reset_v"]["median_S1"] > per_family["reset_m"]["median_S1"] and per_family["stale_v"]["median_S1"] > per_family["stale_mv"]["median_S1"] and per_family["reset_v"]["median_S1"] > per_family["optimizer_reset"]["median_S1"]
    def val_order(k: int) -> list[str]:
        return sorted(FAMILIES, key=lambda f: sum(e["signed_nll_diff"] for e in eval_rows if e["family"] == f and e["k"] == k) / len(SEEDS))
    summary={"status":"COMPLETE", "parameter_count": params, "seeds_completed": SEEDS, "per_family": per_family, "paired_individual_seed_counts": paired, "phenomenon_replicated": replicated, "validation_nll_ordering_k1": val_order(1), "validation_nll_ordering_k20": val_order(20), "checkpoint_hashes": checkpoint_hashes, "elapsed_runtime_seconds": time.perf_counter() - started, "train_meta": train_meta}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "MANIFEST.json", {"files": sorted([str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file() and 'checkpoint' not in p.parts]), "created_utc": datetime.now(timezone.utc).isoformat()})
    md=["# External Lite Summary", f"Status: {summary['status']}", f"Parameter count: {params}", f"Replicated: {replicated}"]
    for fam in FAMILIES: md.append(f"- {fam}: median S1={per_family[fam]['median_S1']}, values={per_family[fam]['values']}")
    (OUT / "summary.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-steps", type=int, default=2)
    args = parser.parse_args()
    report = runtime_gate(args.benchmark_steps)
    if report["status"] == "FAIL":
        print(json.dumps({"status": "STOPPED_RUNTIME_GATE", "estimated_hours": report["estimated_hours"]}, indent=2))
        return
    summary = run_science(); print(json.dumps({"status": "COMPLETE", "elapsed_runtime_seconds": summary["elapsed_runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()










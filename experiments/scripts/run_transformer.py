"""
GPT-mini / WikiText-2 → Plot 9C.

Self-contained transformer experiment for λOpt.
Architecture: 4-layer decoder-only transformer, d_model=128, 4 heads.
~3.2M parameters. Fits in GTX 1650 4GB VRAM with batch_size=16, seq_len=128.

Dropout is forced to 0.0 for determinism (Assumption A5).
Dataset: WikiText-2 via torchtext OR character-level fallback if torchtext
unavailable (downloads ~5MB of raw text).

Usage:
    cd experiments/
    python scripts/run_transformer.py --config configs/gpt_mini_wikitext2.yaml --out results/
"""

import argparse
import json
import math
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src import (
    AdamStep,
    EMARewrite,
    TrainingState,
    apriori_lipschitz_numerical,
    bound_holds,
    loss_gap,
    make_constant_schedule,
)
from src.bounds import apriori_bound, classify_regime
from src.metrics import effective_lipschitz_from_divergence

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GPTMini(nn.Module):
    """Minimal decoder-only transformer. No causal mask complexity — uses
    nn.TransformerEncoder with a causal mask built at forward time."""

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, seq_len=128, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True,  # pre-norm: more stable loss landscape
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, T) token ids
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(positions)
        # causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        return self.head(h)  # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

WIKITEXT2_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
)


def _download_wikitext2(cache_dir="./data/wikitext2_raw"):
    """Download raw WikiText-2 train split (~5MB). Character-level."""
    os.makedirs(cache_dir, exist_ok=True)
    fpath = Path(cache_dir) / "train.txt"
    if not fpath.exists():
        print(f"  Downloading WikiText-2 train split...")
        urllib.request.urlretrieve(WIKITEXT2_URL, fpath)
        print(f"  Saved to {fpath}")
    return fpath.read_text(encoding="utf-8")


def build_char_dataset(text, seq_len, batch_size, seed=42):
    """Character-level: encode text → integer tensor, build (input, target) pairs."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)

    # chunk into non-overlapping sequences
    n_seq = len(data) // seq_len
    data = data[:n_seq * seq_len].view(n_seq, seq_len)

    # shuffle with fixed seed for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_seq, generator=g)
    data = data[perm]

    # batch
    n_batch = n_seq // batch_size
    batches = []
    for i in range(n_batch):
        chunk = data[i * batch_size:(i + 1) * batch_size]  # (B, seq_len)
        x = chunk[:, :-1]    # (B, seq_len-1) — input
        y = chunk[:, 1:]     # (B, seq_len-1) — target (next char)
        batches.append((x, y))

    return batches, vocab_size


# ---------------------------------------------------------------------------
# Param flatten / unflatten
# ---------------------------------------------------------------------------

def flatten_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def unflatten_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].reshape(p.shape))
        idx += n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config, out):
    cfg = yaml.safe_load(open(config))
    os.makedirs(out, exist_ok=True)

    T = cfg["training"]["T"]
    seed = cfg["training"]["seed"]
    bs = cfg["training"]["batch_size"]
    seq_len = cfg["model"]["seq_len"]
    opt_cfg = cfg["optimizer"]
    lip_cfg = cfg["lipschitz"]
    safety = lip_cfg.get("safety_margin", 1.03)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Dataset ---
    print("Loading WikiText-2 (character-level)...")
    text = _download_wikitext2()
    batches, vocab_size = build_char_dataset(text, seq_len, bs, seed=seed)
    print(f"  vocab_size={vocab_size}, total batches={len(batches)}, seq_len={seq_len}")

    if len(batches) < T:
        raise RuntimeError(
            f"Not enough data: need {T} batches but only have {len(batches)}. "
            f"Reduce T or batch_size in the config."
        )

    # --- Model ---
    model = GPTMini(
        vocab_size=vocab_size,
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        seq_len=seq_len - 1,   # input is seq_len-1 after slicing
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.eval()  # dropout=0 but belt-and-suspenders for determinism

    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-mini: {n_params:,} parameters")

    criterion = nn.CrossEntropyLoss()
    schedule = make_constant_schedule(
        lr=opt_cfg["lr"], beta1=opt_cfg["beta1"],
        beta2=opt_cfg["beta2"], eps=opt_cfg["eps"],
    )
    step_fn = AdamStep()

    # --- Loss function (closure over model, criterion, device, batches) ---
    call_counter = [0]

    def loss_fn(theta_flat):
        unflatten_params(model, theta_flat.to(device))
        idx = call_counter[0] % len(batches)
        call_counter[0] += 1
        x, y = batches[idx]
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            # forward without autograd for baseline trajectory
            pass
        # need grad for the step
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        logits = model(x)                         # (B, T-1, vocab)
        B, Tm1, V = logits.shape
        loss = criterion(logits.reshape(B * Tm1, V), y.reshape(B * Tm1))
        loss.backward()
        grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        return loss.item(), grad.detach().cpu()

    # --- Initial state ---
    theta0 = flatten_params(model).cpu()
    s0 = TrainingState(
        theta=theta0,
        moments=torch.zeros(n_params, 2),
        schedule_fn=schedule,
        t=0,
    )

    t_R = int(T * cfg["rewrites"]["rewrite_times"][0])
    print(f"\nT={T}, t_R={t_R}")

    # --- Phase 1: run baseline up to t_R, save state ---
    print(f"\nPhase 1: baseline trajectory to t_R={t_R}...")
    call_counter[0] = 0
    s = s0.clone()
    base_losses = []
    for step in range(t_R):
        loss, grad = loss_fn(s.theta)
        base_losses.append(loss)
        s, _ = step_fn(s, grad)
        if step % 20 == 0:
            print(f"  step {step}/{t_R}  loss={loss:.4f}", end="\r")
    print(f"  step {t_R}/{t_R}  loss={base_losses[-1]:.4f}  [baseline to t_R done]")
    s_at_tR = s.clone()

    # --- Phase 2: a priori Lipschitz at t_R ---
    print(f"\nPhase 2: numerical a priori L at t_R={t_R}...")
    call_counter[0] = t_R   # use same batch sequence as training
    L_raw = apriori_lipschitz_numerical(
        s_at_tR, step_fn, loss_fn,
        n_perturbations=lip_cfg["n_perturbations"],
        n_steps=lip_cfg["n_steps"],
        eps=lip_cfg["eps"],
    )
    L_pred = L_raw * safety
    regime = classify_regime(L_pred)
    print(f"  L_raw={L_raw:.6f}  L_pred={L_pred:.6f}  regime={regime}")

    # --- Phase 3: EMA rewrite ---
    torch.manual_seed(seed + 1)
    R = EMARewrite(
        theta_ema=s_at_tR.theta + torch.randn(n_params) * 0.001,
        alpha=cfg["rewrites"]["ema_alpha"],
    )
    delta = R.delta(s_at_tR)
    print(f"  delta={delta:.6f}")

    # --- Phase 4: run baseline and rewritten trajectories in parallel ---
    print(f"\nPhase 3: parallel trajectories t_R={t_R} → T={T}...")
    s_bl = s_at_tR.clone()
    s_rw = R.apply(s_at_tR)

    # reset counter to replay same batches from t_R
    call_counter_bl = [t_R]
    call_counter_rw = [t_R]

    def loss_fn_ctr(ctr):
        def _fn(theta_flat):
            unflatten_params(model, theta_flat.to(device))
            idx = ctr[0] % len(batches)
            ctr[0] += 1
            x, y = batches[idx]
            x, y = x.to(device), y.to(device)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            logits = model(x)
            B, Tm1, V = logits.shape
            loss = criterion(logits.reshape(B * Tm1, V), y.reshape(B * Tm1))
            loss.backward()
            grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            return loss.item(), grad.detach().cpu()
        return _fn

    lf_bl = loss_fn_ctr(call_counter_bl)
    lf_rw = loss_fn_ctr(call_counter_rw)

    divergence = []
    rw_losses = []
    bl_losses_post = []
    remaining = T - t_R

    for step in range(remaining):
        div = (s_rw.theta - s_bl.theta).norm(2).item()
        divergence.append(div)

        loss_bl, grad_bl = lf_bl(s_bl.theta)
        loss_rw, grad_rw = lf_rw(s_rw.theta)

        bl_losses_post.append(loss_bl)
        rw_losses.append(loss_rw)

        s_bl, _ = step_fn(s_bl, grad_bl)
        s_rw, _ = step_fn(s_rw, grad_rw)

        if step % 10 == 0:
            print(f"  step {t_R + step}/{T}  div={div:.6f}  loss_bl={loss_bl:.4f}", end="\r")

    # final divergence
    divergence.append((s_rw.theta - s_bl.theta).norm(2).item())
    print(f"\n  done. final_div={divergence[-1]:.6f}")

    # --- Metrics ---
    bnd = apriori_bound(delta, L_pred, 0, remaining)  # relative to t_R
    holds = bound_holds(divergence, bnd)
    L_actual = effective_lipschitz_from_divergence(divergence, 0)
    lgap = loss_gap(bl_losses_post, rw_losses)

    print(f"\n{'='*50}")
    print(f"GPT-mini / WikiText-2 Results")
    print(f"{'='*50}")
    print(f"  n_params  : {n_params:,}")
    print(f"  L_pred    : {L_pred:.6f}  ({regime})")
    print(f"  L_actual  : {L_actual:.6f}")
    print(f"  delta     : {delta:.6f}")
    print(f"  max_div   : {max(divergence):.6f}")
    print(f"  final_div : {divergence[-1]:.6f}")
    print(f"  bound_holds: {'PASS' if holds else 'FAIL'}")
    print(f"  max_loss_gap: {max(lgap):.6f}")

    results = {
        "model": "gpt_mini",
        "n_params": n_params,
        "vocab_size": vocab_size,
        "d_model": cfg["model"]["d_model"],
        "num_layers": cfg["model"]["num_layers"],
        "nhead": cfg["model"]["nhead"],
        "L_pred": L_pred,
        "L_raw": L_raw,
        "L_actual": L_actual,
        "regime": regime,
        "delta": delta,
        "t_R": t_R,
        "T": T,
        "divergence": divergence,
        "apriori_bound": bnd,
        "bound_holds": holds,
        "loss_gap": lgap,
        "base_losses": base_losses,
    }

    out_path = Path(out) / "neural_net_gpt_mini_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=lambda x: (
            x if isinstance(x, (int, float, bool, str))
            else list(x) if hasattr(x, "__iter__")
            else str(x)
        ))
    print(f"\nSaved → {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/gpt_mini_wikitext2.yaml")
    p.add_argument("--out", default="results")
    run(**vars(p.parse_args()))
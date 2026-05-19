
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from src.bounds import is_vacuous


def bound_status(bound, holds):
    if is_vacuous(bound):
        return "VACUOUS", "orange"
    return ("HOLDS", "green") if holds else ("VIOLATED", "red")


def load(d, name):
    p = Path(d) / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def plot1(data, out):
    """Divergence vs a priori bound (3 rewrites, log scale)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, kind in zip(axes, ["EMA", "Checkpoint", "LoRA"]):
        runs = [r for r in data["runs"] if r["rewrite"] == kind and r["frac"] == 0.5]
        if not runs: continue
        r = runs[0]
        t_R = r["t_R"]
        t = np.arange(t_R, len(r["divergence"]))
        div_post = r["divergence"][t_R:]
        bnd_post = r["apriori_bound"][t_R:]
        ax.semilogy(t, [max(d, 1e-15) for d in div_post], "b-", lw=1.5, label="D(t)")
        ax.semilogy(t, [max(b, 1e-15) for b in bnd_post], "r--", lw=1.5,
                    label=f"bound (L={r['L_pred']:.3f})")
        ax.set_title(f"{kind} [{r['regime']}]")
        ax.set_xlabel("step"); ax.legend(fontsize=7)
    axes[0].set_ylabel("||Δθ|| (log)")
    fig.suptitle("Plot 1: A Priori Bound vs Divergence", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / "plot1_divergence_vs_bound.pdf", dpi=150)
    plt.close()


def plot2(data, out):
    """Tightness ratio vs rewrite time."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for kind in ["EMA", "Checkpoint", "LoRA"]:
        fracs, tights = [], []
        for r in data["runs"]:
            if r["rewrite"] == kind:
                t_R = r["t_R"]
                post = [t for i, t in enumerate(r["tightness"]) if i >= t_R and 0 < t < 1]
                fracs.append(r["frac"])
                tights.append(float(np.median(post)) if post else 0)
        ax.plot(fracs, tights, "o-", label=kind, ms=8, lw=2)
    ax.set_xlabel("t_R / T"); ax.set_ylabel("median D/B")
    ax.set_title("Plot 2: Bound Tightness"); ax.legend()
    ax.set_ylim(0, 1.1); ax.axhline(1, color="gray", ls=":", alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out) / "plot2_tightness.pdf", dpi=150)
    plt.close()


def plot3(data, out):
    """Compositional bound."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t1 = data["t1"]
    t = np.arange(t1, len(data["divergence"]))
    ax.semilogy(t, [max(d, 1e-15) for d in data["divergence"][t1:]], "b-", lw=1.5, label="D(t)")
    ax.semilogy(t, [max(b, 1e-15) for b in data["compositional_bound"][t1:]], "r--", lw=1.5,
                label=f"Σ bound (L={data['L_pred']:.3f})")
    ax.axvline(data["t1"], color="gray", ls=":", alpha=0.5, label=f"R₁@{data['t1']}")
    ax.axvline(data["t2"], color="orange", ls=":", alpha=0.5, label=f"R₂@{data['t2']}")
    ax.set_xlabel("step"); ax.set_ylabel("||Δθ|| (log)")
    ax.set_title(f"Plot 3: Compositional Bound [{data['regime']}]"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "plot3_composition.pdf", dpi=150)
    plt.close()


def plot4(data, out):
    """Local L vs a priori L."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    L_est = data["L_local_estimates"]
    ax.plot(range(len(L_est)), L_est, "g-", alpha=0.4, lw=0.8, label="local L (finite-diff)")
    ax.axhline(data["L_local"], color="green", ls="--", alpha=0.5,
               label=f"local median={data['L_local']:.1f}")
    # a priori L from runs
    L_preds = [r["L_pred"] for r in data["runs"] if r["frac"] == 0.5]
    if L_preds:
        ax.axhline(np.mean(L_preds), color="red", lw=2, label=f"L_pred={np.mean(L_preds):.4f}")
    ax.axhline(1, color="black", ls=":", alpha=0.3, label="L=1 boundary")
    ax.set_xlabel("step"); ax.set_ylabel("L")
    ax.set_title("Plot 4: Local vs A Priori Lipschitz"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "plot4_lipschitz.pdf", dpi=150)
    plt.close()


def plot5(data, out):
    """δ-scaling linearity."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(data["deltas"], data["divergence_at_T"], "bo-", ms=7, lw=1.5)
    c = np.polyfit(data["deltas"], data["divergence_at_T"], 1)
    x = np.linspace(min(data["deltas"]), max(data["deltas"]), 50)
    ax.plot(x, np.polyval(c, x), "r--", lw=1.5, label=f"R²={data['R2']:.4f}")
    ax.set_xlabel("δ"); ax.set_ylabel("D(T)")
    ax.set_title("Plot 5: δ-Scaling Linearity"); ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out) / "plot5_delta_scaling.pdf", dpi=150)
    plt.close()


def plot6(data, out):
    """Loss gap vs θ-divergence with real Spearman."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    all_d, all_l = [], []
    for r in data["runs"]:
        if r["frac"] == 0.5 and "loss_gap" in r:
            t_R = r["t_R"]
            d = r["divergence"][t_R:-1]
            l = r["loss_gap"][t_R:]
            m = min(len(d), len(l))
            all_d.extend(d[:m]); all_l.extend(l[:m])
    if all_d:
        # filter out near-zero pairs that cause nan correlation
        pairs = [(d, l) for d, l in zip(all_d, all_l) if d > 1e-10 and l > 1e-15]
        if pairs:
            fd, fl = zip(*pairs)
            ax.scatter(fd, fl, alpha=0.4, s=10, c="steelblue")
            rho, _ = stats.spearmanr(fd, fl)
            import math
            rho = rho if not math.isnan(rho) else 0.0
            ax.set_title(f"Plot 6: Loss Gap vs θ-Divergence (ρ={rho:.3f})")
        else:
            ax.set_title("Plot 6: Loss Gap vs θ-Divergence")
    ax.set_xlabel("||Δθ||"); ax.set_ylabel("|ΔL|")
    fig.tight_layout()
    fig.savefig(Path(out) / "plot6_loss_correlation.pdf", dpi=150)
    plt.close()


def plot7(data, out):
    """Separation: contractive vs expansive side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [(axes[0], "contractive", "Safe (Adam, L<1)"),
                            (axes[1], "expansive", "Dangerous (SGD, L>1)")]:
        r = data[key]
        t_R = r["t_R"]
        T_len = len(r["divergence"])
        t = np.arange(t_R, T_len)
        div_post = r["divergence"][t_R:]
        bnd_post = r["apriori_bound"][t_R:]
        ax.semilogy(t, [max(d, 1e-15) for d in div_post], "b-", lw=2, label="D(t)")
        ax.semilogy(t, [max(b, 1e-15) for b in bnd_post], "r--", lw=2,
                    label=f"bound (L={r['L_pred']:.3f})")
        grew = "↑ GREW" if r.get("divergence_grew", False) else "↓ DECAYED"
        opt_label = r.get("optimizer", "")
        ax.set_title(f"{title}: L={r['L_pred']:.3f} [{opt_label}]\n{grew}", fontsize=11)
        ax.set_xlabel("step"); ax.legend(fontsize=8)
    axes[0].set_ylabel("||Δθ|| (log)")
    fig.suptitle("Plot 7: SEPARATION — Same Loss, Same Rewrite, Different Optimizer", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(out) / "plot7_separation.pdf", dpi=150)
    plt.close()


def plot8(data, out):
    """A priori L_pred vs actual L_eff scatter."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    pts = data["points"]
    preds = [p["L_pred"] for p in pts]
    actuals = [p["L_actual"] for p in pts]

    # color by config
    configs = list(set(p["config"] for p in pts))
    colors = {"contractive": "green", "boundary": "orange", "expansive": "red"}
    for p in pts:
        label = [k for k in colors if k in p["config"]]
        c = colors[label[0]] if label else "blue"
        ax.scatter(p["L_pred"], p["L_actual"], c=c, s=50, alpha=0.7)

    lims = [min(min(preds), min(actuals)) * 0.95, max(max(preds), max(actuals)) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (perfect prediction)")
    ax.axhline(1, color="gray", ls=":", alpha=0.3)
    ax.axvline(1, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("L_pred (a priori)"); ax.set_ylabel("L_actual (measured)")
    ax.set_title("Plot 8: Prediction Accuracy"); ax.legend()

    # add legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=patches + [plt.Line2D([0],[0], ls="--", c="k", label="y=x")])
    fig.tight_layout()
    fig.savefig(Path(out) / "plot8_apriori_vs_actual.pdf", dpi=150)
    plt.close()


def plot9(data, out):
    """Neural net bound."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t_R = data["t_R"]
    t = np.arange(t_R, len(data["divergence"]))
    ax.semilogy(t, [max(d, 1e-15) for d in data["divergence"][t_R:]], "b-", lw=1.5, label="D(t)")
    ax.semilogy(t, [max(b, 1e-15) for b in data["apriori_bound"][t_R:]], "r--", lw=1.5,
                label=f"bound (L={data['L_pred']:.3f})")
    ax.set_xlabel("step"); ax.set_ylabel("||Δθ|| (log)")
    status, scolor = bound_status(data["apriori_bound"], data["bound_holds"])
    ax.set_title(f"Plot 9: {data['model']} ({data['n_params']} params) [{data['regime']}]")
    ax.text(0.98, 0.05, status, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=scolor, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out) / f"plot9_neural_{data['model']}.pdf", dpi=150)
    plt.close()

def plot9c(data, out):
    """Plot 9C: GPT-mini divergence vs a priori bound."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: divergence vs bound (log scale)
    ax = axes[0]
    div = data["divergence"]
    bnd = data["apriori_bound"]
    t = np.arange(len(div))
    ax.semilogy(t, [max(d, 1e-15) for d in div], "b-", lw=1.5, label="D(t)")
    ax.semilogy(t, [max(b, 1e-15) for b in bnd], "r--", lw=1.5,
                label=f"bound (L={data['L_pred']:.4f})")
    ax.set_xlabel("steps after rewrite")
    ax.set_ylabel("‖Δθ‖ (log)")
    ax.set_title(
        f"Plot 9C: GPT-mini ({data['n_params']:,} params)\n"
        f"[{data['regime']}] L_pred={data['L_pred']:.4f}, L_actual={data['L_actual']:.4f}"
    )
    ax.legend(fontsize=8)
    status, scolor = bound_status(data["apriori_bound"], data["bound_holds"])
    ax.text(0.98, 0.05, status, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color=scolor, fontweight="bold")

    # Right: loss gap over same window
    ax2 = axes[1]
    lgap = data["loss_gap"]
    ax2.plot(np.arange(len(lgap)), lgap, "purple", lw=1.2, alpha=0.8)
    ax2.set_xlabel("steps after rewrite")
    ax2.set_ylabel("|ΔL|")
    ax2.set_title(f"Plot 9C: Loss Gap (GPT-mini)\nδ={data['delta']:.4f}")

    fig.suptitle(
        f"GPT-mini / WikiText-2 — Transformer Checkpoint Safety\n"
        f"vocab={data['vocab_size']}, d_model={data['d_model']}, "
        f"layers={data['num_layers']}, heads={data['nhead']}",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(Path(out) / "plot9c_transformer.pdf", dpi=150)
    plt.close()
    print(f"Saved plot9c_transformer.pdf")

def plot10(data, out):
    """Checkpoint safety analysis: 3 scenarios side by side."""
    scenarios = data["scenarios"]
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    colors = {"stable recovery": "green", "mild drift": "orange", "divergence growth": "red"}

    for ax, (name, s) in zip(axes, scenarios.items()):
        div = s["divergence"]
        bnd = s["bound"]
        t = np.arange(len(div))

        ax.semilogy(t, [max(d, 1e-15) for d in div], "b-", lw=2, label="D(t)")
        if max(bnd) > 0:
            ax.semilogy(t, [max(b, 1e-15) for b in bnd], "r--", lw=1.5,
                        label=f"bound (L={s['L_pred']:.3f})")

        # color title by prediction correctness
        mark = "CORRECT" if s["correct"] else "WRONG"
        c = "green" if s["correct"] else "red"
        title = f"{name}\npred: {s['prediction']}\nactual: {s['actual_outcome']} [{mark}]"
        ax.set_title(title, fontsize=9, color=c)
        ax.set_xlabel("steps after rewrite")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("||Δθ|| (log)")
    fig.suptitle("Plot 10: Checkpoint Safety Analysis — ResNet-18/CIFAR-10",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(out) / "plot10_checkpoint_safety.pdf", dpi=150)
    plt.close()


def main(results, out):
    os.makedirs(out, exist_ok=True)

    sr = load(results, "single_rewrite_results.json")
    if sr:
        plot1(sr, out); plot2(sr, out)
        plot4(sr, out); plot6(sr, out)

    comp = load(results, "composition_results.json")
    if comp: plot3(comp, out)

    ds = load(results, "delta_scaling_results.json")
    if ds: plot5(ds, out)

    sep = load(results, "separation_results.json")
    if sep: plot7(sep, out)

    ap = load(results, "apriori_vs_actual_results.json")
    if ap: plot8(ap, out)

    for name in ["neural_net_mlp_results.json", "neural_net_resnet18_results.json"]:
        nn_data = load(results, name)
        if nn_data: plot9(nn_data, out)

    cs = load(results, "checkpoint_safety_results.json")
    if cs: plot10(cs, out)

    gpt = load(results, "neural_net_gpt_mini_results.json")
    if gpt: plot9c(gpt, out)

    print(f"Plots saved to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    main(**vars(p.parse_args()))
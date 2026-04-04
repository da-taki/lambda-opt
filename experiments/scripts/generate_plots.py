"""Generate all 9 λOpt plots."""
import argparse, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats


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
        ax.scatter(all_d, all_l, alpha=0.4, s=10, c="steelblue")
        rho, _ = stats.spearmanr(all_d, all_l)
        ax.set_title(f"Plot 6: Loss Gap vs θ-Divergence (ρ={rho:.3f})")
    ax.set_xlabel("||Δθ||"); ax.set_ylabel("|ΔL|")
    fig.tight_layout()
    fig.savefig(Path(out) / "plot6_loss_correlation.pdf", dpi=150)
    plt.close()


def plot7(data, out):
    """Separation: contractive vs expansive side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, title in [(axes[0], "contractive", "Safe (L<1)"),
                            (axes[1], "expansive", "Dangerous (L>1)")]:
        r = data[key]
        t_R = r["t_R"]
        t = np.arange(t_R, len(r["divergence"]))
        div_post = r["divergence"][t_R:]
        bnd_post = r["apriori_bound"][t_R:]
        ax.semilogy(t, [max(d, 1e-15) for d in div_post], "b-", lw=2, label="D(t)")
        ax.semilogy(t, [max(b, 1e-15) for b in bnd_post], "r--", lw=2,
                    label=f"bound (L={r['L_pred']:.3f})")
        grew = "↑ GREW" if r["divergence_grew"] else "↓ DECAYED"
        ax.set_title(f"{title}: L_pred={r['L_pred']:.3f}\n{grew}", fontsize=11)
        ax.set_xlabel("step"); ax.legend(fontsize=8)
    axes[0].set_ylabel("||Δθ|| (log)")
    fig.suptitle("Plot 7: SEPARATION — Same Rewrite, Different Regimes", fontsize=13, fontweight="bold")
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
    ax.set_title(f"Plot 9: {data['model']} ({data['n_params']} params) [{data['regime']}]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out) / f"plot9_neural_{data['model']}.pdf", dpi=150)
    plt.close()


def main(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    sr = load(results_dir, "single_rewrite_results.json")
    if sr:
        plot1(sr, out_dir); plot2(sr, out_dir)
        plot4(sr, out_dir); plot6(sr, out_dir)

    comp = load(results_dir, "composition_results.json")
    if comp: plot3(comp, out_dir)

    ds = load(results_dir, "delta_scaling_results.json")
    if ds: plot5(ds, out_dir)

    sep = load(results_dir, "separation_results.json")
    if sep: plot7(sep, out_dir)

    ap = load(results_dir, "apriori_vs_actual_results.json")
    if ap: plot8(ap, out_dir)

    for name in ["neural_net_mlp_results.json", "neural_net_resnet18_results.json"]:
        nn_data = load(results_dir, name)
        if nn_data: plot9(nn_data, out_dir)

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    main(**vars(p.parse_args()))
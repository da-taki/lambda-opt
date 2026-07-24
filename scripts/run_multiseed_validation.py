"""Experiment 4: Multiple-seed validation.

Repeats the canonical analyzer over several seeds per diagnostic and reports
mean +/- std of every metric. Targets:
  quadratic : 5 seeds
  MNIST MLP : 3 seeds
  ResNet-18 : 3 seeds if feasible (default 1 on CPU, documented)
  GPT-mini  : 3 seeds if feasible (default 1 on CPU, documented)

Per-run metrics: delta_linearity_r2, spearman_loss_divergence, false_negatives,
false_positives, certified_safe, flagged_dangerous, abstained, bound_failures,
coverage_rate, runtime_seconds.

Usage:
    .venv\\Scripts\\python.exe scripts/run_multiseed_validation.py
"""
import argparse
import statistics
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_experiments_common as C
from src.metrics import delta_linearity_r2, spearman_correlation

RUN_FIELDS = [
    "model", "dataset", "seed", "n_scenarios",
    "delta_linearity_r2", "spearman_loss_divergence",
    "false_negatives", "false_positives", "certified_safe", "flagged_dangerous",
    "abstained", "bound_failures", "coverage_rate", "runtime_seconds",
]

NUMERIC_METRICS = [
    "delta_linearity_r2", "spearman_loss_divergence", "false_negatives",
    "false_positives", "certified_safe", "flagged_dangerous", "abstained",
    "bound_failures", "coverage_rate", "runtime_seconds",
]

DEFAULT_SEEDS = {
    "quadratic": [0, 1, 2, 3, 4],
    "mnist_mlp": [0, 1, 2],
    "resnet18": [0],
    "gpt_mini": [0],
}


def per_run_metrics(rows):
    deltas = [r["delta"] for r in rows]
    maxdiv = [r["actual_max_divergence"] for r in rows]
    lgap = [r["actual_loss_gap"] for r in rows]
    n = len(rows)
    cert = sum(1 for r in rows if r["decision"] == "certified_safe")
    flag = sum(1 for r in rows if r["decision"] == "flagged_dangerous")
    abst = sum(1 for r in rows if r["decision"] == "abstained")
    fp = sum(1 for r in rows if r["decision"] == "flagged_dangerous" and r["actual_label"] == "benign")
    fn = sum(1 for r in rows if r["decision"] == "certified_safe" and r["actual_label"] == "dangerous")
    bound_fail = sum(1 for r in rows if not r["bound_holds_fixed"])
    return {
        "delta_linearity_r2": delta_linearity_r2(deltas, maxdiv),
        "spearman_loss_divergence": spearman_correlation(lgap, maxdiv),
        "false_negatives": fn, "false_positives": fp,
        "certified_safe": cert, "flagged_dangerous": flag, "abstained": abst,
        "bound_failures": bound_fail,
        "coverage_rate": (cert + flag) / n if n else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["quadratic", "mnist_mlp", "resnet18", "gpt_mini"],
                   choices=list(C.DIAGNOSTIC_BUILDERS))
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=1.08)
    p.add_argument("--seeds-quadratic", nargs="+", type=int, default=DEFAULT_SEEDS["quadratic"])
    p.add_argument("--seeds-mnist_mlp", nargs="+", type=int, default=DEFAULT_SEEDS["mnist_mlp"])
    p.add_argument("--seeds-resnet18", nargs="+", type=int, default=DEFAULT_SEEDS["resnet18"])
    p.add_argument("--seeds-gpt_mini", nargs="+", type=int, default=DEFAULT_SEEDS["gpt_mini"])
    args = p.parse_args()
    seed_map = {m: getattr(args, f"seeds_{m}") for m in C.DIAGNOSTIC_BUILDERS}

    C.ensure_results_dir()
    run_rows = []
    all_scenario_rows = []
    skipped = {}

    for model in args.models:
        for seed in seed_map[model]:
            t0 = time.perf_counter()
            try:
                ctx = C.DIAGNOSTIC_BUILDERS[model](seed=seed)
                srows = C.full_scenario_analysis(ctx, args.epsilon, args.margin, seed)
            except Exception as e:  # noqa: BLE001
                skipped[f"{model}/seed{seed}"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                continue
            runtime = time.perf_counter() - t0
            m = per_run_metrics(srows)
            m.update({"model": model, "dataset": srows[0]["dataset"], "seed": seed,
                      "n_scenarios": len(srows), "runtime_seconds": runtime})
            run_rows.append(m)
            all_scenario_rows.extend(srows)
            print(f"{model:10s} seed{seed}: cov={m['coverage_rate']:.2f} FN={m['false_negatives']} "
                  f"FP={m['false_positives']} r2={m['delta_linearity_r2']:.3f} "
                  f"rho={m['spearman_loss_divergence']:.3f} t={runtime:.1f}s")

    # aggregate mean/std per model
    summary_rows = []
    for model in sorted(set(r["model"] for r in run_rows)):
        sub = [r for r in run_rows if r["model"] == model]
        agg = {"model": model, "dataset": sub[0]["dataset"], "n_seeds": len(sub)}
        for metric in NUMERIC_METRICS:
            vals = [r[metric] for r in sub]
            agg[f"{metric}_mean"] = statistics.mean(vals) if vals else None
            agg[f"{metric}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(agg)

    summary_fields = ["model", "dataset", "n_seeds"] + \
        [f"{m}_{s}" for m in NUMERIC_METRICS for s in ("mean", "std")]

    out = C.RESULTS_DIR
    C.save_csv(out / "multiseed_validation_runs.csv", run_rows, RUN_FIELDS)
    C.save_csv(out / "multiseed_validation_summary.csv", summary_rows, summary_fields)
    C.save_json(out / "multiseed_validation.json", {
        "runs": run_rows, "summary": summary_rows, "skipped": skipped,
        "config": {"models": args.models, "seed_map": seed_map,
                   "epsilon": args.epsilon, "margin": args.margin}})

    # markdown
    lines = ["# Experiment 4 -- Multiple-seed validation", "", C.ANALYZER_DISCLAIMER, "",
             "## Setup", "",
             "The canonical analyzer is run across multiple seeds per diagnostic; we "
             "report mean +/- standard deviation of every numeric metric. "
             "`delta_linearity_r2` is the R^2 of repaired-delta vs max divergence "
             "across the 12 scenarios; `spearman_loss_divergence` is Spearman rho "
             "between max loss-gap and max divergence.", "",
             "Exact command:", "", "```",
             "python scripts/run_multiseed_validation.py", "```", "",
             f"Seeds used: " + "; ".join(f"{m}={seed_map[m]}" for m in args.models) + ".", "",
             "## Mean +/- std by model", ""]
    hdr = ["model", "seeds", "coverage", "FN", "FP", "abstained", "bound failures",
           "delta-R^2", "loss/div rho"]
    trows = []
    for a in summary_rows:
        def ms(metric, pct=False):
            mean = a[f"{metric}_mean"]; std = a[f"{metric}_std"]
            if mean is None:
                return "n/a"
            if pct:
                return f"{mean:.2f}+/-{std:.2f}"
            if abs(mean) >= 100 or (mean != 0 and abs(mean) < 0.01):
                return f"{mean:.3g}+/-{std:.2g}"
            return f"{mean:.2f}+/-{std:.2f}"
        trows.append([a["model"], a["n_seeds"], ms("coverage_rate"), ms("false_negatives"),
                      ms("false_positives"), ms("abstained"), ms("bound_failures"),
                      ms("delta_linearity_r2"), ms("spearman_loss_divergence")])
    lines.append(C.md_table(hdr, trows))
    if skipped:
        lines += ["", "## Skipped runs", ""]
        for k, v in skipped.items():
            lines.append(f"* `{k}`: {v}")
    lines += ["", "## Notes & limitations", "",
              "* Quadratic uses 5 seeds, MNIST MLP 3; ResNet-18 and GPT-mini default to "
              "1 seed on CPU (each big-model run re-pretrains + analyzes 12 scenarios "
              "with two sampled-L estimates apiece, which is the dominant cost). Raise "
              "`--seeds-resnet18`/`--seeds-gpt_mini` to add seeds on a GPU host.",
              "* `false_negatives` is the headline robustness metric: a dangerous "
              "rewrite certified safe. Across seeds it should stay at/near zero.",
              "* `L` is sampled; all decisions are empirical analyzer decisions."]
    C.save_text(out / "multiseed_validation.md", "\n".join(lines))

    print("\n=== SUMMARY (mean) ===")
    for a in summary_rows:
        print(f"  {a['model']:10s} seeds={a['n_seeds']} cov={a['coverage_rate_mean']:.3f}"
              f"+/-{a['coverage_rate_std']:.3f} FN={a['false_negatives_mean']:.2f} "
              f"r2={a['delta_linearity_r2_mean']:.3f}")
    if skipped:
        print("  skipped:", skipped)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

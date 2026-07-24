"""Experiment 2: Certification coverage.

How often does the repaired analyzer return a *useful* decision
(certified_safe or flagged_dangerous) instead of abstaining?

Runs the canonical analyzer over the 12 scenarios for each available
diagnostic (quadratic, MNIST MLP, ResNet-18/CIFAR-10, GPT-mini) and reports the
coverage rate, abstention rate, and the confusion against the threshold-defined
ground truth.

Usage:
    .venv\\Scripts\\python.exe scripts/run_decision_coverage.py
    .venv\\Scripts\\python.exe scripts/run_decision_coverage.py --models quadratic mnist_mlp resnet18 gpt_mini
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_experiments_common as C

CSV_FIELDS = [
    "model", "dataset", "seed", "scenario", "delta",
    "L_clean", "L_rewritten", "L_max", "L_stepwise",
    "fixed_bound_final", "stepwise_bound_final",
    "actual_max_divergence", "actual_loss_gap", "epsilon", "safety_margin",
    "decision", "decision_reason", "runtime_seconds",
]


def confusion(rows):
    """Confusion vs ground truth, abstentions excluded from FP/FN."""
    cert = sum(1 for r in rows if r["decision"] == "certified_safe")
    flag = sum(1 for r in rows if r["decision"] == "flagged_dangerous")
    abst = sum(1 for r in rows if r["decision"] == "abstained")
    n = len(rows)
    fp = sum(1 for r in rows if r["decision"] == "flagged_dangerous" and r["actual_label"] == "benign")
    fn = sum(1 for r in rows if r["decision"] == "certified_safe" and r["actual_label"] == "dangerous")
    tp = sum(1 for r in rows if r["decision"] == "flagged_dangerous" and r["actual_label"] == "dangerous")
    tn = sum(1 for r in rows if r["decision"] == "certified_safe" and r["actual_label"] == "benign")
    return {
        "total_cases": n,
        "certified_safe": cert, "flagged_dangerous": flag, "abstained": abst,
        "coverage_rate": (cert + flag) / n if n else 0.0,
        "abstention_rate": abst / n if n else 0.0,
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["quadratic", "mnist_mlp", "resnet18"],
                   choices=list(C.DIAGNOSTIC_BUILDERS))
    p.add_argument("--small-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--big-seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=1.08)
    args = p.parse_args()

    C.ensure_results_dir()
    rows = []
    for model in args.models:
        for seed in C.seeds_for_model(model, args.small_seeds, args.big_seeds):
            ctx = C.DIAGNOSTIC_BUILDERS[model](seed=seed)
            print(f"\n=== {model} (seed {seed}) n_params={ctx.n_params}")
            srows = C.full_scenario_analysis(ctx, args.epsilon, args.margin, seed)
            for r in srows:
                print(f"  {r['scenario']:38s} dec={r['decision']:17s} "
                      f"delta={r['delta']:.3g} L_max={r['L_max']:.3f} actual={r['actual_label']}")
            rows.extend(srows)

    overall = confusion(rows)
    by_model = {}
    for model in sorted(set(r["model"] for r in rows)):
        by_model[model] = confusion([r for r in rows if r["model"] == model])

    # most common abstention reason
    from collections import Counter
    reasons = Counter(r["decision_reason"] for r in rows if r["decision"] == "abstained")
    main_reason = reasons.most_common(1)[0] if reasons else ("none", 0)

    out = C.RESULTS_DIR
    C.save_csv(out / "certification_coverage.csv", rows, CSV_FIELDS)
    C.save_json(out / "certification_coverage.json", {
        "rows": rows, "overall": overall, "by_model": by_model,
        "abstention_reasons": dict(reasons),
        "config": {"models": args.models, "small_seeds": args.small_seeds,
                   "big_seeds": args.big_seeds, "epsilon": args.epsilon, "margin": args.margin}})

    lines = ["# Experiment 2 -- Certification coverage", "", C.ANALYZER_DISCLAIMER, "",
             "## Setup", "",
             "The canonical repaired analyzer (repaired displacement + "
             "`L=max(L_clean,L_rewritten)` + abstain) is run on all 12 scenarios for "
             "each diagnostic. Coverage = fraction of cases with a non-abstaining "
             "decision. Ground-truth danger is threshold-defined (see per-row "
             "`div_threshold`/`loss_gap_threshold`).", "",
             "Exact command:", "", "```",
             "python scripts/run_decision_coverage.py --models "
             + " ".join(args.models), "```", "",
             "## Overall", "",
             C.md_table(
                 ["metric", "value"],
                 [["total cases", overall["total_cases"]],
                  ["certified safe", overall["certified_safe"]],
                  ["flagged dangerous", overall["flagged_dangerous"]],
                  ["abstained", overall["abstained"]],
                  ["coverage rate", overall["coverage_rate"]],
                  ["abstention rate", overall["abstention_rate"]],
                  ["false positives", overall["false_positives"]],
                  ["false negatives", overall["false_negatives"]]]),
             "", f"Main reason for abstention: **{main_reason[0]}** ({main_reason[1]} cases).",
             "", "## By model", "",
             C.md_table(
                 ["model", "cases", "certified", "flagged", "abstained",
                  "coverage", "abstention", "FP", "FN"],
                 [[m, c["total_cases"], c["certified_safe"], c["flagged_dangerous"],
                   c["abstained"], c["coverage_rate"], c["abstention_rate"],
                   c["false_positives"], c["false_negatives"]]
                  for m, c in by_model.items()]),
             "", "## Reading", "",
             f"* Coverage rate overall is **{overall['coverage_rate']:.1%}** "
             f"({overall['certified_safe'] + overall['flagged_dangerous']}/"
             f"{overall['total_cases']} cases decided).",
             f"* Abstention rate is **{overall['abstention_rate']:.1%}**; the dominant "
             f"cause is `{main_reason[0]}`.",
             f"* False negatives: **{overall['false_negatives']}** "
             "(a dangerous rewrite certified safe). "
             f"False positives: **{overall['false_positives']}** "
             "(a benign rewrite flagged dangerous -- the acceptable conservative cost).",
             "", "## Limitations", "",
             "* `L` is sampled, so `certified_safe` is an empirical analyzer verdict.",
             "* ResNet-18/GPT-mini rows use the small CPU-budget configs; coverage on "
             "those is indicative, not production-scale.",
             "* Abstention is the designed, honest output when sampled `L` sits on the "
             "contraction boundary or the bound falls between epsilon and the danger "
             "threshold."]
    C.save_text(out / "certification_coverage.md", "\n".join(lines))

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

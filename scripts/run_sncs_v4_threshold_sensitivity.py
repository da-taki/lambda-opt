"""SNCS v4 -- Experiment 8: Threshold sensitivity and ROC-style sweep.

Sweeps the analyzer's epsilon (certification tolerance) and safety margin and
reports sensitivity / specificity / FPR / FNR / coverage / abstention.

To stay cheap, per-scenario displacement and *raw* (unmargined) L are computed
once per (model, seed); the epsilon x margin grid then only re-evaluates the
(very cheap) decision rule.

epsilon grid: 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0
margin grid : 1.0, 1.05, 1.1, 1.25, 1.5, 2.0

Usage:
    .venv\\Scripts\\python.exe scripts/run_sncs_v4_threshold_sensitivity.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sncs_v4_common as C

CSV_FIELDS = [
    "model", "dataset", "epsilon", "margin", "sensitivity", "specificity",
    "false_positive_rate", "false_negative_rate", "coverage_rate",
    "abstention_rate", "num_cases", "notes",
]

EPSILON_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
MARGIN_GRID = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0]


def precompute(ctx, seed):
    """Per-scenario delta, raw L (margin=1), actual label -- computed once."""
    _, _ = C.estimate_L(ctx.resume_state, ctx.step_fn, ctx.make_loss_fn(ctx.pretrain_steps),
                        n_perturbations=ctx.lip_perturbations, n_steps=ctx.lip_steps,
                        eps=ctx.lip_eps, safety_margin=1.0)
    L_raw_clean, _ = C.estimate_L(ctx.resume_state, ctx.step_fn,
                                  ctx.make_loss_fn(ctx.pretrain_steps),
                                  n_perturbations=ctx.lip_perturbations,
                                  n_steps=ctx.lip_steps, eps=ctx.lip_eps, safety_margin=1.0)
    data = []
    for sc in C.build_scenarios(ctx):
        L_raw_rw, _ = C.estimate_L_for_state(ctx, sc.state_after, margin=1.0)
        delta = C.repaired_delta(ctx.resume_state, sc.state_after)
        gt = C.run_ground_truth(ctx, sc, steps=ctx.gt_steps)
        data.append({"model": ctx.name, "dataset": ctx.dataset, "seed": seed,
                     "scenario": sc.name, "delta": delta,
                     "L_raw_clean": L_raw_clean, "L_raw_rw": L_raw_rw,
                     "actual_label": gt.actual_label, "post_steps": ctx.post_steps,
                     "danger": ctx.div_threshold})
    return data


def evaluate(data, epsilon, margin):
    tp = tn = fp = fn = abst = 0
    for d in data:
        L_used = max(d["L_raw_clean"], d["L_raw_rw"]) * margin
        dec = C.analyzer_decision(d["delta"], L_used, d["post_steps"], epsilon, margin, d["danger"])
        actual_danger = d["actual_label"] == "dangerous"
        if dec.decision == "abstained":
            abst += 1
        elif dec.decision == "flagged_dangerous":
            if actual_danger:
                tp += 1
            else:
                fp += 1
        else:  # certified_safe
            if actual_danger:
                fn += 1
            else:
                tn += 1
    n = len(data)
    decided_pos = tp + fn  # decided dangerous
    decided_neg = tn + fp  # decided benign
    return {
        "sensitivity": tp / decided_pos if decided_pos else None,
        "specificity": tn / decided_neg if decided_neg else None,
        "false_positive_rate": fp / decided_neg if decided_neg else None,
        "false_negative_rate": fn / decided_pos if decided_pos else None,
        "coverage_rate": (n - abst) / n if n else 0.0,
        "abstention_rate": abst / n if n else 0.0,
        "num_cases": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "abst": abst,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["quadratic", "mnist_mlp", "resnet18"],
                   choices=list(C.DIAGNOSTIC_BUILDERS))
    p.add_argument("--small-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--big-seeds", nargs="+", type=int, default=[0])
    args = p.parse_args()

    C.ensure_results_dir()
    all_data = []
    for model in args.models:
        for seed in C.seeds_for_model(model, args.small_seeds, args.big_seeds):
            ctx = C.DIAGNOSTIC_BUILDERS[model](seed=seed)
            print(f"precomputing {model} seed{seed} ...")
            all_data.extend(precompute(ctx, seed))

    rows = []
    models = sorted(set(d["model"] for d in all_data)) + ["ALL"]
    for model in models:
        data = all_data if model == "ALL" else [d for d in all_data if d["model"] == model]
        dataset = "all" if model == "ALL" else data[0]["dataset"]
        for eps in EPSILON_GRID:
            for margin in MARGIN_GRID:
                m = evaluate(data, eps, margin)
                rows.append({
                    "model": model, "dataset": dataset, "epsilon": eps, "margin": margin,
                    "sensitivity": m["sensitivity"], "specificity": m["specificity"],
                    "false_positive_rate": m["false_positive_rate"],
                    "false_negative_rate": m["false_negative_rate"],
                    "coverage_rate": m["coverage_rate"], "abstention_rate": m["abstention_rate"],
                    "num_cases": m["num_cases"],
                    "notes": f"tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']} abst={m['abst']}",
                })

    # analysis on ALL
    all_rows = [r for r in rows if r["model"] == "ALL"]

    def fn_count(r):
        # parse fn from notes
        return int(r["notes"].split("fn=")[1].split()[0])

    zero_fn = [r for r in all_rows if fn_count(r) == 0]
    nonzero_fn = [r for r in all_rows if fn_count(r) > 0]
    # best balance = maximize sensitivity+specificity (Youden's J) with decent coverage
    def youden(r):
        if r["sensitivity"] is None or r["specificity"] is None:
            return -9
        return r["sensitivity"] + r["specificity"] - 1
    decided = [r for r in all_rows if r["sensitivity"] is not None and r["specificity"] is not None
               and r["coverage_rate"] >= 0.5]
    best = max(decided, key=youden) if decided else None

    out = C.RESULTS_DIR
    C.save_csv(out / "threshold_sensitivity.csv", rows, CSV_FIELDS)
    C.save_json(out / "threshold_sensitivity.json", {
        "rows": rows,
        "analysis": {
            "n_grid_points": len(all_rows),
            "grid_points_with_zero_FN": len(zero_fn),
            "grid_points_with_nonzero_FN": len(nonzero_fn),
            "best_balance": best,
        },
        "config": {"models": args.models, "epsilon_grid": EPSILON_GRID,
                   "margin_grid": MARGIN_GRID}})

    # ALL-model sensitivity table (rows=epsilon, cols=margin) and coverage table
    def grid_table(metric):
        hdr = ["epsilon \\ margin"] + [str(mg) for mg in MARGIN_GRID]
        trows = []
        for eps in EPSILON_GRID:
            row = [str(eps)]
            for mg in MARGIN_GRID:
                r = next(x for x in all_rows if x["epsilon"] == eps and x["margin"] == mg)
                v = r[metric]
                row.append("n/a" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
            trows.append(row)
        return C.md_table(hdr, trows)

    lines = ["# Experiment 8 -- Threshold sensitivity and ROC-style sweep", "",
             C.ANALYZER_DISCLAIMER, "",
             "## Setup", "",
             "Per-scenario displacement and raw L are computed once; the epsilon x "
             "margin grid only re-evaluates the decision rule. Sensitivity/specificity "
             "are computed over **decided** cases (abstentions reported separately as "
             "coverage/abstention).", "",
             "Exact command:", "", "```",
             "python scripts/run_sncs_v4_threshold_sensitivity.py --models "
             + " ".join(args.models), "```", "",
             "## ALL-models sensitivity (rows = epsilon, cols = margin)", "",
             grid_table("sensitivity"),
             "", "## ALL-models specificity", "", grid_table("specificity"),
             "", "## ALL-models coverage rate", "", grid_table("coverage_rate"),
             "", "## ALL-models false-negative rate", "", grid_table("false_negative_rate"),
             "", "## Answers to the required questions", "",
             f"**1. Is zero false negatives only caused by an extreme threshold?** "
             f"{len(zero_fn)}/{len(all_rows)} of the grid points have zero false "
             f"negatives. " + (
                 "Zero-FN holds across a broad region of the grid, not just an extreme "
                 "corner -- the no-false-negative property is robust to threshold choice."
                 if len(zero_fn) > len(all_rows) * 0.3 else
                 "Zero-FN is confined to a narrow (conservative) corner of the grid; the "
                 "claim is threshold-sensitive and must be stated with the operating point."),
             "",
             f"**2. What threshold gives the best balance?** "
             + (f"epsilon={best['epsilon']}, margin={best['margin']} "
                f"(sensitivity={best['sensitivity']:.2f}, specificity={best['specificity']:.2f}, "
                f"coverage={best['coverage_rate']:.2f}) by Youden's J among points with "
                f">=50% coverage." if best else "no decided point with sufficient coverage."),
             "",
             "**3. How sensitive are claims to epsilon?** Reading across each row "
             "(fixed margin), increasing epsilon converts abstentions/flags into "
             "`certified_safe`, raising coverage but eventually admitting false "
             "negatives. See the coverage and FNR grids above.", "",
             f"**4. What operating point should the paper report?** "
             + (f"Report epsilon={best['epsilon']}, margin={best['margin']} as the "
                f"balanced operating point, and explicitly state that zero-FN holds over "
                f"{len(zero_fn)}/{len(all_rows)} grid points. Pair any 'zero false "
                f"negative' claim with this operating point rather than presenting it as "
                f"threshold-free." if best else "see CSV."), "",
             "## Limitations", "",
             "* Sensitivity/specificity are over decided cases; a conservative operating "
             "point trades coverage for safety via abstention.",
             "* `L` is sampled; the grid inherits that empirical caveat.",
             "* Ground-truth danger thresholds (divergence/loss-gap) are fixed here; "
             "they are separate from the analyzer's epsilon/margin."]
    C.save_text(out / "threshold_sensitivity.md", "\n".join(lines))

    print(f"\nzero-FN grid points: {len(zero_fn)}/{len(all_rows)}")
    if best:
        print(f"best balance: eps={best['epsilon']} margin={best['margin']} "
              f"sens={best['sensitivity']:.2f} spec={best['specificity']:.2f} cov={best['coverage_rate']:.2f}")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

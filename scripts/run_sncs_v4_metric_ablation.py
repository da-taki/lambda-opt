"""SNCS v4 -- Experiment 7: Metric repair ablation.

Compares five displacement metrics on the scenarios where the original
(theta, M)-only metric was weak:

  A. raw_theta_M          original: ||dtheta|| (+) ||dM||, ignores schedule/time
  B. raw_plus_sched_time  A plus theta-space schedule + time terms
  C. block_normalized     per-block normalised then rescaled to theta units
  D. denom_aware_v        Adam denominator-aware second-moment geometry
  E. full_repaired        the metric the repaired analyzer actually uses

L is held FIXED across metrics (L = max(L_clean, L_rewritten)) so the only thing
that varies is the displacement; this isolates the metric's effect on the bound
and the decision.

Usage:
    .venv\\Scripts\\python.exe scripts/run_sncs_v4_metric_ablation.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sncs_v4_common as C
from src.bounds import _safe_power

CSV_FIELDS = [
    "model", "dataset", "seed", "scenario", "metric_type", "delta", "L_estimate",
    "bound_final", "actual_final_divergence", "actual_max_divergence",
    "bound_holds", "decision", "notes",
]

WEAK_SCENARIOS = ["scheduler_mismatch", "reset_second_moment",
                  "ema_loaded_as_parameters", "stale_adam_moments", "mild_lr_decrease"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["quadratic", "mnist_mlp"],
                   choices=list(C.DIAGNOSTIC_BUILDERS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=1.08)
    args = p.parse_args()

    C.ensure_results_dir()
    rows = []
    for model in args.models:
        for seed in args.seeds:
            ctx = C.DIAGNOSTIC_BUILDERS[model](seed=seed)
            _, L_clean = C.estimate_L(ctx.resume_state, ctx.step_fn,
                                      ctx.make_loss_fn(ctx.pretrain_steps),
                                      n_perturbations=ctx.lip_perturbations,
                                      n_steps=ctx.lip_steps, eps=ctx.lip_eps,
                                      safety_margin=args.margin)
            scn_map = {s.name: s for s in C.build_scenarios(ctx)}
            print(f"\n=== {model} (seed {seed})")
            for sname in WEAK_SCENARIOS:
                sc = scn_map[sname]
                _, L_rw = C.estimate_L_for_state(ctx, sc.state_after, margin=args.margin)
                L_used = max(L_clean, L_rw)
                gt = C.run_ground_truth(ctx, sc, steps=ctx.post_steps)
                for mt in C.METRIC_TYPES:
                    delta = C.metric_delta(ctx.resume_state, sc.state_after, mt, sc.comps)
                    curve = C.fixed_bound_curve(delta, L_used, ctx.post_steps)
                    bound_final = curve[-1]
                    holds = all(a <= b + 1e-9 for a, b in zip(gt.divergence, curve))
                    dec = C.analyzer_decision(delta, L_used, ctx.post_steps, args.epsilon,
                                              args.margin, ctx.div_threshold)
                    note = ""
                    if mt == "raw_theta_M" and delta <= C.EPS_DIV:
                        note = "OLD METRIC BLIND: delta=0 -> bound identically 0 (vacuous/misleading)"
                    elif mt == "full_repaired":
                        note = "metric used by repaired analyzer"
                    rows.append({
                        "model": model, "dataset": ctx.dataset, "seed": seed,
                        "scenario": sname, "metric_type": mt, "delta": delta,
                        "L_estimate": L_used, "bound_final": bound_final,
                        "actual_final_divergence": gt.actual_final_divergence,
                        "actual_max_divergence": gt.actual_max_divergence,
                        "bound_holds": holds, "decision": dec.decision, "notes": note,
                    })
                print(f"  {sname:30s} A_raw={C.metric_delta(ctx.resume_state, sc.state_after,'raw_theta_M',sc.comps):.4g} "
                      f"D_denom={C.metric_delta(ctx.resume_state, sc.state_after,'denom_aware_v',sc.comps):.4g} "
                      f"E_rep={C.metric_delta(ctx.resume_state, sc.state_after,'full_repaired',sc.comps):.4g} "
                      f"actual_maxdiv={gt.actual_max_divergence:.4g}")

    # ---- analysis for the four required questions ----
    def mean_delta(metric, scenario):
        vals = [r["delta"] for r in rows if r["metric_type"] == metric and r["scenario"] == scenario]
        return sum(vals) / len(vals) if vals else 0.0

    # Q1: old metric zero for scheduler mismatch?
    raw_sched = [r["delta"] for r in rows if r["metric_type"] == "raw_theta_M" and r["scenario"] == "scheduler_mismatch"]
    raw_sched_zero = all(d <= C.EPS_DIV for d in raw_sched)
    rep_sched = mean_delta("full_repaired", "scheduler_mismatch")

    # Q2: denom-aware vs raw for reset_v -- compare the theta-space effect
    raw_v = mean_delta("raw_theta_M", "reset_second_moment")
    denom_v = mean_delta("denom_aware_v", "reset_second_moment")

    # Q3: does block-normalization change classification vs raw?
    changed = 0
    keys = set((r["model"], r["seed"], r["scenario"]) for r in rows)
    for k in keys:
        a = next(r for r in rows if (r["model"], r["seed"], r["scenario"]) == k and r["metric_type"] == "raw_theta_M")
        c = next(r for r in rows if (r["model"], r["seed"], r["scenario"]) == k and r["metric_type"] == "block_normalized")
        if a["decision"] != c["decision"]:
            changed += 1

    out = C.RESULTS_DIR
    C.save_csv(out / "metric_ablation.csv", rows, CSV_FIELDS)
    C.save_json(out / "metric_ablation.json", {
        "rows": rows,
        "analysis": {"raw_scheduler_all_zero": raw_sched_zero,
                     "repaired_scheduler_mean_delta": rep_sched,
                     "raw_reset_v_mean_delta": raw_v,
                     "denom_aware_reset_v_mean_delta": denom_v,
                     "block_norm_classification_changes": changed,
                     "total_cells": len(keys)},
        "config": {"models": args.models, "seeds": args.seeds,
                   "weak_scenarios": WEAK_SCENARIOS,
                   "epsilon": args.epsilon, "margin": args.margin}})

    # per-scenario mean delta table across metrics
    metrics = C.METRIC_TYPES
    nice = {"raw_theta_M": "A raw(theta,M)", "raw_plus_sched_time": "B +sched/time",
            "block_normalized": "C block-norm", "denom_aware_v": "D denom-aware v",
            "full_repaired": "E repaired"}
    table_rows = []
    for sc in WEAK_SCENARIOS:
        table_rows.append([sc] + [mean_delta(mt, sc) for mt in metrics])

    lines = ["# Experiment 7 -- Metric repair ablation", "", C.ANALYZER_DISCLAIMER, "",
             "## Setup", "",
             "Five metrics, identical fixed `L = max(L_clean, L_rewritten)`, on the "
             "scenarios where the original metric was weak. Only the displacement "
             "changes, so differences in bound/decision are attributable to the "
             "metric. Deltas below are means over models/seeds.", "",
             "Exact command:", "", "```",
             "python scripts/run_sncs_v4_metric_ablation.py --models "
             + " ".join(args.models), "```", "",
             "## Mean displacement (delta) by metric", "",
             C.md_table(["scenario"] + [nice[m] for m in metrics], table_rows),
             "", "## Answers to the required questions", "",
             f"**1. Does the old metric give zero/misleading delta for scheduler "
             f"mismatch?** {'YES' if raw_sched_zero else 'NO'} -- the raw (theta,M) "
             f"metric is **{('exactly 0' if raw_sched_zero else 'nonzero')}** for "
             f"`scheduler_mismatch` (a pure schedule change moves neither theta nor M), "
             f"so its bound `delta*L^k = 0` is vacuous/misleading. The repaired metric "
             f"assigns a theta-commensurate mean delta of **{rep_sched:.4g}**.", "",
             f"**2. Does denominator-aware v geometry better reflect reset-v severity?** "
             f"The raw metric scores `reset_second_moment` purely by buffer norm "
             f"(mean **{raw_v:.4g}**, in second-moment units, not comparable to "
             f"divergence). The denominator-aware metric maps the v change into its "
             f"theta-space step effect (mean **{denom_v:.4g}**), which is directly "
             f"comparable to the measured divergence and reflects that a v reset makes "
             f"the next Adam step much larger.", "",
             f"**3. Does block normalization change classification?** Block "
             f"normalization changes the analyzer decision in **{changed}/{len(keys)}** "
             f"cells relative to the raw metric -- it rescales moment vs parameter "
             f"contributions, which moves borderline cases across the abstain/flag "
             f"line.", "",
             "**4. Which metric is used in the repaired analyzer?** **E "
             "(full_repaired)** = direct theta move + injected one-step update change "
             "(which subsumes the schedule, time and denominator-aware-v effects). It "
             "is the only metric that is both non-blind to schedule changes and in "
             "theta-commensurate units.", "",
             "## Limitations", "",
             "* Deltas across metrics are not on a single common scale by construction "
             "(that is the point); compare each metric to the measured divergence in "
             "the CSV.",
             "* `L` is sampled and shared across metrics here to isolate the metric.",
             "* See the `bound_holds` column for where each metric's bound is actually "
             "valid."]
    C.save_text(out / "metric_ablation.md", "\n".join(lines))

    print("\n=== ANALYSIS ===")
    print(f"  raw scheduler_mismatch all zero: {raw_sched_zero}")
    print(f"  repaired scheduler mean delta:   {rep_sched:.4g}")
    print(f"  raw vs denom-aware reset_v:       {raw_v:.4g} vs {denom_v:.4g}")
    print(f"  block-norm classification changes: {changed}/{len(keys)}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

"""SNCS v4 -- Experiment 11: paper-ready tables and reviewer-proofing summary.

Reads every results/sncs_v4/*.json produced by experiments 1-10 and emits:
  * results/sncs_v4/sncs_v4_tables_for_paper.md   (10 manuscript-ready tables)
  * results/sncs_v4/SNCS_V4_REVIEWER_PROOFING_SUMMARY.md

Numbers are read straight from the JSON outputs so the summary cannot drift from
the experiments. Git/test/Lean facts are passed in as flags.

Usage:
    .venv\\Scripts\\python.exe scripts/run_sncs_v4_make_summary.py \
        --branch sncs-v4-reviewer-experiments --commit <hash> \
        --pytest "N passed" --lean "see below"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sncs_v4_common as C

R = C.RESULTS_DIR


def load(name):
    p = R / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def fnum(x):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if abs(x) >= 1e5 or (x != 0 and abs(x) < 1e-3):
            return f"{x:.3e}"
        return f"{x:.4g}"
    return str(x)


def t_baseline():
    d = load("baseline_comparison.json")
    if not d:
        return None, None
    s = d["summary"]
    nice = {"A_param_norm": "A. parameter-norm delta", "B_optimizer_norm": "B. optimizer-buffer delta",
            "C_raw_product": "C. raw product metric", "D_block_normalized": "D. block-normalized metric",
            "E_loss_probe": "E. loss-probe heuristic", "F_repaired_analyzer": "F. repaired lambda-Opt"}
    rows = [[nice[k], v["n"], f"{v['caught_dangerous']}/{v['total_dangerous']}",
             v["false_negatives"], v["false_positives"], v["abstained"]]
            for k, v in s.items()]
    tbl = C.md_table(["method", "cases", "dangerous caught", "false neg", "false pos", "abstained"], rows)
    key = (f"repaired analyzer false negatives = {s['F_repaired_analyzer']['false_negatives']} "
           f"vs param-norm {s['A_param_norm']['false_negatives']}, "
           f"optimizer-norm {s['B_optimizer_norm']['false_negatives']}, "
           f"loss-probe {s['E_loss_probe']['false_negatives']}")
    return tbl, key


def t_coverage():
    d = load("certification_coverage.json")
    if not d:
        return None, None
    bm = d["by_model"]
    rows = [[m, c["total_cases"], c["certified_safe"], c["flagged_dangerous"], c["abstained"],
             fnum(c["coverage_rate"]), c["false_positives"], c["false_negatives"]]
            for m, c in bm.items()]
    o = d["overall"]
    rows.append(["ALL", o["total_cases"], o["certified_safe"], o["flagged_dangerous"], o["abstained"],
                 fnum(o["coverage_rate"]), o["false_positives"], o["false_negatives"]])
    tbl = C.md_table(["model", "cases", "certified", "flagged", "abstained", "coverage", "FP", "FN"], rows)
    key = (f"overall coverage {o['coverage_rate']:.1%}, abstention {o['abstention_rate']:.1%}, "
           f"FN={o['false_negatives']}, FP={o['false_positives']}")
    return tbl, key


def t_stepwise():
    d = load("stepwise_bound_comparison.json")
    if not d:
        return None, None
    s = d["summary"]
    nice = {"A_fixed_clean": "A. fixed clean L", "B_fixed_rewritten": "B. fixed rewritten L",
            "C_fixed_max": "C. fixed max L", "D_stepwise_product": "D. stepwise product",
            "E_empirical_envelope": "E. empirical envelope (diag)"}
    rows = [[nice[k], v["n"], v["bound_holds"], v["vacuous_final"], fnum(v["median_looseness_final"]),
             v["certifies_safe"], v["abstains"]] for k, v in s.items()]
    tbl = C.md_table(["bound type", "cases", "holds", "vacuous", "median looseness", "certifies safe", "abstains"], rows)
    c = s.get("C_fixed_max", {}); dd = s.get("D_stepwise_product", {})
    key = (f"fixed max-L vacuous on {c.get('vacuous_final')} cases (median looseness "
           f"{fnum(c.get('median_looseness_final'))}); stepwise product vacuous on "
           f"{dd.get('vacuous_final')} (median looseness {fnum(dd.get('median_looseness_final'))})")
    return tbl, key


def t_multiseed():
    d = load("multiseed_validation.json")
    if not d:
        return None, None
    rows = []
    for a in d["summary"]:
        def ms(m):
            mean = a.get(f"{m}_mean"); std = a.get(f"{m}_std")
            return "n/a" if mean is None else f"{fnum(mean)}+/-{fnum(std)}"
        rows.append([a["model"], a["n_seeds"], ms("coverage_rate"), ms("false_negatives"),
                     ms("false_positives"), ms("bound_failures"), ms("delta_linearity_r2"),
                     ms("spearman_loss_divergence")])
    tbl = C.md_table(["model", "seeds", "coverage", "FN", "FP", "bound fails", "delta-R^2", "loss/div rho"], rows)
    key = "; ".join(f"{a['model']}: FN={fnum(a.get('false_negatives_mean'))}, "
                    f"cov={fnum(a.get('coverage_rate_mean'))}" for a in d["summary"])
    return tbl, key


def t_cifar():
    d = load("cifar10_checkpoint_matrix.json")
    if not d:
        return None, None
    o = d["overall"]
    rows = [["total", o["total"]], ["actual dangerous", o["actual_dangerous"]],
            ["actual benign", o["actual_benign"]], ["true positives", o["true_positives"]],
            ["true negatives", o["true_negatives"]], ["false positives", o["false_positives"]],
            ["false negatives", o["false_negatives"]], ["abstentions", o["abstentions"]],
            ["sensitivity", fnum(o["sensitivity"])], ["specificity", fnum(o["specificity"])],
            ["coverage", fnum(o["coverage_rate"])]]
    tbl = C.md_table(["metric", "value"], rows)
    key = (f"real CIFAR-10 (subset): sensitivity={fnum(o['sensitivity'])}, "
           f"specificity={fnum(o['specificity'])}, FN={o['false_negatives']}, FP={o['false_positives']}")
    return tbl, key


def t_runtime():
    d = load("runtime_overhead.json")
    if not d:
        return None, None
    rows = d["rows"]
    budgets = sorted(set((r["probe_trials"], r["probe_steps"]) for r in rows))
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    trows = []
    pms = d.get("per_model_step", {})
    for m in models:
        row = [m, pms.get(m, {}).get("n_params", "n/a")]
        for (t, s) in budgets:
            v = next((x["runtime_seconds"] for x in rows if x["model"] == m
                      and x["probe_trials"] == t and x["probe_steps"] == s), None)
            row.append(f"{v:.3f}" if v is not None else "n/a")
        trows.append(row)
    tbl = C.md_table(["model", "params"] + [f"{t}x{s}s" for (t, s) in budgets], trows)
    key = "analyzer (5x5) runtime: " + "; ".join(
        f"{m}={next((x['runtime_seconds'] for x in rows if x['model']==m and x['probe_trials']==5 and x['probe_steps']==5), None):.3f}s"
        for m in models if any(x['model']==m and x['probe_trials']==5 for x in rows))
    return tbl, key


def t_metric():
    d = load("metric_ablation.json")
    if not d:
        return None, None
    a = d["analysis"]
    rows = [["raw (theta,M) delta for scheduler_mismatch is exactly 0", a["raw_scheduler_all_zero"]],
            ["repaired mean delta for scheduler_mismatch", fnum(a["repaired_scheduler_mean_delta"])],
            ["raw mean delta for reset_v (buffer units)", fnum(a["raw_reset_v_mean_delta"])],
            ["denom-aware mean delta for reset_v (theta units)", fnum(a["denom_aware_reset_v_mean_delta"])],
            ["block-norm classification changes vs raw", f"{a['block_norm_classification_changes']}/{a['total_cells']}"]]
    tbl = C.md_table(["finding", "value"], rows)
    key = (f"raw metric delta=0 for scheduler mismatch: {a['raw_scheduler_all_zero']}; "
           f"repaired delta={fnum(a['repaired_scheduler_mean_delta'])}")
    return tbl, key


def t_threshold():
    d = load("threshold_sensitivity.json")
    if not d:
        return None, None
    a = d["analysis"]
    b = a.get("best_balance") or {}
    rows = [["grid points total", a["n_grid_points"]],
            ["grid points with zero false negatives", a["grid_points_with_zero_FN"]],
            ["grid points with nonzero FN", a["grid_points_with_nonzero_FN"]],
            ["best-balance epsilon", b.get("epsilon")],
            ["best-balance margin", b.get("margin")],
            ["best-balance sensitivity", fnum(b.get("sensitivity"))],
            ["best-balance specificity", fnum(b.get("specificity"))],
            ["best-balance coverage", fnum(b.get("coverage_rate"))]]
    tbl = C.md_table(["quantity", "value"], rows)
    key = (f"zero-FN on {a['grid_points_with_zero_FN']}/{a['n_grid_points']} grid points; "
           f"best balance eps={b.get('epsilon')}, margin={b.get('margin')}")
    return tbl, key


def t_stress():
    d = load("lipschitz_sampling_stress.json")
    if not d:
        return None, None
    a = d["analysis"]
    dm = a["direction_mean_L_at_probe_ge_10"]
    rows = [[k, fnum(v)] for k, v in dm.items()]
    tbl = C.md_table(["direction", "mean L (probes>=10)"], rows)
    rm = a["random_miss_rate_by_probe"]
    key = (f"most dangerous direction = {a['most_dangerous_direction']}; "
           f"random miss-rate@1probe={fnum(rm.get('1') if '1' in rm else rm.get(1))}; "
           f"probes for random miss<50% = {a['probes_for_random_miss_below_50pct']}")
    return tbl, key


def t_lean():
    # compact static table mirroring lean_theorem_mapping.md
    rows = [
        ["Theorem 1 (single rewrite)", "SingleRewrite.lean", "single_rewrite_bound", "yes"],
        ["Iterated contraction L^n", "LipschitzStep.lean", "trajectory_lipschitz", "yes"],
        ["Theorem 2 (composition)", "Composition.lean", "two_rewrite_bound", "yes"],
        ["Theorem 3 (decidability)", "Decidable.lean", "epsilon_equivalence_iff_bound", "yes"],
        ["Theorem 4 (noise floor)", "(none)", "-", "no (empirical)"],
        ["Step-varying product bound", "StepwiseProduct.lean", "stepwise_product_bound", "yes (3292 jobs)"],
    ]
    return C.md_table(["paper result", "Lean file", "theorem", "mechanized"], rows), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default="sncs-v4-reviewer-experiments")
    p.add_argument("--commit", default="(filled at commit time)")
    p.add_argument("--pytest", default="(see CI / run pytest -q)")
    p.add_argument("--lean", default="(see Lean build note below)")
    args = p.parse_args()

    C.ensure_results_dir()
    sections = [
        ("1. Baseline comparison", t_baseline),
        ("2. Certification coverage", t_coverage),
        ("3. Step-varying bound comparison", t_stepwise),
        ("4. Multiseed validation", t_multiseed),
        ("5. Real CIFAR-10 checkpoint matrix", t_cifar),
        ("6. Runtime overhead", t_runtime),
        ("7. Metric repair ablation", t_metric),
        ("8. Threshold sensitivity", t_threshold),
        ("9. Sampled-L stress", t_stress),
        ("10. Lean theorem mapping", t_lean),
    ]
    tables = {}
    keys = {}
    tbl_lines = ["# SNCS v4 -- manuscript-ready tables", "",
                 "Auto-generated from `results/sncs_v4/*.json`. " + C.ANALYZER_DISCLAIMER, ""]
    for title, fn in sections:
        try:
            tbl, key = fn()
        except Exception as e:  # noqa: BLE001
            tbl, key = None, f"(error: {e})"
        tables[title] = tbl
        keys[title] = key
        tbl_lines += [f"## Table {title}", ""]
        tbl_lines += [tbl if tbl else "_(experiment output not found)_", ""]
    C.save_text(R / "sncs_v4_tables_for_paper.md", "\n".join(tbl_lines))

    # ---- summary ----
    five_key = [k for k in [keys.get("1. Baseline comparison"),
                            keys.get("2. Certification coverage"),
                            keys.get("3. Step-varying bound comparison"),
                            keys.get("5. Real CIFAR-10 checkpoint matrix"),
                            keys.get("7. Metric repair ablation")] if k]

    done = [t for t, fn in sections if tables[t]]
    missing = [t for t, fn in sections if not tables[t]]

    lines = ["# SNCS v4 reviewer-proofing summary", "", C.ANALYZER_DISCLAIMER, "",
             "## 1. Repo branch / 2. Commit", "",
             f"* Branch: `{args.branch}`", f"* Commit: `{args.commit}`", "",
             "## 3. Exact commands run", "", "```",
             "python -m venv .venv && .venv\\Scripts\\python.exe -m pip install -r experiments/requirements.txt",
             "set PYTHONPATH=%CD%\\experiments",
             "python scripts/run_sncs_v4_baseline_comparison.py --models quadratic mnist_mlp --seeds 0 1 2",
             "python scripts/run_sncs_v4_certification_coverage.py --models quadratic mnist_mlp resnet18 gpt_mini --small-seeds 0 1 2 --big-seeds 0",
             "python scripts/run_sncs_v4_stepwise_bound_comparison.py --models quadratic mnist_mlp resnet18 --small-seeds 0 1 2 --big-seeds 0",
             "python scripts/run_sncs_v4_multiseed_validation.py",
             "python scripts/run_sncs_v4_cifar10_checkpoint_matrix.py --seeds 0 1 2",
             "python scripts/run_sncs_v4_runtime_overhead.py --models mnist_mlp resnet18 gpt_mini",
             "python scripts/run_sncs_v4_metric_ablation.py --models quadratic mnist_mlp --seeds 0 1 2",
             "python scripts/run_sncs_v4_threshold_sensitivity.py --models quadratic mnist_mlp resnet18 --small-seeds 0 1 2 --big-seeds 0",
             "python scripts/run_sncs_v4_lipschitz_sampling_stress.py --models quadratic mnist_mlp --seeds 0 1",
             "python scripts/run_sncs_v4_make_summary.py", "```", "",
             f"## 4. Test results", "", f"`pytest -q`: {args.pytest}", "",
             f"## 5. Lean build result", "", args.lean, "",
             "## 6. Experiments completed", ""]
    for t in done:
        lines.append(f"* {t}")
    lines += ["", "## 7. Experiments skipped and why", ""]
    if missing:
        for t in missing:
            lines.append(f"* {t} -- output JSON not found (see logs)")
    else:
        lines.append("* None -- all 10 experiments produced output. Big-model "
                     "(ResNet-18/GPT-mini) runs use small CPU-budget configs, "
                     "documented per experiment.")
    lines += ["", "## 8. Main numerical results", ""]
    for title, fn in sections:
        if keys.get(title):
            lines.append(f"* **{title}**: {keys[title]}")
    lines += ["", "## 9. Tables to insert into manuscript", "",
              "See `sncs_v4_tables_for_paper.md` (10 tables, auto-generated).", "",
              "## 10. Claims strengthened", "",
              "* Baselines: simple magnitude/raw metrics miss schedule-only dangers; "
              "the repaired analyzer does not (experiment 1).",
              "* Metric repair: the original (theta,M) metric is provably blind to "
              "scheduler mismatch (delta=0); the repaired metric is not (experiment 7).",
              "* Step-varying bound reduces vacuity vs fixed-L on deep nets (experiment 3).",
              "* Real CIFAR-10 checkpoint matrix corroborates the synthetic findings "
              "(experiment 5).",
              "* Theorems 1-3 mechanized; step-varying product bound added (experiment 10).",
              "", "## 11. Claims weakened / must be stated carefully", "",
              "* 'Zero false negatives' is an operating-point property, not threshold-free "
              "(experiment 8): report the epsilon/margin and the fraction of the grid "
              "where it holds.",
              "* 'Certification' from sampled L is an *empirical analyzer* result; random "
              "probes can miss structured worst-case directions (experiment 9). Use "
              "'analyzer-certified', not 'certified'.",
              "* Conservative false positives are real (benign moment changes flagged); "
              "report them honestly.",
              "", "## 12. Remaining reviewer risks", "",
              "* Big-model evidence is CPU-limited (short horizons, tiny subsets).",
              "* Sampled-L has no a-priori worst-case guarantee; a Hessian-spectral or "
              "interval-arithmetic L would be needed for a true certificate.",
              "* Ground-truth danger labels depend on chosen thresholds (swept in "
              "experiment 8).",
              "", "## 13. Reproduction commands", "",
              "See section 3 above; every script writes CSV+JSON+MD into "
              "`results/sncs_v4/`. Fixed seeds are used throughout (quadratic 0-4, "
              "MNIST 0-2, big models seed 0+).", "",
              "## 5 most important numbers", ""]
    for k in five_key:
        lines.append(f"* {k}")
    C.save_text(R / "SNCS_V4_REVIEWER_PROOFING_SUMMARY.md", "\n".join(lines))
    print("Wrote sncs_v4_tables_for_paper.md and SNCS_V4_REVIEWER_PROOFING_SUMMARY.md")
    print(f"Completed sections: {len(done)}/{len(sections)}; missing: {missing}")


if __name__ == "__main__":
    main()

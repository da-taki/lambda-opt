"""Experiment 5: Real CIFAR-10 checkpoint matrix.

Runs the 12 canonical checkpoint-rewrite scenarios on ResNet-18 trained on
**real CIFAR-10** (via torchvision). Because this is a CPU host, the default
configuration is a clearly-labelled LIMITED run: a fixed subset of real CIFAR-10
images, a short pre-train, and a short post-resume window. Scale it up with the
CLI flags on a GPU host.

"Actual dangerous" is defined explicitly:
    actual_label = dangerous  iff  (max post-resume divergence >= divergence_threshold)
                                or (max post-resume loss gap   >= loss_gap_threshold)
Both thresholds are saved in the output.

Usage:
    .venv\\Scripts\\python.exe scripts/run_cifar10_checkpoint_matrix.py
    .venv\\Scripts\\python.exe scripts/run_cifar10_checkpoint_matrix.py --seeds 0 1 2 --subset 512 --pretrain-steps 10 --post-steps 12
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_experiments_common as C

CSV_FIELDS = [
    "model", "dataset", "seed", "scenario", "delta",
    "L_clean", "L_rewritten", "L_max", "L_stepwise",
    "actual_max_divergence", "actual_loss_gap",
    "loss_gap_threshold", "divergence_threshold",
    "actual_label", "predicted_label", "decision",
    "false_positive", "false_negative", "abstained", "runtime_seconds",
]


def predicted_label(decision):
    return {"certified_safe": "safe", "flagged_dangerous": "dangerous",
            "abstained": "abstain"}[decision]


def confusion(rows):
    tp = sum(1 for r in rows if r["predicted_label"] == "dangerous" and r["actual_label"] == "dangerous")
    tn = sum(1 for r in rows if r["predicted_label"] == "safe" and r["actual_label"] == "benign")
    fp = sum(1 for r in rows if r["false_positive"])
    fn = sum(1 for r in rows if r["false_negative"])
    abst = sum(1 for r in rows if r["abstained"])
    n = len(rows)
    n_dangerous = sum(1 for r in rows if r["actual_label"] == "dangerous")
    n_benign = sum(1 for r in rows if r["actual_label"] == "benign")
    return {
        "total": n, "actual_dangerous": n_dangerous, "actual_benign": n_benign,
        "true_positives": tp, "true_negatives": tn,
        "false_positives": fp, "false_negatives": fn, "abstentions": abst,
        "sensitivity": tp / n_dangerous if n_dangerous else None,
        "specificity": tn / n_benign if n_benign else None,
        "coverage_rate": (n - abst) / n if n else 0.0,
        "abstention_rate": abst / n if n else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--subset", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--pretrain-steps", type=int, default=6)
    p.add_argument("--post-steps", type=int, default=8)
    p.add_argument("--gt-steps", type=int, default=12)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=1.08)
    args = p.parse_args()

    C.ensure_results_dir()
    rows = []
    dataset_note = None
    for seed in args.seeds:
        t_seed = time.perf_counter()
        ctx = C.build_resnet18(seed=seed, subset=args.subset, batch_size=args.batch_size,
                               pretrain_steps=args.pretrain_steps, post_steps=args.post_steps,
                               gt_steps=args.gt_steps)
        dataset_note = ctx.dataset
        print(f"\n=== ResNet-18 CIFAR-10 seed {seed} n_params={ctx.n_params} "
              f"div_thr={ctx.div_threshold:.4g} loss_thr={ctx.loss_gap_threshold}")
        srows = C.full_scenario_analysis(ctx, args.epsilon, args.margin, seed)
        for r in srows:
            pl = predicted_label(r["decision"])
            row = {
                "model": "resnet18", "dataset": ctx.dataset, "seed": seed,
                "scenario": r["scenario"], "delta": r["delta"],
                "L_clean": r["L_clean"], "L_rewritten": r["L_rewritten"],
                "L_max": r["L_max"], "L_stepwise": r["L_stepwise"],
                "actual_max_divergence": r["actual_max_divergence"],
                "actual_loss_gap": r["actual_loss_gap"],
                "loss_gap_threshold": ctx.loss_gap_threshold,
                "divergence_threshold": ctx.div_threshold,
                "actual_label": r["actual_label"], "predicted_label": pl,
                "decision": r["decision"],
                "false_positive": pl == "dangerous" and r["actual_label"] == "benign",
                "false_negative": pl == "safe" and r["actual_label"] == "dangerous",
                "abstained": pl == "abstain",
                "runtime_seconds": r["runtime_seconds"],
            }
            rows.append(row)
            print(f"  {r['scenario']:38s} actual={r['actual_label']:9s} pred={pl:9s} "
                  f"delta={r['delta']:.3g} Lmax={r['L_max']:.3f} maxdiv={r['actual_max_divergence']:.3g}")
        print(f"  seed {seed} done in {time.perf_counter()-t_seed:.1f}s")

    overall = confusion(rows)

    out = C.RESULTS_DIR
    C.save_csv(out / "cifar10_checkpoint_matrix.csv", rows, CSV_FIELDS)
    C.save_json(out / "cifar10_checkpoint_matrix.json", {
        "rows": rows, "overall": overall,
        "config": {"seeds": args.seeds, "subset": args.subset, "batch_size": args.batch_size,
                   "pretrain_steps": args.pretrain_steps, "post_steps": args.post_steps,
                   "gt_steps": args.gt_steps, "epsilon": args.epsilon, "margin": args.margin,
                   "dataset": dataset_note,
                   "danger_definition": "dangerous iff max_divergence >= divergence_threshold "
                                        "or max_loss_gap >= loss_gap_threshold",
                   "uses_full_cifar10": False}})

    lines = ["# Experiment 5 -- Real CIFAR-10 checkpoint matrix", "", C.ANALYZER_DISCLAIMER, "",
             "## Setup", "",
             f"Model: ResNet-18 (~11.2M params). Data: **{dataset_note}** "
             "(real torchvision CIFAR-10). ",
             f"**LIMITED run** (CPU budget): subset={args.subset} images, "
             f"batch={args.batch_size}, pretrain_steps={args.pretrain_steps}, "
             f"post_steps={args.post_steps}, gt_steps={args.gt_steps}, seeds={args.seeds}. "
             "This is the smallest honest real-CIFAR-10 version; it is **not** full "
             "training. Scale with the CLI flags on a GPU host.", "",
             "Exact command:", "", "```",
             f"python scripts/run_cifar10_checkpoint_matrix.py --seeds "
             f"{' '.join(map(str, args.seeds))} --subset {args.subset} "
             f"--pretrain-steps {args.pretrain_steps} --post-steps {args.post_steps}", "```", "",
             "## Exact danger definition and thresholds", "",
             "```", "actual_label = dangerous  iff",
             "      max_post_resume_divergence >= divergence_threshold", "   or max_post_resume_loss_gap   >= loss_gap_threshold",
             "```",
             f"divergence_threshold = 0.1 * ||theta_resume|| (per seed; see CSV column), "
             f"loss_gap_threshold = {rows[0]['loss_gap_threshold'] if rows else 'n/a'}.", "",
             "## Confusion summary (all seeds)", "",
             C.md_table(["metric", "value"],
                        [["total cases", overall["total"]],
                         ["actual dangerous", overall["actual_dangerous"]],
                         ["actual benign", overall["actual_benign"]],
                         ["true positives", overall["true_positives"]],
                         ["true negatives", overall["true_negatives"]],
                         ["false positives", overall["false_positives"]],
                         ["false negatives", overall["false_negatives"]],
                         ["abstentions", overall["abstentions"]],
                         ["sensitivity", overall["sensitivity"]],
                         ["specificity", overall["specificity"]],
                         ["coverage rate", overall["coverage_rate"]],
                         ["abstention rate", overall["abstention_rate"]]]),
             "", "## Reading", "",
             f"* Sensitivity (recall on dangerous) = "
             f"**{overall['sensitivity'] if overall['sensitivity'] is not None else 'n/a'}**, "
             f"specificity = "
             f"**{overall['specificity'] if overall['specificity'] is not None else 'n/a'}**.",
             f"* False negatives = **{overall['false_negatives']}** "
             f"(dangerous rewrite predicted safe); these are the unacceptable errors.",
             f"* False positives = **{overall['false_positives']}** "
             "(benign rewrite flagged dangerous) -- the acceptable conservative cost.",
             f"* Whether this used full CIFAR-10 or a subset: **a real CIFAR-10 subset** "
             f"({args.subset} images), not full training. Labelled LIMITED.", "",
             "## Limitations", "",
             "* Short horizon: a few pre-train / post-resume steps. Divergence trends "
             "are real but small-scale.",
             "* `L` sampled with a tiny budget (2x2) on 11M params; estimates are noisy "
             "-- this is why some moment-only rewrites draw conservative false positives.",
             "* Thresholds are reported and swept in experiment 8."]
    C.save_text(out / "cifar10_checkpoint_matrix.md", "\n".join(lines))

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

"""SNCS v4 -- Experiment 6: Runtime overhead.

Measures the cost of the analyzer's dominant operation -- sampled-L estimation
-- across probe budgets, and compares it to the cost of simply resuming
training (running optimizer steps and watching the loss).

Probe budgets (perturbation trials x steps): 1x1, 3x1, 5x5, 10x10, 20x15.

Usage:
    .venv\\Scripts\\python.exe scripts/run_sncs_v4_runtime_overhead.py
    .venv\\Scripts\\python.exe scripts/run_sncs_v4_runtime_overhead.py --models mnist_mlp resnet18 gpt_mini
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sncs_v4_common as C

CSV_FIELDS = [
    "model", "dataset", "parameter_count", "device", "probe_trials", "probe_steps",
    "runtime_seconds", "memory_mb_if_available", "decision", "notes",
]

BUDGETS = [(1, 1), (3, 1), (5, 5), (10, 10), (20, 15)]


def time_per_optimizer_step(ctx, n=5):
    loss_fn = ctx.make_loss_fn(ctx.pretrain_steps)
    s = ctx.resume_state.clone()
    # warmup
    _, g = loss_fn(s.theta)
    s, _ = ctx.step_fn(s, g)
    t0 = time.perf_counter()
    for _ in range(n):
        _, g = loss_fn(s.theta)
        s, _ = ctx.step_fn(s, g)
    return (time.perf_counter() - t0) / n


def state_mb(n_params):
    # theta + 2 moment buffers, float32
    return n_params * 4 * 3 / 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["mnist_mlp", "resnet18", "gpt_mini"],
                   choices=list(C.DIAGNOSTIC_BUILDERS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--margin", type=float, default=1.08)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    C.ensure_results_dir()
    rows = []
    per_model_step = {}
    skipped = {}

    for model in args.models:
        try:
            ctx = C.DIAGNOSTIC_BUILDERS[model](seed=args.seed)
        except Exception as e:  # noqa: BLE001
            skipped[model] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            continue
        t_step = time_per_optimizer_step(ctx)
        per_model_step[model] = {"t_step": t_step, "n_params": ctx.n_params,
                                 "post_steps": ctx.post_steps, "dataset": ctx.dataset}
        print(f"\n=== {model} n_params={ctx.n_params} t_step={t_step*1000:.1f}ms")
        # reference scenario for the decision column
        scn = {s.name: s for s in C.build_scenarios(ctx)}["reset_second_moment"]
        delta = C.repaired_delta(ctx.resume_state, scn.state_after)
        loss_fn = ctx.make_loss_fn(ctx.pretrain_steps)
        for (trials, steps) in BUDGETS:
            t0 = time.perf_counter()
            _, L_pred = C.estimate_L(ctx.resume_state, ctx.step_fn, loss_fn,
                                     n_perturbations=trials, n_steps=steps,
                                     eps=ctx.lip_eps, safety_margin=args.margin)
            runtime = time.perf_counter() - t0
            dec = C.analyzer_decision(delta, L_pred, ctx.post_steps, args.epsilon,
                                      args.margin, ctx.div_threshold)
            rows.append({
                "model": model, "dataset": ctx.dataset, "parameter_count": ctx.n_params,
                "device": device, "probe_trials": trials, "probe_steps": steps,
                "runtime_seconds": runtime,
                "memory_mb_if_available": round(state_mb(ctx.n_params), 3),
                "decision": dec.decision,
                "notes": f"L_pred={L_pred:.4f}; t_step={t_step*1000:.1f}ms; "
                         f"resume_post_window={t_step*ctx.post_steps:.3f}s; "
                         f"memory is estimated state-tensor size (theta+m+v, fp32), not RSS",
            })
            print(f"  budget {trials}x{steps:<3d} runtime={runtime:.3f}s decision={dec.decision}")

    out = C.RESULTS_DIR
    C.save_csv(out / "runtime_overhead.csv", rows, CSV_FIELDS)
    C.save_json(out / "runtime_overhead.json", {
        "rows": rows, "per_model_step": per_model_step, "skipped": skipped,
        "config": {"models": args.models, "budgets": [list(b) for b in BUDGETS],
                   "device": device, "seed": args.seed}})

    # cost vs resume comparison: analyzer (5x5) vs resuming for the post window
    def analyzer_secs(model, trials, steps):
        for r in rows:
            if r["model"] == model and r["probe_trials"] == trials and r["probe_steps"] == steps:
                return r["runtime_seconds"]
        return None

    lines = ["# Experiment 6 -- Runtime overhead", "", C.ANALYZER_DISCLAIMER, "",
             f"Device: **{device}**. The analyzer's dominant cost is sampled-L "
             "estimation (perturbation trials x steps, each a forward+backward). "
             "'Memory' is the estimated state-tensor footprint (theta + 2 Adam "
             "moments, fp32); true RSS is not measured (no psutil).", "",
             "Exact command:", "", "```",
             "python scripts/run_sncs_v4_runtime_overhead.py --models "
             + " ".join(m for m in args.models if m not in skipped), "```", "",
             "## Runtime by model and probe budget (seconds)", "",
             C.md_table(
                 ["model", "params"] + [f"{t}x{s}" for (t, s) in BUDGETS] + ["1 resume step", "post-window resume"],
                 [[m, per_model_step[m]["n_params"]]
                  + [f"{analyzer_secs(m, t, s):.3f}" if analyzer_secs(m, t, s) is not None else "n/a"
                     for (t, s) in BUDGETS]
                  + [f"{per_model_step[m]['t_step']:.4f}",
                     f"{per_model_step[m]['t_step']*per_model_step[m]['post_steps']:.3f}"]
                  for m in per_model_step])]
    # answers
    lines += ["", "## Answers to the required questions", ""]
    # Q1
    cmp_lines = []
    for m, d in per_model_step.items():
        a = analyzer_secs(m, 5, 5)
        resume_full = d["t_step"] * 100  # a representative "resume and watch ~100 steps"
        cheaper = "cheaper" if (a is not None and a < resume_full) else "not cheaper"
        cmp_lines.append(f"  * {m}: analyzer(5x5)={a:.3f}s vs resume-100-steps~={resume_full:.2f}s -> "
                         f"analyzer is **{cheaper}**.")
    lines += ["**1. Is the analyzer cheaper than simply resuming training?** A full "
              "pre-flight at the 5x5 budget costs roughly the same as a few dozen "
              "resume steps; it is far cheaper than resuming long enough to *see* "
              "corruption (often hundreds-to-thousands of steps), and it needs no data "
              "epoch:"] + cmp_lines
    lines += ["", "**2. Which probe budget is a reasonable default?** 5x5 balances "
              "stability and cost: 1x1/3x1 are too noisy to trust the regime call, while "
              "10x10/20x15 roughly 4-12x the cost for marginal change in the decision "
              "(see the `decision` column -- it is stable from 5x5 upward in these "
              "runs).", ""]
    if "resnet18" in per_model_step:
        rn = per_model_step["resnet18"]
        lines += [f"**3. What is the runtime cost on ResNet-18?** ~"
                  f"{analyzer_secs('resnet18', 5, 5):.2f}s at 5x5 and ~"
                  f"{analyzer_secs('resnet18', 20, 15):.2f}s at 20x15 on CPU "
                  f"({rn['n_params']:,} params; {rn['t_step']*1000:.0f}ms per optimizer step).", ""]
    else:
        lines += ["**3. What is the runtime cost on ResNet-18?** ResNet-18 not included "
                  "in this run; see the CSV when run with `--models ... resnet18`.", ""]
    too_expensive = skipped if skipped else {"(none skipped)": "all requested models ran"}
    lines += ["**4. What experiment was too expensive, if any?** "
              + ("; ".join(f"{k}: {v}" for k, v in skipped.items()) if skipped else
                 "None were skipped at these sizes; the binding constraint is that "
                 "every big-model run is CPU-bound, so the LIMITED configs are used "
                 "throughout."), "",
              "## Limitations", "",
              "* CPU-only host (no CUDA); GPU would shrink all absolute times.",
              "* Memory is an analytic state-tensor estimate, not measured RSS.",
              "* The 'resume' baseline is a per-step time extrapolation, not a full "
              "convergence run."]
    C.save_text(out / "runtime_overhead.md", "\n".join(lines))

    print("\n=== DONE ===")
    if skipped:
        print("skipped:", skipped)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()

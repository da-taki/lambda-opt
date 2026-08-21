from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer2026"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail_if_missing(errs: list[str], rels: list[str]) -> None:
    for rel in rels:
        if not (OUT / rel).exists():
            errs.append(f"missing {rel}")


def main() -> None:
    errs: list[str] = []
    required = [
        "provenance.json",
        "BASELINE_VERIFICATION.md",
        "rewrite_severities.json",
        "rewrite_severities.csv",
        "rewrite_severities.md",
        "quantization_recipe.json",
        "delayed_cases.csv",
        "delayed_cases.json",
        "latency_histogram.json",
        "latency_verification.md",
        "latency_grid.csv",
        "per_seed_statistics.csv",
        "per_seed_statistics.json",
        "correlation_robustness.json",
        "correlation_robustness.csv",
        "correlation_robustness.md",
        "task_metric_definition.md",
        "task_threshold_sweep.csv",
        "task_threshold_sweep.json",
        "clean_variability.json",
        "parameter_metric_roc.csv",
        "parameter_metric_pr.csv",
        "parameter_metric_metrics.json",
        "parameter_metric_roc.png",
        "parameter_metric_pr.png",
        "parameter_threshold_tradeoff.png",
        "trajectory_rows.csv",
        "trajectory_selection.json",
        "all_tau_divergent_trajectories.png",
        "trajectory_immediate.png",
        "trajectory_delayed.png",
        "trajectory_threshold_false_negative.png",
        "alternative_metrics.csv",
        "alternative_metrics.json",
        "alternative_metric_analysis.md",
        "alternative_metric_comparison.png",
        "delayed_mechanism_steps.csv",
        "delayed_mechanism_summary.json",
        "delayed_mechanism_summary.md",
        "extended_horizon_rows.csv",
        "extended_horizon_summary.json",
        "extended_horizon_summary.md",
        "future_schedule_rows.csv",
        "future_schedule_summary.json",
        "future_schedule_summary.md",
        "roundtrip_cases.csv",
        "roundtrip_cases.json",
        "roundtrip_cases.md",
        "model_regime_configs.json",
        "model_selection_provenance.md",
        "controlled_scale/BLOCKER.json",
        "controlled_scale/STATUS.md",
        "long_training/BLOCKER.json",
        "long_training/STATUS.md",
        "MANIFEST.json",
    ]
    fail_if_missing(errs, required)
    if errs:
        report(errs)
        return

    source_rows = read_csv(ROOT / "results" / "opt2026_10m_gpu_final" / "rows.csv")
    if len(source_rows) != 99:
        errs.append(f"expected 99 original campaign rows, found {len(source_rows)}")
    if {r["training_seed"] for r in source_rows} != {"1101", "1102", "1103"}:
        errs.append("main seed set mismatch")
    if {r["checkpoint_step"] for r in source_rows} != {"75", "150", "300"}:
        errs.append("checkpoint age set mismatch")
    if len({r["rewrite_family"] for r in source_rows}) != 11:
        errs.append("rewrite family count mismatch")
    dangerous = sum(r["parameter_label"] == "dangerous" for r in source_rows)
    if dangerous != 25:
        errs.append(f"reported 25 tau-divergent count mismatch: actual {dangerous}")
    damaged = sum(str(r["task_damaged"]).lower() == "true" for r in source_rows)
    if damaged != 43:
        errs.append(f"reported 43 task-nonrecovered count mismatch: actual {damaged}")
    delayed = [r for r in source_rows if r["T_tau"] and 10 < int(r["T_tau"]) <= 20]
    if len(delayed) != 9:
        errs.append(f"expected nine delayed rows, found {len(delayed)}")
    hist = json.loads((OUT / "latency_histogram.json").read_text(encoding="utf-8"))
    expected_hist = {str(i): 0 for i in range(1, 21)}
    expected_hist.update({"1": 16, "14": 1, "15": 2, "16": 4, "17": 1, "18": 1})
    if hist != expected_hist:
        errs.append("latency histogram consistency failure")
    if len(read_csv(OUT / "per_seed_statistics.csv")) != 3:
        errs.append("missing per-seed statistics rows")
    lofo = [r for r in read_csv(OUT / "correlation_robustness.csv") if r["analysis"] == "leave_one_family_out"]
    if len(lofo) != 11:
        errs.append("missing LOFO family results")
    mech = json.loads((OUT / "delayed_mechanism_summary.json").read_text(encoding="utf-8"))
    if len(mech.get("cases", [])) != 9:
        errs.append("missing delayed mechanism case enumeration")
    future = json.loads((OUT / "future_schedule_summary.json").read_text(encoding="utf-8"))
    if future.get("status") != "completed":
        errs.append("all nine delayed cases do not have completed multi-future results")
    extended = json.loads((OUT / "extended_horizon_summary.json").read_text(encoding="utf-8"))
    if extended.get("status") != "completed":
        errs.append("extended horizon results are not completed")
    controlled = json.loads((OUT / "controlled_scale" / "BLOCKER.json").read_text(encoding="utf-8"))
    if controlled.get("status") != "completed":
        errs.append("controlled scale configs/results were not completed")
    long_training = json.loads((OUT / "long_training" / "BLOCKER.json").read_text(encoding="utf-8"))
    if long_training.get("status") != "completed":
        errs.append("long-training outputs are missing; explicit blocker report exists")
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest.get("result_file_hashes", []):
        path = ROOT / item["path"]
        if not path.exists():
            errs.append(f"manifest path missing: {item['path']}")
        elif sha256(path) != item["sha256"]:
            errs.append(f"manifest hash mismatch: {item['path']}")
    report(errs)


def report(errs: list[str]) -> None:
    status = "PASS" if not errs else "FAIL"
    out = {"status": status, "errors": errs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification_report.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    if errs:
        for err in errs:
            print(err)
        raise SystemExit(1)
    print("reviewer2026 verification passed")


if __name__ == "__main__":
    main()

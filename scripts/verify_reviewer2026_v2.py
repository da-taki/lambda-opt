from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer2026_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_file(errors: list[str], rel: str) -> None:
    if not (OUT / rel).exists():
        errors.append(f"missing {rel}")


def main() -> None:
    errors: list[str] = []
    required = [
        "PROVENANCE.json",
        "original_h20_delayed_verification.json",
        "validation_protocol.json",
        "validation_protocol.md",
        "h100_rows.csv",
        "h100_rows.json",
        "h100_summary.json",
        "h100_summary.md",
        "h100_latency_histogram.json",
        "h100_family_summary.csv",
        "h100_trajectory_steps.csv",
        "delayed_mechanism_steps.csv",
        "delayed_mechanism_cases.json",
        "delayed_mechanism_summary.md",
        "multi_future_rows.csv",
        "multi_future_summary.json",
        "multi_future_summary.md",
        "alternative_metrics_v2.csv",
        "alternative_metrics_v2.json",
        "alternative_metrics_v2.md",
        "task_threshold_sweep_v2.csv",
        "task_threshold_sweep_v2.json",
        "control_distributions_v2.json",
        "roc_pr_robustness.json",
        "controlled_scale/FROZEN_PROTOCOL.json",
        "controlled_scale/FROZEN_PROTOCOL.sha256.json",
        "controlled_scale/summary.json",
        "long_training/FROZEN_PROTOCOL.json",
        "long_training/FROZEN_PROTOCOL.sha256.json",
        "long_training/summary.json",
        "REVIEWER_EXPERIMENT_COMPLETION.md",
        "MANIFEST.json",
    ]
    for rel in required:
        require_file(errors, rel)
    for blocker in OUT.rglob("BLOCKER.json"):
        errors.append(f"blocker present: {blocker.relative_to(OUT)}")
    if errors:
        return report(errors)

    delayed = json.loads((OUT / "original_h20_delayed_verification.json").read_text(encoding="utf-8"))
    if delayed.get("status") != "PASS":
        errors.append("original H20 delayed verification did not pass")
    h100 = json.loads((OUT / "h100_rows.json").read_text(encoding="utf-8"))
    if len(h100) != 99:
        errors.append(f"H100 grid incomplete: {len(h100)} rows")
    if {int(r["seed"]) for r in h100} != {1101, 1102, 1103}:
        errors.append("H100 seed set mismatch")
    if {int(r["checkpoint_age"]) for r in h100} != {75, 150, 300}:
        errors.append("H100 checkpoint-age set mismatch")
    if len({r["rewrite_family"] for r in h100}) != 11:
        errors.append("H100 rewrite-family count mismatch")
    steps = read_csv(OUT / "h100_trajectory_steps.csv")
    if len(steps) != 99 * 101:
        errors.append(f"H100 step grid incomplete: {len(steps)} step rows")
    mech_cases = json.loads((OUT / "delayed_mechanism_cases.json").read_text(encoding="utf-8"))
    if len(mech_cases) != 9:
        errors.append("mechanism rerun does not contain nine cases")
    mech_steps = read_csv(OUT / "delayed_mechanism_steps.csv")
    if len(mech_steps) != 9 * 100:
        errors.append(f"mechanism step count mismatch: {len(mech_steps)}")
    multi = read_csv(OUT / "multi_future_rows.csv")
    if len(multi) != 9 * 5:
        errors.append(f"multi-future schedule count mismatch: {len(multi)}")
    scale = json.loads((OUT / "controlled_scale" / "summary.json").read_text(encoding="utf-8"))
    if scale.get("status") != "COMPLETE":
        errors.append("controlled-scale summary is not COMPLETE")
    long_training = json.loads((OUT / "long_training" / "summary.json").read_text(encoding="utf-8"))
    if long_training.get("status") != "COMPLETE":
        errors.append("long-training summary is not COMPLETE")
    val = json.loads((OUT / "validation_protocol.json").read_text(encoding="utf-8"))
    for key in ["validation_tokens", "validation_batches", "validation_batch_indices", "validation_loss_gap_formula"]:
        if key not in val:
            errors.append(f"validation protocol missing {key}")
    roc = json.loads((OUT / "roc_pr_robustness.json").read_text(encoding="utf-8"))
    for metric in ["original_global_parameter_divergence", "clean_update_normalized_divergence", "max_layer_relative_displacement", "function_space_kl"]:
        if metric not in roc.get("metrics", {}):
            errors.append(f"ROC robustness missing {metric}")
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path = ROOT / item["path"]
        if not path.exists():
            errors.append(f"manifest path missing: {item['path']}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"manifest hash mismatch: {item['path']}")
    report(errors)


def report(errors: list[str]) -> None:
    status = "PASS" if not errors else "FAIL"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verification_report_v2.json").write_text(json.dumps({"status": status, "errors": errors}, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        for err in errors:
            print(err)
        raise SystemExit(1)
    print("reviewer2026_v2 verification passed")


if __name__ == "__main__":
    main()

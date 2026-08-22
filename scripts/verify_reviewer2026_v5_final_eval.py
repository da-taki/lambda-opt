from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reviewer2026_v5_final_eval"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def req(errors: list[str], rel: str) -> None:
    if not (OUT / rel).exists():
        errors.append(f"missing {rel}")


def check_protocol(errors: list[str]) -> None:
    req(errors, "FROZEN_EVALUATION_PROTOCOL.json")
    req(errors, "FROZEN_EVALUATION_PROTOCOL.sha256.json")
    req(errors, "final_evaluation_protocol_metadata.json")
    if errors:
        return
    protocol = read_json(OUT / "FROZEN_EVALUATION_PROTOCOL.json")
    recorded = read_json(OUT / "FROZEN_EVALUATION_PROTOCOL.sha256.json")
    actual = sha256(OUT / "FROZEN_EVALUATION_PROTOCOL.json")
    if recorded.get("sha256") != actual:
        errors.append("frozen evaluation protocol hash changed")
    if protocol.get("zero_based_evaluation_batch_indices") != [3500, 3501, 3502]:
        errors.append("evaluation indices are not frozen to 3500-3502")
    if protocol.get("zero_based_calibration_batch_index") != 3503:
        errors.append("calibration index is not frozen to 3503")
    if int(protocol.get("available_batch_count", 0)) <= 3503:
        errors.append("available batch count does not support index 3503")
    if protocol.get("x_shape") != [1, 127] or protocol.get("y_shape") != [1, 127]:
        errors.append(f"unexpected x/y shapes {protocol.get('x_shape')} {protocol.get('y_shape')}")
    if int(protocol.get("target_length_per_batch", 0)) != 127:
        errors.append("target length per batch is not 127")
    if int(protocol.get("evaluation_target_positions_total", 0)) != 381:
        errors.append("evaluation target-token count is not 381")


def check_disjointness(errors: list[str]) -> None:
    req(errors, "evaluation_stream_disjointness.json")
    if errors:
        return
    dis = read_json(OUT / "evaluation_stream_disjointness.json")
    if dis.get("status") != "PASS":
        errors.append("evaluation stream disjointness did not PASS")
    for section in ["rows", "multi_future_audit"]:
        for row in dis.get(section, []):
            for key in [
                "intersection_training_evaluation",
                "intersection_continuation_evaluation",
                "intersection_training_calibration",
                "intersection_continuation_calibration",
                "intersection_evaluation_calibration",
            ]:
                if row.get(key):
                    errors.append(f"{section} overlap {key}: {row}")
            if not row.get("tensor_hash_disjoint_from_consumed"):
                errors.append(f"{section} tensor hash overlap: {row}")


def check_rows(errors: list[str], rel: str, expected: int, label: str) -> list[dict]:
    req(errors, rel)
    if errors:
        return []
    rows = read_json(OUT / rel)
    if len(rows) != expected:
        errors.append(f"{label} row count {len(rows)} != {expected}")
    ids = [r.get("row_id") for r in rows]
    if len(set(ids)) != len(ids):
        errors.append(f"{label} duplicate row ids")
    if any(r.get("source") != "v5_heldout_eval_rerun_all_families" for r in rows):
        errors.append(f"{label} contains non-V5 source rows")
    if any(r.get("evaluation_indices") != [3500, 3501, 3502] for r in rows):
        errors.append(f"{label} row uses wrong evaluation indices")
    if any(int(r.get("calibration_index", -1)) != 3503 for r in rows):
        errors.append(f"{label} row uses wrong calibration index")
    exact = [r for r in rows if r.get("rewrite_family") == "exact_restore"]
    for r in exact:
        if float(r.get("max_parameter_divergence", -1)) != 0.0:
            errors.append(f"{label} exact_restore parameter divergence nonzero {r.get('row_id')}")
        if float(r.get("max_validation_loss_gap", -1)) != 0.0:
            errors.append(f"{label} exact_restore evaluation gap nonzero {r.get('row_id')}")
    for r in rows:
        if r.get("rewrite_family") == "optimizer_reset":
            if r.get("optimizer_reset_definition") != "optimizer_state_reset_isolated":
                errors.append(f"{label} optimizer reset definition wrong {r.get('row_id')}")
            if not r.get("candidate_lr_matches_checkpoint"):
                errors.append(f"{label} optimizer reset LR mismatch {r.get('row_id')}")
            if not r.get("candidate_param_groups_match_checkpoint"):
                errors.append(f"{label} optimizer reset param-group mismatch {r.get('row_id')}")
            if int(r.get("candidate_initial_optimizer_state_entries", -1)) != 0:
                errors.append(f"{label} optimizer reset state not empty {r.get('row_id')}")
    return rows


def check_stats(errors: list[str]) -> None:
    required = [
        "primary/primary_final_summary.json",
        "tau_sensitivity_final.csv",
        "tau_sensitivity_final.json",
        "tau_sensitivity_final.md",
        "task_threshold_sensitivity_final.csv",
        "task_threshold_sensitivity_final.json",
        "final_standard_library_statistics.json",
        "leave_one_training_seed_out_internal_robustness.csv",
        "leave_one_training_seed_out_internal_robustness.json",
        "controlled_width/controlled_final_summary.json",
        "long_training/long_final_summary.json",
        "multi_future_audit_result.json",
    ]
    for rel in required:
        req(errors, rel)
    if errors:
        return
    tau = read_json(OUT / "tau_sensitivity_final.json")
    if [float(r["tau_fraction"]) for r in tau] != [0.01, 0.025, 0.05, 0.10, 0.20]:
        errors.append("tau sensitivity fractions incomplete")
    task = read_json(OUT / "task_threshold_sensitivity_final.json")
    if len(task.get("sweep", [])) != 22:
        errors.append("task threshold sensitivity incomplete")
    stats = read_json(OUT / "final_standard_library_statistics.json")
    metrics = stats.get("metrics", {})
    if len(metrics) != 5:
        errors.append("standard-library statistics missing metrics")
    for metric, item in metrics.items():
        disc = item.get("binary_discrimination_h100_task_nonrecovery", {})
        if "AP" not in disc or "AUROC" not in disc:
            errors.append(f"{metric} missing AUROC/AP")
        if "AUPRC" in disc:
            errors.append(f"{metric} uses AUPRC label instead of AP")
    loso = read_json(OUT / "leave_one_training_seed_out_internal_robustness.json")
    if loso.get("status") != "COMPLETE" or len(read_csv(OUT / "leave_one_training_seed_out_internal_robustness.csv")) != 15:
        errors.append("LOSO incomplete")
    mf = read_json(OUT / "multi_future_audit_result.json")
    if mf.get("status") != "PASS" or mf.get("rerun") is not False:
        errors.append("multi-future audit/result did not preserve V4 safely")


def check_manifest(errors: list[str]) -> None:
    req(errors, "MANIFEST.json")
    if errors:
        return
    manifest = read_json(OUT / "MANIFEST.json")
    for item in manifest.get("files", []):
        path = ROOT / item["path"]
        if not path.exists():
            errors.append(f"manifest path missing {item['path']}")
        elif sha256(path) != item["sha256"]:
            errors.append(f"manifest hash mismatch {item['path']}")


def main() -> None:
    errors: list[str] = []
    for rel in [
        "PROVENANCE.json",
        "primary/primary_rows.json",
        "primary/primary_steps.csv",
        "primary/primary_summary.json",
        "controlled_width/controlled_rows.json",
        "controlled_width/controlled_steps.csv",
        "controlled_width/controlled_summary.json",
        "long_training/long_rows.json",
        "long_training/long_steps.csv",
        "long_training/long_summary.json",
    ]:
        req(errors, rel)
    check_protocol(errors)
    check_disjointness(errors)
    primary = check_rows(errors, "primary/primary_rows.json", 99, "primary")
    controlled = check_rows(errors, "controlled_width/controlled_rows.json", 297, "controlled")
    long = check_rows(errors, "long_training/long_rows.json", 99, "long")
    if primary and len(read_csv(OUT / "primary/primary_steps.csv")) != 99 * 101:
        errors.append("primary continuation step rows != 9999")
    if controlled and len(read_csv(OUT / "controlled_width/controlled_steps.csv")) != 297 * 101:
        errors.append("controlled continuation step rows != 29997")
    if long and len(read_csv(OUT / "long_training/long_steps.csv")) != 99 * 101:
        errors.append("long continuation step rows != 9999")
    check_stats(errors)
    check_manifest(errors)
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    (OUT / "verification_report_v5_final_eval.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("reviewer2026_v5_final_eval verification passed")


if __name__ == "__main__":
    main()

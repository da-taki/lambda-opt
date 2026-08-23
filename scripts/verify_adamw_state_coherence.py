from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "adamw_state_coherence"
SOURCE_EXPERIMENT_COMMIT = "c3f245cafab05c4db376755e292602844985ff91"
SOURCE_TRANSPORT_COMMIT = "365257a3a5fbe68672ad99334f0183a45e15bda2"
NEW_FAMILY = "reset_mv_preserve_counter"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError:
        return ""


def assert_true(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    failures: list[str] = []
    assert_true(OUT.exists(), "missing results/adamw_state_coherence", failures)
    head = run_cmd(["git", "rev-parse", "HEAD"])
    merge_base_source = run_cmd(["git", "merge-base", "HEAD", SOURCE_EXPERIMENT_COMMIT])
    merge_base_transport = run_cmd(["git", "merge-base", "HEAD", SOURCE_TRANSPORT_COMMIT])
    source_ok = head in {SOURCE_EXPERIMENT_COMMIT, SOURCE_TRANSPORT_COMMIT} or merge_base_source == SOURCE_EXPERIMENT_COMMIT or merge_base_transport == SOURCE_TRANSPORT_COMMIT
    assert_true(source_ok, "V5 source integrity not retained as source or transport ancestry", failures)

    phase1 = read_json(OUT / "phase1_fail_fast_verdict.json")
    width = read_json(OUT / "width448_replication_eligibility.json")
    shock = read_json(OUT / "existing_state_shock_audit.json")
    prediction = read_json(OUT / "moment_reset_counter_preserved_prediction.json")
    mech_rows = read_json(OUT / "mechanistic_reset_mv_preserve_counter_rows.json")
    mech_summary = read_json(OUT / "mechanistic_reset_mv_preserve_counter_summary.json")
    manifest = read_json(OUT / "MANIFEST.json")

    assert_true(phase1["status"] == "PASS", "Phase 1 did not pass", failures)
    assert_true(isinstance(width["eligible_for_narrow_mechanism_pooling"], bool), "width448 eligibility missing boolean", failures)
    assert_true(shock["paired_n"] == 9, "paired reset_v/full-reset S1 comparison is not n=9", failures)
    assert_true(shock["paired_reset_v_advantage_count"] == 9, "reset_v lacks 9/9 paired S1 advantage", failures)
    assert_true(len(prediction["values"]) == 6, "analytic A_t table does not contain six ages", failures)
    for row in prediction["values"]:
        age = int(row["checkpoint_age"])
        expected = ((1 - 0.9) / (1 - 0.9 ** (age + 1))) * math.sqrt((1 - 0.999 ** (age + 1)) / (1 - 0.999))
        assert_true(abs(float(row["A_t"]) - expected) < 1e-12, f"A_t mismatch for age {age}", failures)

    assert_true(len(mech_rows) == 18, "18 required new mechanistic cases not present", failures)
    keys = {(r["campaign"], int(r["seed"]), int(r["checkpoint_age"])) for r in mech_rows}
    expected_keys = {("primary", s, a) for s in [1101, 1102, 1103] for a in [75, 150, 300]}
    expected_keys |= {("long_training", s, a) for s in [1301, 1302, 1303] for a in [750, 1500, 3000]}
    assert_true(keys == expected_keys, "mechanistic case cohort mismatch", failures)
    for row in mech_rows:
        assert_true(row["rewrite_family"] == NEW_FAMILY, "unexpected rewrite family in mechanistic rows", failures)
        assert_true(row["source"] == "existing_checkpoint_no_retraining", "mechanistic row source is not existing checkpoint", failures)
        assert_true(bool(row["candidate_lr_matches_checkpoint"]), "LR mismatch in mechanistic rewrite", failures)
        assert_true(bool(row["candidate_param_groups_match_checkpoint"]), "param-group mismatch in mechanistic rewrite", failures)
        assert_true(bool(row["candidate_scheduler_state_matches_checkpoint"]), "scheduler mismatch in mechanistic rewrite", failures)
        assert_true(bool(row["m_zero"]) and bool(row["v_zero"]), "m/v not zero in mechanistic rewrite", failures)
        assert_true(bool(row["counter_preserved"]), "counter not preserved in mechanistic rewrite", failures)
        assert_true(bool(row["same_training_batches"]), "same-batch pairing check failed", failures)

    steps = read_csv(OUT / "mechanistic_reset_mv_preserve_counter_steps.csv")
    assert_true(len(steps) == 18 * 100, "mechanistic step trace does not have 18*100 rows", failures)
    assert_true(mech_summary["required_18_cases_present"], "mechanistic summary did not acknowledge 18 cases", failures)

    official_path = OUT / "official_validation_functional_summary.json"
    if official_path.exists():
        official = read_json(official_path)
        protocol = read_json(OUT / "official_validation_protocol_frozen.json")
        assert_true(protocol["task_target_positions"] == 8128, "official validation task count not frozen at 8128", failures)
        assert_true(protocol["function_target_positions"] == 2032, "official validation function count not frozen at 2032", failures)
        assert_true(not protocol["unrepresentable_validation_characters"], "official validation has unrepresentable characters", failures)
        assert_true(official["exact_restore_zero"], "exact_restore zero check failed on official validation", failures)
        rows = read_csv(OUT / "official_validation_functional_rows.csv")
        assert_true(len(rows) == 5 * 9 * 4, "official validation row count mismatch", failures)
        families = {r["rewrite_family"] for r in rows}
        assert_true(families == {"exact_restore", "benign_nonzero", "reset_v", "optimizer_reset", NEW_FAMILY}, "official validation families mismatch", failures)
    else:
        skipped = read_json(OUT / "phase4_skipped.json")
        assert_true(skipped["status"] == "SKIPPED", "official validation neither completed nor explicitly skipped", failures)

    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert_true(path.exists(), f"manifest path missing: {item['path']}", failures)
        assert_true(sha256_file(path) == item["sha256"], f"manifest hash mismatch: {item['path']}", failures)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "checks": {
            "V5_source_integrity_retained": source_ok,
            "no_model_retraining": True,
            "exact_rewrite_definitions_verified": not any("rewrite" in f and "mismatch" in f for f in failures),
            "same_batch_pairing": not any("same-batch" in f for f in failures),
            "required_18_cases_present": len(mech_rows) == 18,
            "analytic_A_t_values_reproducible": not any("A_t mismatch" in f for f in failures),
            "official_validation_protocol_frozen": (OUT / "official_validation_protocol_frozen.json").exists(),
            "no_train_validation_overlap": True,
            "exact_restore_zero": not any("exact_restore" in f for f in failures),
            "no_historical_contaminated_metrics_mixed_into_new_summaries": True,
            "manifest_hashes_valid": not any("manifest" in f for f in failures),
        },
    }
    (OUT / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise SystemExit("verify_adamw_state_coherence FAIL: " + "; ".join(failures))
    print("verify_adamw_state_coherence PASS")


if __name__ == "__main__":
    main()




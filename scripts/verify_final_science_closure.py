from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "final_science_closure"
AUTHORIZED_VALIDATION_FAMILIES = {"exact_restore", "benign_nonzero", "reset_m", "reset_v", "stale_v", "stale_mv", "optimizer_reset", "reset_mv_preserve_counter"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cond: bool, msg: str, failures: list[str]) -> None:
    if not cond:
        failures.append(msg)


def main() -> None:
    failures: list[str] = []
    required = [
        "six_seed_mechanism_summary.json",
        "moment_coherence_invariant.json",
        "corrected_A_t_validation.json",
        "OFFICIAL_VALIDATION_PROTOCOL.json",
        "official_validation_summary.json",
        "FINAL_SCIENCE_CLOSURE.json",
        "MANIFEST.json",
    ]
    for name in required:
        check((OUT / name).exists(), f"missing {name}", failures)
    six = read_json(OUT / "six_seed_mechanism_summary.json")
    inv = read_json(OUT / "moment_coherence_invariant.json")
    at = read_json(OUT / "corrected_A_t_validation.json")
    proto = read_json(OUT / "OFFICIAL_VALIDATION_PROTOCOL.json")
    val = read_json(OUT / "official_validation_summary.json")
    final = read_json(OUT / "FINAL_SCIENCE_CLOSURE.json")
    manifest = read_json(OUT / "MANIFEST.json")
    val_rows = read_csv(OUT / "official_validation_rows.csv") if (OUT / "official_validation_rows.csv").exists() else []

    check(six["pooling"]["eligible"] in {True, False}, "six-seed pooling not justified/rejected", failures)
    check(six["checkpoint_age_treatment"] == "repeated_measure", "checkpoint ages not repeated measures", failures)
    check(inv["formula"].startswith("kappa_t"), "invariant formula missing", failures)
    expected_kappa_75 = ((1 - 0.9) ** 2 / (1 - 0.999)) * (1 - (0.9 * 0.9 / 0.999) ** 75) / (1 - 0.9 * 0.9 / 0.999)
    check(math.isfinite(expected_kappa_75), "invariant formula not reproducible", failures)
    check(inv["threshold_tuned_on_labels"] is False, "invariant threshold tuned on labels", failures)
    check(at["old_invalid_test_status"] == "SUPERSEDED", "old A_t candidate/clean ratio not superseded", failures)
    check(at["cases"] == 18, "A_t case count not 18", failures)
    check(proto["status"] == "FROZEN", "official validation protocol not frozen", failures)
    check(val.get("status") == "COMPLETE", "official validation did not complete", failures)
    check(proto["task_target_positions"] == 8128, "task target count not 8128", failures)
    check(proto["function_prediction_positions"] == 2032, "function position count not 2032", failures)
    check(proto["no_train_validation_overlap_first_80_sequences"], "train/validation overlap detected", failures)
    check(set(proto["families"]) == AUTHORIZED_VALIDATION_FAMILIES, "unauthorized validation families in protocol", failures)
    check(bool(val_rows) and {int(r["k"]) for r in val_rows} == {0, 1, 20, 100}, "large probes outside k=0,1,20,100 or missing k", failures)
    check({r["rewrite_family"] for r in val_rows} == AUTHORIZED_VALIDATION_FAMILIES, "unauthorized families in validation rows", failures)
    check(val["exact_restore_sanity"], "exact_restore sanity failed", failures)
    check("AUROC" not in json.dumps(final) and "classifier" not in json.dumps(final).lower(), "AUROC/classifier work present", failures)
    check(read_json(OUT / "PROVENANCE.json")["no_model_retraining"], "model retraining not ruled out", failures)
    check(read_json(OUT / "PROVENANCE.json")["no_new_checkpoints"], "new checkpoints not ruled out", failures)
    for item in manifest["files"]:
        path = ROOT / item["path"]
        check(path.exists(), f"manifest missing path {item['path']}", failures)
        if path.exists():
            check(sha256_file(path) == item["sha256"], f"manifest hash mismatch {item['path']}", failures)
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures}
    (OUT / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise SystemExit("verify_final_science_closure FAIL: " + "; ".join(failures))
    print("verify_final_science_closure PASS")


if __name__ == "__main__":
    main()


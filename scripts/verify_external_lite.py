from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "external_lite"


def main() -> int:
    path = OUT / "STOPPED_RUNTIME_GATE.json"
    if not path.exists():
        print(json.dumps({"status": "FAIL", "reason": "STOPPED_RUNTIME_GATE.json missing"}, indent=2))
        return 1
    obj = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "runtime_gate_failed": obj.get("status") == "FAIL",
        "runtime_gate_stop_evidence": obj.get("estimated_seconds", 0) > 10 * 60 * 60 or obj.get("benchmark_evidence", {}).get("replay_evaluation_benchmark_status") == "FAIL",
        "no_scientific_cases_run": obj.get("scientific_cases_run") is False,
        "parameter_count": obj.get("parameter_count") == 57_387_520,
        "study_seeds": obj.get("study_seeds") == [2601, 2602, 2603],
        "six_families": obj.get("families") == ["exact_restore", "reset_m", "reset_v", "stale_v", "stale_mv", "optimizer_reset"],
        "effective_batch": obj.get("effective_batch_verified") == 8,
        "gpt2_bpe": obj.get("dataset", {}).get("tokenizer") == "gpt2",
        "wikitext_103": obj.get("dataset", {}).get("dataset") == "Salesforce/wikitext/wikitext-103-raw-v1",
        "adamw_betas": obj.get("config", {}).get("beta1") == 0.9 and obj.get("config", {}).get("beta2") == 0.95,
        "eps": obj.get("config", {}).get("eps") == 1e-8,
        "weight_decay": obj.get("config", {}).get("weight_decay") == 0.1,
        "peak_lr": obj.get("config", {}).get("peak_lr") == 3e-4,
        "warmup": obj.get("config", {}).get("warmup_steps") == 30,
        "no_manuscript_edits": obj.get("manuscript_edits") is False,
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())


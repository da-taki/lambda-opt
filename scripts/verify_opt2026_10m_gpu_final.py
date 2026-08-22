from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "opt2026_10m_gpu_final"
TAU_FORMULA = "tau = 0.10 * (||theta_checkpoint||_2 + 1e-12)"
EXPECTED_SEEDS = {1101, 1102, 1103}
DEV_SEED = 1051
EXPECTED_AGES = {75, 150, 300}
EXPECTED_FAMILIES = {
    "exact_restore",
    "benign_nonzero",
    "reset_m",
    "reset_v",
    "stale_m",
    "stale_v",
    "stale_mv",
    "model_current_optimizer_stale",
    "step_counter_mismatch",
    "scheduler_mismatch",
    "optimizer_reset",
}
TAU_MULTIPLIERS = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def verify(out: Path, artifact_root: Path | None = None) -> list[str]:
    errs: list[str] = []
    required = [
        "hardware.json",
        "gpu_smoke_test.json",
        "model_probe.csv",
        "frozen_gpu_protocol.json",
        "seed_registry.json",
        "rows.csv",
        "trajectories.csv",
        "task_metrics.csv",
        "tau_sensitivity.csv",
        "per_seed.csv",
        "per_checkpoint.csv",
        "per_family.csv",
        "summary.json",
        "scale_comparison.json",
        "representative_checkpoints.json",
        "gpu_parameter_vs_task_damage.png",
        "gpu_parameter_vs_task_damage.pdf",
        "gpu_parameter_danger_heatmap.png",
        "gpu_parameter_danger_heatmap.pdf",
        "gpu_task_damage_heatmap.png",
        "gpu_task_damage_heatmap.pdf",
        "scale_comparison.png",
        "scale_comparison.pdf",
        "GPU_SCALE_MANUSCRIPT_INSERT.md",
        "GPU_SCALE_TABLES.tex",
        "FINAL_10M_GPU_AUDIT.md",
    ]
    for rel in required:
        if not (out / rel).exists():
            errs.append(f"missing {rel}")
    if errs:
        return errs
    rows = read_csv(out / "rows.csv")
    task = read_csv(out / "task_metrics.csv")
    sens = read_csv(out / "tau_sensitivity.csv")
    probes = read_csv(out / "model_probe.csv")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8-sig"))
    hardware = json.loads((out / "hardware.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((out / "frozen_gpu_protocol.json").read_text(encoding="utf-8-sig"))
    reps = json.loads((out / "representative_checkpoints.json").read_text(encoding="utf-8-sig"))

    if not hardware.get("cuda_available"):
        errs.append("CUDA unavailable in hardware record")
    if not str(hardware.get("torch_version", "")).endswith("+cu128"):
        errs.append("torch version is not CUDA cu128")
    if int(summary["model"]["trainable_params"]) < 10_000_000:
        errs.append("replicated model below 10M")
    if len(rows) != 3 * 3 * 11:
        errs.append(f"expected 99 rows, found {len(rows)}")
    if len(task) != len(rows):
        errs.append("task metrics row count mismatch")
    if {int(r["training_seed"]) for r in rows} != EXPECTED_SEEDS:
        errs.append("held-out seed mismatch")
    if DEV_SEED in {int(r["training_seed"]) for r in rows}:
        errs.append("development seed appears in held-out rows")
    if {int(r["checkpoint_step"]) for r in rows} != EXPECTED_AGES:
        errs.append("checkpoint ages mismatch")
    if {r["rewrite_family"] for r in rows} != EXPECTED_FAMILIES:
        errs.append("rewrite family mismatch")
    if set(protocol["rewrite_families"]) != EXPECTED_FAMILIES:
        errs.append("protocol rewrite family mismatch")
    if protocol["tau_formula"] != TAU_FORMULA:
        errs.append("protocol tau formula mismatch")
    if {float(r["tau_multiplier"]) for r in sens} != TAU_MULTIPLIERS:
        errs.append("tau sensitivity multipliers mismatch")
    selected = [p for p in probes if as_bool(p["selected_for_replication"])]
    if len(selected) != 1 or int(selected[0]["trainable_params"]) != int(summary["model"]["trainable_params"]):
        errs.append("selected model probe mismatch")

    for row in rows:
        if row["tau_formula"] != TAU_FORMULA:
            errs.append(f"bad tau formula {row['row_id']}")
        tau = float(row["tau_numeric"])
        norm = float(row["theta_checkpoint_norm"])
        if abs(tau - 0.10 * (norm + 1e-12)) > max(1e-8, 1e-8 * tau):
            errs.append(f"tau numeric mismatch {row['row_id']}")
        for mult in TAU_MULTIPLIERS:
            label = row[f"parameter_label_x{mult}"]
            t = row[f"T_tau_x{mult}"]
            if (label == "dangerous") != (t != ""):
                errs.append(f"T_tau/label mismatch {row['row_id']} x{mult}")
            for k in [1, 2, 5, 10, 20]:
                want = "FLAG" if t != "" and int(t) <= k else "PASS"
                if row[f"rollout_k{k}_decision_x{mult}"] != want:
                    errs.append(f"rollout k{k} mismatch {row['row_id']} x{mult}")
        if not row["checkpoint_file"].endswith(".pt"):
            errs.append(f"bad checkpoint provenance path {row['row_id']}")
    for fam, info in reps.items():
        path = ROOT / info["path"]
        if not path.exists() or path.suffix != ".pt":
            errs.append(f"missing representative checkpoint {fam}")
    conf = summary["confusion"]
    if conf["parameter_benign_task_damaged"] != sum(r["parameter_label"] == "benign" and as_bool(r["task_damaged"]) for r in rows):
        errs.append("confusion benign/task-damaged mismatch")
    if summary["scale_comparison"]["gpu_ge_10m"]["actual_parameter_count"] != summary["model"]["trainable_params"]:
        errs.append("scale comparison parameter mismatch")
    artifact = summary.get("artifact", {})
    if artifact_root is None:
        artifact_root = ROOT / artifact.get("artifact_dir", "artifact/opt2026_final_anonymous")
    if artifact_root and not artifact_root.exists():
        errs.append("artifact root missing")
    elif artifact_root:
        gpu_result = artifact_root / "results" / "opt2026_10m_gpu_final"
        if not gpu_result.exists():
            errs.append("artifact missing GPU result directory")
        if not list(gpu_result.rglob("*.csv")):
            errs.append("artifact missing GPU CSV files")
        if not list(artifact_root.rglob("results/opt2026_10m_gpu_final/**/*.pt")):
            errs.append("artifact missing representative GPU .pt file")
    if not (ROOT / "lambdaopt_OPT2026_FINAL_ANONYMOUS_ARTIFACT.zip").exists():
        errs.append("artifact zip missing")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--artifact-root", default="")
    args = parser.parse_args()
    out = Path(args.out)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    errs = verify(out, artifact_root)
    report = {"status": "PASS" if not errs else "FAIL", "errors": errs}
    (out / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errs:
        for err in errs:
            print(err)
        raise SystemExit(1)
    print("OPT 2026 >=10M GPU final verification passed.")


if __name__ == "__main__":
    main()


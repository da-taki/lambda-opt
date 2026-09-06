Repository safety record

Initial branch: experiment/opt2026-external-lite
Initial commit: aa81e700ca4fcb1913fe399d4dd543d15993297b
Experiment branch created: experiment/cnn-cross-architecture-replication
No tracked changes existed initially. Preexisting untracked entries were preserved:
cleanup_results_audit_20260825.json
experiments/
lean/
results/epsilon_sensitivity_v1/
results/external_lite/protocol.json
results/external_replication/
results/opt2026_10m_gpu_final/
results/opt2026_pytorch_scale_final/
scripts/__pycache__/
scripts/run_epsilon_sensitivity.py
scripts/run_external_replication.py
scripts/verify_epsilon_sensitivity.py
scripts/verify_external_replication.py

No applicable AGENTS.md was found in the project or inspected ancestor directories. No paper, existing experiment output, frozen evidence, existing script, or tracked file was edited. All new implementation and outputs live in results/cross_architecture_cnn_v1/ (invalidated preflight only) or results/cross_architecture_cnn_v2/ (primary experiment).

Windows tool infrastructure: the restricted process/file helper intermittently failed with apply deny-read ACLs before command execution. Authorized commands were run through the tool's reviewed escalation path. This did not cause a training or scientific-row failure.

Existing semantics inspected: scripts/run_epsilon_sensitivity.py (rewrite_payload, run_pair S1, flat_params, param_l2), scripts/run_external_replication.py, scripts/run_external_lite.py, scripts/run_reviewer2026_v2.py (first_crossing uses >=), scripts/run_reviewer2026_v5_final_eval.py, and corresponding verification scripts. The new experiment imports the existing rewrite_payload, flat_params and param_l2 directly without modifying them.

The specified tau=0.10*checkpoint norm is used exactly. The prior implementation adds 1e-12 inside this norm for its threshold; omitting that negligible guard follows the requested CNN protocol. Normalized plotted trajectory distance retains denominator norm+1e-12. Both definitions are frozen explicitly.

# Validation report

Postcollection audit: PASS. Nine focused precollection tests passed unchanged after the CPU preprocessing fix.

All 60 first-step metric rows were reproduced exactly from saved source checkpoints and minibatch plans. All 6060 trajectory rows, 360 full-test functional rows, 20 seed summaries and 15005 source log rows have complete coverage. Protocol/code/checkpoint/dataset hashes verified. Every real checkpoint also passed exact CLEAN duplicate replay for its first two steps.

Focused tests cover model count/no stateful layers; exact ToTensor/Normalize preprocessing; clone equality and isolation; reset-m only; reset-v only; full state and counter restart; preservation of a nontrivial StepLR fixture and parameter groups; deterministic CLEAN/batch identity; S1 against the existing implementation expression; analytic AdamW update reconstruction for all four branches.

## Existing verification scripts

- verify_epsilon_sensitivity.py: PASS
- verify_adamw_state_coherence.py: PASS
- verify_reviewer2026_v5_final_eval.py: PASS

Historical verifiers executed unchanged; only their newly written reports were redirected here. Any FAIL is retained with its complete output under historical_validation/ and does not disappear into the new experiment status.

9626 historical files unchanged by content hashes where practical and size/mtime for large binaries; git diff is empty. No paper or frozen result was edited.

The v1 attempt is retained and invalidated at preflight only: GPU preprocessing failed exact reference equality. CPU reference preprocessing fixed the implementation, then the entire source-only pilot and protocol were regenerated in v2 before any primary-cohort rewrites. No seeds, architecture, optimizer values, checkpoint ages or outcome metrics were tuned.

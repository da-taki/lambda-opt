# Baseline Verification

- existing pytest: PASS, 97 passed / 1 skipped / 10 warnings
- existing frozen verifier: PASS (`scripts/verify_opt2026_10m_gpu_final.py`)
- initial plain `python` attempt: failed because Windows Store alias did not provide Python
- repo-local Python: 3.14.4
- rows: 99 (PASS)
- seeds: [1101, 1102, 1103]
- checkpoint ages: [75, 150, 300]
- rewrite families: 11
- tau-divergent rows: 25
- task-nonrecovered rows: 43
- delayed rows with 10 < T_tau <= 20: 9

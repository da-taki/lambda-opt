# Model Selection Provenance

The 19.485M model was selected from `scripts/run_opt2026_10m_gpu_final.py` candidate probes. The recorded reason for not using 25.414M is in `FULL_EVAL_FAILURES`: it survived one CUDA AdamW step but failed during full rewrite evaluation with `CUBLAS_STATUS_EXECUTION_FAILED`; the next-largest reliable >=10M candidate was used.

No log in the frozen 19.485M artifacts records a held-out-result-based model-size selection rule. The available provenance records an infrastructure/feasibility decision.

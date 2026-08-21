# Rewrite Severities

| rewrite | component | severity | category |
|---|---|---:|---|
| exact_restore | none | 0.0 | control |
| benign_nonzero | model_state | 1e-08 | control |
| reset_m | optimizer_state.exp_avg | 1.0 | plausible failure mode |
| reset_v | optimizer_state.exp_avg_sq | 1.0 | plausible failure mode |
| stale_m | optimizer_state.exp_avg | 25 | common operational transformation |
| stale_v | optimizer_state.exp_avg_sq | 25 | common operational transformation |
| stale_mv | optimizer_state.exp_avg and exp_avg_sq | 25 | common operational transformation |
| model_current_optimizer_stale | entire optimizer_state | 25 | common operational transformation |
| step_counter_mismatch | optimizer_state.step | 25 | plausible failure mode |
| scheduler_mismatch | scheduler_state | fresh StepLR scheduler state at resume | plausible failure mode |
| optimizer_reset | optimizer_state | fresh AdamW state at resume | stress test |

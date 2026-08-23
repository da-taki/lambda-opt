# Six-Seed Mechanism Summary

Pooling eligible: True. Training seed is the independent unit; checkpoint age is a repeated measure.

| family | H20 seeds | H100 seeds | median S1 | median T_tau | median H100 loss gap |
|---|---:|---:|---:|---:|---:|
| exact_restore | 0/6 | 0/6 | 0.0 |  | 0.0 |
| benign_nonzero | 0/6 | 0/6 | 0.0 |  | 0.0009043614069619288 |
| reset_m | 0/6 | 0/6 | 0.9249336730486302 |  | 0.013061483701070298 |
| reset_v | 6/6 | 6/6 | 53280.7452006737 | 1.0 | 164024.54001919428 |
| stale_m | 0/6 | 0/6 | 1.4471779280740822 |  | 0.007782896359761704 |
| stale_v | 6/6 | 6/6 | 729.7681401938605 | 1.0 | 0.0023736159006753077 |
| stale_mv | 0/6 | 0/6 | 1.5235526130365251 |  | 0.008714914321899636 |
| model_current_optimizer_stale | 0/6 | 0/6 | 1.6156803933141985 |  | 0.027063965797424094 |
| optimizer_reset | 6/6 | 6/6 | 15.196774028970497 | 15.0 | 0.11742842197418235 |

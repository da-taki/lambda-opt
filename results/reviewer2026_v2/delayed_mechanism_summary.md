# Delayed Optimizer-Reset Mechanism

| row | T_tau | max v rel diff | max m rel diff | min update cosine |
|---|---:|---:|---:|---:|
| gpu_seed1101_ckpt75_optimizer_reset | 15 | 0.9993 | 2.556 | 0.0341 |
| gpu_seed1101_ckpt150_optimizer_reset | 16 | 0.9998 | 6.497 | 0.013 |
| gpu_seed1101_ckpt300_optimizer_reset | 18 | 1 | 6.382 | 0.01121 |
| gpu_seed1102_ckpt75_optimizer_reset | 15 | 0.9995 | 4.112 | 0.01091 |
| gpu_seed1102_ckpt150_optimizer_reset | 16 | 0.9999 | 8.523 | 0.01019 |
| gpu_seed1102_ckpt300_optimizer_reset | 17 | 0.9999 | 5.755 | 0.007732 |
| gpu_seed1103_ckpt75_optimizer_reset | 14 | 0.9982 | 2.168 | 0.004334 |
| gpu_seed1103_ckpt150_optimizer_reset | 16 | 0.9998 | 4.337 | 0.004505 |
| gpu_seed1103_ckpt300_optimizer_reset | 16 | 1 | 9.429 | 0.00946 |

Evidence is per-case; the files retain the full stepwise moment, update-vector, scheduler, and divergence traces.

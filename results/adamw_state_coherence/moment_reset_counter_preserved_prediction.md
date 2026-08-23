# Moment Reset Counter-Preserved Prediction

A_t = ((1-beta1)/(1-beta1^(t+1))) * sqrt((1-beta2^(t+1))/(1-beta2))

This ignores epsilon for nonzero gradient coordinates and is not exact when epsilon matters.

| checkpoint age | A_t |
|---:|---:|
| 75 | 0.855966249 |
| 150 | 1.184133841 |
| 300 | 1.612555968 |
| 750 | 2.298440469 |
| 1500 | 2.787938663 |
| 3000 | 3.082754156 |

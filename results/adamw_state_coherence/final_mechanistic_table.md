# Final Mechanistic Table

| family | state | median S1 | median update ratio | T_tau | large validation |
|---|---|---:|---:|---|---|
| reset_v | parameters, m, counter, param groups, scheduler preserved; v cleared | 52393.901319445 | 52393.983720938 | existing V5 primary: 9/9 crossings, median T_tau=1 |  |
| isolated full reset | parameters, param groups, scheduler preserved; optimizer per-parameter state empty | 14.799063149 | 15.000038788 | existing V5 primary: 3/9 crossings, delayed to T_tau=14-15 |  |
| reset_mv_preserve_counter | parameters, param groups, scheduler, LR, and counter preserved; m and v cleared | 23.586703011 | 23.795761330 | new 18-case: 15/18 crossings; median among crossers 8.0 |  |

Q1: yes
Q2: no
Q3: no
Q4: not evaluated

PARTIALLY SUPPORTED

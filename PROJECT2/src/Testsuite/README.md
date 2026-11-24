Made by Claude

# Plots

```bash
python plots.py --results ../sequential/results --figs figs
```

Reads every `results/<tag>/<impl>/N{N}_P{P}_rep{r}.npz` and writes the
figures to `figs/`. Runs with the same `(impl, tag, N, P)` are combined by
median over `rep`. Tag `conv` is used for the convergence plot; every other tag
gets speedup / times / overhead / imbalance plots. Meaning you can call the tag anything other than `conv`, this is only used to determine the tolarance since we want max iterations for everything but the convergence plot

## One .npz file = one run

Per-rank arrays have one row per rank (P rows).

| key | shape | meaning |
|---|---|---|
| `timers` | (P, 5) | seconds, columns = `timer_names` |
| `timer_names` | (5,) | `total, compute, comm_p2p, comm_coll, other` (other = total − the rest) |
| `t_setup` | (P,) | decomposition setup time, not in total |
| `work` | (P,) | owned grid points |
| `bytes_sent`, `n_msgs` | (P,) | halo traffic over the whole run |
| `coords` | (P, 3) | position in the process grid |
| `N`, `h`, `P`, `n_iter`, `rep` | scalar | grid size, spacing, ranks, iterations, repetition |
| `residual`, `error_Linf`, `u_norm` | scalar | final residual, max-norm error vs exact solution, RMS of u |
| `meta` | JSON string | `impl`, `tag`, `dims`, `nodes` |

Any implementation that writes this format can be plotted next to the others;
give it its own `impl` name.


## Overview of dictionary from `metrics`

`metrics` turns one run into one row of the
results table. Below, `t_i` is rank `i`'s timer, `T` is the wall time and `P*T` is the total core-time.

| name in dict | meaning |
| --- | --- |
| `impl` | Implementation name, e.g. `disjoint_boundaries` (from `meta`) |
| `tag` | Test type, e.g. `scaling` or `conv` (from `meta`) |
| `N` | Grid size |
| `P` | Number of ranks |
| `rep` | Repetition index of the run |
| `h` | Grid spacing |
| `T` | Wall time: `max_i total_i` (s) |
| `t_compute_max` | Largest compute time over ranks: `max_i compute_i` (s) |
| `t_comm_max` | Largest communication time over ranks: `max_i (comm_p2p_i + comm_coll_i)` (s) |
| `lam` | Load imbalance from compute time: `max_i compute_i / mean_i compute_i − 1` (NaN if no compute time) |
| `f_compute` | Fraction of `P*T` spent computing: `sum_i compute_i / (P*T)` |
| `f_comm_p2p` | Fraction of `P*T` in point-to-point (halo) communication |
| `f_comm_coll` | Fraction of `P*T` in collective communication |
| `f_other` | Fraction of `P*T` in untimed work (`other` timer) |
| `f_idle` | Fraction of `P*T` where ranks were done and waiting: `sum_i (T − total_i) / (P*T)`. The five `f_*` sum to 1 |
| `error_Linf` | Max-norm error vs the exact solution |
| `file` | Path of the `.npz` file the row came from |

`summarize` then takes the median over `rep` for each `(impl, tag, N, P)` and adds:

| name | meaning |
| --- | --- |
| `S` | Speedup `S(P) = T(P0) * P0 / T(P)`, with `P0` the smallest `P` run for that `(impl, tag, N)` |
| `E` | Parallel efficiency `E(P) = S(P) / P` |

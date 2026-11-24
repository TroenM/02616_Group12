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

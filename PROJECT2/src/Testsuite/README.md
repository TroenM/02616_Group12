Made by Claude updated by Gemini
# Plots

```bash
python plots.py --results ../sequential/results --figs figs [--run <one .npz>]
```

Reads every `N{N}_P{P}_rep{r}.npz` recursively under `--results` (for example, `results/<tag>/<impl>/N{N}_P{P}_rep{r}.npz` or `results/<tag>/<impl>/<mapping>/N{N}_P{P}_rep{r}.npz`) and writes the figures to `figs/`.

Runs with the same `(impl, mapping, tag, N, P)` are combined by median over `rep`. Tag `conv` is used for the convergence plot; every other tag generates scaling, throughput, and mapping comparison plots (each skipped automatically if its data is missing).

## Generated figures

| file | description |
|---|---|
| `speedup_<tag>.png` | speedup `S` and parallel efficiency `E` vs `P` (requires ≥ 2 values of `P`) |
| `times_<tag>.png` | total, max-rank compute, and max-rank comm time vs `P` (requires ≥ 2 values of `P`) |
| `overhead_<tag>.png` | fractions of `P·T`: compute, p2p, collective, idle, other |
| `imbalance_<tag>.png` | load imbalance `λ = max/mean − 1` vs `P` (requires ≥ 2 values of `P`) |
| `throughput_<tag>.png` | aggregate `GFLOP/s` and memory ops/s vs `P` (requires `flops` or `mem_ops`) |
| `mapping_<tag>_<impl>_N<N>_P<P>.png` | side-by-side comparison of time breakdown, imbalance, and throughput across MPI rank mappings at fixed `(impl, N, P)` (requires ≥ 2 mappings) |
| `rank_timings.png` | per-rank stacked phases for one run (default: largest `P`, then largest `N`) |
| `bandwidth.png` | effective halo bandwidth vs message size |
| `convergence.png` | max-norm error vs grid spacing `h` with fitted order (tag `conv`) |

## One `.npz` file = one run

Per-rank arrays have one row per rank (`P` rows). Keys marked *(optional)* can be omitted; older `.npz` files remain compatible.

| key | shape | meaning |
|---|---|---|
| `timers` | `(P, 5)` | seconds, columns = `timer_names` |
| `timer_names` | `(5,)` | `total, compute, comm_p2p, comm_coll, other` (`other = total − the rest`) |
| `t_setup` | `(P,)` | decomposition setup time, not in `total` |
| `work` | `(P,)` | owned grid points |
| `bytes_sent`, `n_msgs` | `(P,)` | halo traffic over the whole run |
| `flops` *(optional)* | `(P,)` | total floating-point operations executed per rank over the whole run |
| `mem_ops` *(optional)* | `(P,)` | total memory operations (or bytes) executed per rank over the whole run |
| `coords` | `(P, 3)` | position in the process grid |
| `N`, `h`, `P`, `n_iter`, `rep` | scalar | grid size, spacing, ranks, iterations, repetition |
| `residual`, `error_Linf`, `u_norm` | scalar | final residual, max-norm error vs exact solution, RMS of `u` |
| `meta` | JSON string | `impl`, `tag`, `dims`, `nodes`, and optional `mapping` (defaults to `"default"`) |

Any implementation that writes this format can be plotted next to the others; give it its own `impl` name (and `mapping` name when comparing MPI rank placements).
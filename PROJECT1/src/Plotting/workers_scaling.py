from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures" / "Scaling"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SIZE = (25000, 25000)
LIMS = "-2,2x0,75_-1,3x1,3"
CORES_PER_NODE = 24  # span[ptile=24]
KEEP = (16, 24, 48, 72)  # worker counts to include for Rows/Grid (Dynamic keeps all runs)

# label: (folder, job names in filename, marker, rank 0 computes)
VARIANTS = {
    "Rows":    ("Rows",     ("staticRow", "numworkRow"),            "o", True),
    "Grid":    ("Blocking", ("staticGrid", "numworkGrid"),          "s", True),
    "Dynamic": ("Dynamic",  ("dynamicGrid", "numworkDynamicGrid"),  "^", True),
}
COLORS = {"Rows": "C0", "Grid": "C1", "Dynamic": "C2"}

# Rankstats columns
#   Rows/Grid: col0 = compute, col1 = blocking send (workers) / 0 (rank 0),
#              col2 = 0 (workers) / master wait (rank 0)
#   Dynamic:   col0 = compute, col1 = idle, col2 = comm; rank 0: send, wait


def parse(stem):
    # summary_numworkRow_25000x25000_-2,2x0,75_-1,3x1,3_64_64
    p = stem.split("_")
    w, h = map(int, p[2].split("x"))
    return (w, h), int(p[-1])


def load(folder, jobs, rank0_computes, keep=None, extra_ranks=0):
    runs = {}
    for job in jobs:
        # rglob: Dynamic results also live in the nested Dynamic/Dynamic folder
        for f in (RESULTS / folder).rglob(f"summary_{job}_*_{LIMS}_*.npy"):
            size, n = parse(f.stem)               # n = number of computing ranks
            n += extra_ranks
            if size != SIZE or (keep is not None and n not in keep):
                continue
            summary = np.load(f, allow_pickle=True)
            stats = np.load(f.with_name(f.name.replace("summary_", "rankstats_")),
                            allow_pickle=True)[:, :3]
            comp = stats[:, 0] if rank0_computes else stats[1:, 0]
            ranks = n if rank0_computes else n + 1
            runs[n] = dict(t_total=summary[0], stats=stats, imb=comp.max() / comp.mean(),
                           nodes=math.ceil(ranks / CORES_PER_NODE))
    return dict(sorted(runs.items()))


# Dynamic files store the number of computing workers; add the master rank so the
# count matches the number of ranks (cores) used, e.g. 71 workers -> 72
data = {name: load(folder, jobs, r0, None if name == "Dynamic" else KEEP,
                   extra_ranks=1 if name == "Dynamic" else 0)
        for name, (folder, jobs, _, r0) in VARIANTS.items()}
# New numworkGrid runs (Mandelbrot_Grid_numwork.py) are saved in results/Grid
data["Grid"] = dict(sorted({**data["Grid"],
                            **load("Grid", VARIANTS["Grid"][1], VARIANTS["Grid"][3], KEEP)}.items()))
workers = sorted(set().union(*[d.keys() for d in data.values()]))
if not workers:
    raise SystemExit("No result files found")

print("variant   N  nodes  wall [s]  imbalance  speedup vs N=16  efficiency")
for name, runs in data.items():
    t16 = runs[16]["t_total"] if 16 in runs else None
    for n, r in runs.items():
        s = t16 / r["t_total"] if t16 else float("nan")
        print(f"{name:7} {n:3d} {r['nodes']:5d} {r['t_total']:9.1f} {r['imb']:10.2f}"
              f" {s:16.2f} {s * 16 / n:11.2f}")

# ---- Figure 1: wall time, imbalance, speedup vs number of workers ----
fig, (ax_t, ax_i, ax_s) = plt.subplots(1, 3, figsize=(16, 4.5))
for name, (_, _, marker, _) in VARIANTS.items():
    runs = data[name]
    if not runs:
        continue
    n = np.array(list(runs))
    ax_t.loglog(n, [r["t_total"] for r in runs.values()], marker=marker,
                color=COLORS[name], label=name)
    ax_i.plot(n, [r["imb"] for r in runs.values()], marker=marker,
              color=COLORS[name], label=name)

# Ideal scaling: time halves when the number of workers doubles (from the smallest Rows run)
if data["Rows"]:
    n0 = min(data["Rows"])
    n_ideal = np.array([n0, max(workers)])
    ax_t.loglog(n_ideal, data["Rows"][n0]["t_total"] * n0 / n_ideal, "k--", lw=1,
                label="Ideal scaling")

# Speedup relative to the dynamic grid. Where Dynamic was not run at exactly the same
# worker count, its wall time is interpolated in log-log space (no extrapolation).
dyn_n = np.array(list(data["Dynamic"]), dtype=float)
dyn_t = np.array([r["t_total"] for r in data["Dynamic"].values()])


def t_dynamic(n):
    if n in data["Dynamic"]:
        return data["Dynamic"][n]["t_total"]
    if len(dyn_n) < 2 or not dyn_n[0] <= n <= dyn_n[-1]:
        return None
    return np.exp(np.interp(np.log(n), np.log(dyn_n), np.log(dyn_t)))


ax_s.axhline(1, color=COLORS["Dynamic"], ls="--", lw=1, label="Dynamic (reference)")
print("\nWall time relative to dynamic grid (T_variant / T_dynamic)")
for name, (_, _, marker, _) in VARIANTS.items():
    if name == "Dynamic":
        continue
    pts = [(n, r["t_total"] / t_dynamic(n)) for n, r in data[name].items()
           if t_dynamic(n) is not None]
    for n, s in pts:
        print(f"{name:7} {n:3d} {s:6.2f}" + ("" if n in data["Dynamic"] else "  (interpolated)"))
    if pts:
        ax_s.plot(*zip(*pts), marker=marker, color=COLORS[name], label=name)

# Skip tick labels that would sit on top of a KEEP tick (e.g. Dynamic's 71 next to 72)
ticks = [n for n in workers if n in KEEP or all(abs(n - k) > 2 for k in KEEP)]
for ax in (ax_t, ax_i, ax_s):
    ax.set_xticks(ticks, [str(t) for t in ticks])
ax_t.xaxis.set_minor_formatter(plt.NullFormatter())
ax_t.set(xlabel="Number of workers", ylabel="Total wall time [s]",
         title="Wall time vs number of workers")
ax_i.axhline(1, color="k", ls="--", lw=1, label="Perfect balance")
ax_i.set(xlabel="Number of workers", ylabel="max / mean worker compute time",
         title="Load imbalance", ylim=(0.9, None))
ax_s.set(xlabel="Number of workers", ylabel="$T_{variant} / T_{DynamicGrid}$",
         title="Wall time relative to dynamic grid")
for ax in (ax_t, ax_i, ax_s):
    ax.grid(alpha=0.3, which="both")
for ax in (ax_t, ax_i, ax_s):
    ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Rows_vs_Grid_workers.png", dpi=300)


# ---- Figure 2: per-rank breakdown at the worker counts run for both Rows and Grid ----
rows_n = [n for n in workers if n in data["Rows"] and n in data["Grid"]]
cols = [v for v in VARIANTS if any(n in data[v] for n in rows_n)]
fig, axes = plt.subplots(len(rows_n), len(cols), figsize=(8 * len(cols), 3.2 * len(rows_n)),
                         sharex="row", sharey="row", squeeze=False)
for row, n in enumerate(rows_n):
    for col, name in enumerate(cols):
        ax = axes[row, col]
        if n not in data[name]:
            ax.text(0.5, 0.5, "not run", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{name}  {n} workers")
            continue
        run = data[name][n]
        st = run["stats"]
        ranks = np.arange(len(st))
        if VARIANTS[name][3]:
            # Rank 0 computes, then waits for the others; workers block in their send
            ax.bar(ranks, st[:, 0], color="C0", width=0.9, label="compute")
            ax.bar(ranks[1:], st[1:, 1], bottom=st[1:, 0], color="C1", width=0.9,
                   label="idle (blocking send)")
            ax.bar(0, st[0, 2], bottom=st[0, 0], color="0.3", width=0.9, label="master wait")
        else:
            # Same layout as the static/dynamic per-rank figure in the report
            bottom = np.zeros(len(st) - 1)
            for j, (lab, c) in enumerate(zip(["compute", "idle", "comm"], ["C0", "C1", "C2"])):
                ax.bar(ranks[1:], st[1:, j], bottom=bottom, color=c, width=0.9, label=lab)
                bottom += st[1:, j]
            ax.bar(0, st[0, 1], color="0.6", width=0.9, label="master send")
            ax.bar(0, st[0, 2], bottom=st[0, 1], color="0.3", width=0.9, label="master wait")
        nodes = run["nodes"]
        ax.set_title(f"{name}  {n} workers, {nodes} node{'s' * (nodes > 1)}"
                     f"  (wall time {run['t_total']:.1f} s)")
        ax.grid(alpha=0.3, axis="y")
        if col == 0:
            ax.set_ylabel("Time [s]")

for col in range(len(cols)):
    handles, labels = axes[0, col].get_legend_handles_labels()
    if handles:
        axes[0, col].legend(ncol=3, fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Rank")
fig.tight_layout()
fig.savefig(FIG_DIR / "Rows_vs_Grid_workers_ranks.png", dpi=300)
print(f"Saved figures to {FIG_DIR}")

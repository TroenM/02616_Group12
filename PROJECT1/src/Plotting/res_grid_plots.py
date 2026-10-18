from pathlib import Path
import numpy as np
import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures" / "Grid"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# label: (folder, job name in filename, marker)
VARIANTS = {
    "Row S/B": ("Rows", "staticRow", "d"),
    "Grid S/B": ("Grid", "staticGrid", "o"),
    "Grid D/N": ("Dynamic", "dynamicGrid", "s"),
}
TARGET_WORKERS = ["71", "72"]
# Worker columns in rankstats. Rank 0 is the master: col1 = send, col2 = wait
COLS = ["compute", "idle", "comm"]
COLORS = ["C0", "C1", "C2"]


def parse(stem):
    # summary_dynamicGrid_5000x5000_-2,2x0,75_-1,3x1,3_65536_71
    p = stem.split("_")
    w, h = map(int, p[2].split("x"))
    return w, h


def load(folder, job):
    runs = {}
    for f in (RESULTS / folder).glob(f"summary_{job}_*.npy"):
        if f.name[-6:-4] not in TARGET_WORKERS:
            continue
        w, h = parse(f.stem)
        summary = np.load(f, allow_pickle=True)
        stats = np.load(
            f.with_name(f.name.replace("summary_", "rankstats_")), allow_pickle=True
        )[:, :3]  # cols 3+ unused
        comp = stats[1:, 0]  # workers only
        runs[(w, h)] = dict(
            pixels=w * h, t_total=summary[0], stats=stats, imb=comp.max() / comp.mean()
        )
    return runs


data = {name: load(folder, job) for name, (folder, job, _) in VARIANTS.items()}
sizes = sorted(
    set().union(*[d.keys() for d in data.values()]), key=lambda s: s[0] * s[1]
)
if not sizes:
    raise SystemExit("No result files found")

# ---- Figure 1: wall time, imbalance, speedup vs image size ----
fig, (ax_t, ax_i, ax_s) = plt.subplots(1, 3, figsize=(16, 4.5))
for name, (_, _, marker) in VARIANTS.items():
    runs = data[name]
    ss = [s for s in sizes if s in runs]
    if not ss:
        continue
    px = np.array([runs[s]["pixels"] for s in ss])
    ax_t.loglog(px, [runs[s]["t_total"] for s in ss], marker=marker, label=name)
    ax_i.plot(
        [f"{w // 1000}x{h // 1000}" for w, h in ss],
        [runs[s]["imb"] for s in ss],
        marker=marker,
        label=name,
    )

both = [s for s in sizes if all(s in data[n] for n in VARIANTS)]
if both:
    baseline = "Grid D/N"
    x_labels = [f"{w // 1000}x{h // 1000}" for w, h in both]

    # Plot speedup (Relative execution time vs Grid D/N) for all variants as a line plot
    for name, (_, _, marker) in VARIANTS.items():
        speedups = [
            data[name][s]["t_total"] / data[baseline][s]["t_total"] for s in both
        ]
        ax_s.plot(x_labels, speedups, marker=marker, label=name)

    # Dashed line for No Speedup (Ratio = 1)
    ax_s.axhline(1, color="k", ls="--", lw=1, label="No speedup")

    # Console output updated to reflect all variants
    header_names = "  ".join(f"{n:>11}" for n in VARIANTS.keys())
    print(f"size            {header_names}")
    for s in both:
        times_str = "  ".join(f"{data[n][s]['t_total']:11.1f}" for n in VARIANTS)
        print(f"{s[0]:>6}x{s[1]:<8} {times_str}")

ax_t.set(
    xlabel="Number of pixels",
    ylabel="Total wall time [s]",
    title="Wall time vs image size",
)
ax_i.axhline(1, color="k", ls="--", lw=1, label="Perfect balance")
ax_i.set(
    xlabel="Image size (1000 pixels)",
    ylabel="max / mean worker compute time",
    title="Load imbalance",
)
ax_s.set(
    xlabel="Image size (1000 pixels)",
    ylabel="$T_{variant} / T_{Grid D/N}$",
    title="Relative time vs Grid D/N",
)
for ax in (ax_t, ax_i, ax_s):
    ax.grid(alpha=0.3, which="both")
for ax in (ax_t, ax_i, ax_s):
    ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "Static_vs_Dynamic_size.png", dpi=300)


# ---- Figure 2: per-rank breakdown, only sizes where both variants exist ----
if not both:
    raise SystemExit("No sizes with both Static and Dynamic data")

# Fixed to 'len(VARIANTS)' columns so we don't hit an IndexError when plotting 3 variants
fig, axes = plt.subplots(
    len(both),
    len(VARIANTS),
    figsize=(16, 3.2 * len(both)),
    sharex=True,
    sharey="row",
    squeeze=False,
)
for row, s in enumerate(both):
    for col, name in enumerate(VARIANTS):
        ax = axes[row, col]
        run = data[name][s]
        st = run["stats"]
        workers = np.arange(1, len(st))
        bottom = np.zeros(len(workers))
        for j, (lab, c) in enumerate(zip(COLS, COLORS)):
            ax.bar(workers, st[1:, j], bottom=bottom, color=c, width=0.9, label=lab)
            bottom += st[1:, j]
        ax.bar(0, st[0, 1], color="0.6", width=0.9, label="master send")
        ax.bar(
            0, st[0, 2], bottom=st[0, 1], color="0.3", width=0.9, label="master wait"
        )
        ax.set_title(f"{name}  {s[0]}x{s[1]}  (wall time {run['t_total']:.1f} s)")
        ax.grid(alpha=0.3, axis="y")
        if col == 0:
            ax.set_ylabel("Time [s]")

axes[0, 0].legend(ncol=5, fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("Rank")
fig.tight_layout()
fig.savefig(FIG_DIR / "Static_vs_Dynamic_ranks.png", dpi=300)
print(f"Saved figures to {FIG_DIR}")

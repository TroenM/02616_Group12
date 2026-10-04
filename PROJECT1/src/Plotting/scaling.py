import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANKS = [2, 4, 6, 8, 10, 12, 14, 16]
# VARIANTS = ["Blocking", "Nonblocking"]
# MARKERS = {"Blocking": "o", "Nonblocking": "s"}
VARIANTS = ["Nonblocking"]
MARKERS = {"Nonblocking": "o"}


def load_variant(variant):
    """Return arrays (n_ranks, t_total, imbalance) for the runs that exist."""
    n, t_total, imbalance = [], [], []
    for r in RANKS:
        path = f"../results/{variant}/{variant}_N{r}.npy"
        if not os.path.exists(path):
            continue
        res = np.load(path, allow_pickle=True).item()
        n.append(res["n_ranks"])
        t_total.append(res["t_total"])
        imbalance.append(res["t_comp"].max() / res["t_comp"].mean())
    return np.array(n), np.array(t_total), np.array(imbalance)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ax_time, ax_speed, ax_imb = axes
max_ranks = 0

for variant in VARIANTS:
    n, t, imb = load_variant(variant)
    if len(n) == 0:
        continue
    max_ranks = max(max_ranks, n.max())
    ax_time.plot(n, t, marker=MARKERS[variant], label=variant)

    # Speedup relative to the smallest rank count that was run
    speedup = t[0] / t
    ax_speed.plot(n, speedup, marker=MARKERS[variant], label=variant)
    ax_imb.plot(n, imb, marker=MARKERS[variant], label=variant)

# Ideal speedup: doubling the ranks halves the time
n_ideal = np.array(RANKS)
n_ideal = n_ideal[n_ideal <= max_ranks]
ax_speed.plot(n_ideal, n_ideal / n_ideal[0], "k--", label="Ideal")

ax_time.set_title("Total wall time")
ax_time.set_xlabel("Number of MPI ranks")
ax_time.set_ylabel("Time [s]")

ax_speed.set_title(f"Speedup relative to {RANKS[0]} ranks")
ax_speed.set_xlabel("Number of MPI ranks")
ax_speed.set_ylabel(f"$T_{{{RANKS[0]}}} / T_N$")

ax_imb.set_title("Load imbalance")
ax_imb.set_xlabel("Number of MPI ranks")
ax_imb.set_ylabel("max / mean compute time")
ax_imb.axhline(1, color="k", ls="--", lw=1, label="Perfect balance")

for ax in axes:
    ax.set_xticks(RANKS)
    ax.grid(alpha=0.3)
    ax.legend()

fig.tight_layout()
fig.savefig("../figures/Scaling/Nonblocking.png", dpi=300)
fig.show()

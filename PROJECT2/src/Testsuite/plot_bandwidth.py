from Testsuite.plots import load_runs
import matplotlib.pyplot as plt

df = load_runs("results/memory_bandwidth/disjoint_boundaries")

print(df.columns)

fig, ax = plt.subplots()
N = df["N"].values 
T = df["t_compute_max"].values 
maxit = 200

mops = N**3 * 4*(9)  * maxit
mops_per_second = mops/T


ax.plot(N, mops_per_second/10**9)
ax.set_xlabel("N")
ax.set_ylabel("Memory ops/second [GB/s]")

fig.savefig("figs/memory_bandwidth/mem_bandwidth.png", dpi = 200)

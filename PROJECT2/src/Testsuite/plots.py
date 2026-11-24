"""All report plots from a results folder (see README.md for the file format).

    python plots.py --results ../sequential/results --figs figs [--run <one .npz>]

Figures (each skipped if its data is missing):
    speedup_<tag>.png       speedup S and efficiency E vs P (ideal dashed)
    times_<tag>.png         total / compute / comm time vs P
    overhead_<tag>.png      fractions of P*T: compute, p2p, collective, idle, other
    imbalance_<tag>.png     load imbalance lambda vs P
    rank_timings.png        per-rank stacked phases for one run (default: largest P)
    bandwidth.png           effective halo bandwidth vs message size
    convergence.png         error vs h, log-log, with fitted order (tag "conv")
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP = ["impl", "tag", "N", "P"]
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 12})


def load_run(fname):
    z = np.load(fname)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    d["t"] = {n: d["timers"][:, i] for i, n in enumerate(d["timer_names"])}
    return d


def metrics(d, fname):
    t, P = d["t"], int(d["P"])
    T = float(t["total"].max())
    core = P * T
    comm = t["comm_p2p"] + t["comm_coll"]
    m = d["meta"]
    return dict(
        impl=m["impl"], tag=m["tag"], N=int(d["N"]), P=P, rep=int(d["rep"]),
        h=float(d["h"]), T=T,
        t_compute_max=float(t["compute"].max()), t_comm_max=float(comm.max()),
        lam=float(t["compute"].max() / t["compute"].mean() - 1) if t["compute"].mean() > 0 else np.nan,
        lam_work=float(d["work"].max() / d["work"].mean() - 1),
        f_compute=t["compute"].sum() / core, f_comm_p2p=t["comm_p2p"].sum() / core,
        f_comm_coll=t["comm_coll"].sum() / core, f_other=t["other"].sum() / core,
        f_idle=(T - t["total"]).sum() / core,
        msg_size=float(d["bytes_sent"].sum() / max(1, d["n_msgs"].sum())),
        bw_eff=float(d["bytes_sent"].sum() / t["comm_p2p"].sum()) if t["comm_p2p"].sum() > 0 else np.nan,
        error_Linf=float(d["error_Linf"]),
        file=fname,
    )


def load_runs(root):
    files = sorted(glob.glob(os.path.join(root, "**", "N*_P*_rep*.npz"), recursive=True))
    return pd.DataFrame([metrics(load_run(f), f) for f in files])


def summarize(runs):
    """Median over repetitions, then speedup and efficiency per (impl, tag, N)."""
    num = runs.select_dtypes("number").columns.drop(["N", "P", "rep"], errors="ignore")
    s = runs.groupby(GROUP)[list(num)].median().reset_index()
    s = s.sort_values(GROUP).reset_index(drop=True)
    first = s.groupby(["impl", "tag", "N"])
    P0, T0 = first["P"].transform("first"), first["T"].transform("first")
    s["S"] = T0 * P0 / s["T"]
    s["E"] = s["S"] / s["P"]
    return s


def _save(fig, figs, name):
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, name))
    plt.close(fig)
    print("wrote", os.path.join(figs, name))


def _plog(ax, Ps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ps)
    ax.set_xticklabels([str(p) for p in Ps])
    ax.minorticks_off()


def plot_speedup(s, figs, tag):
    d = s[s.tag == tag]
    if d.P.nunique() < 2:
        return
    Ps = sorted(d.P.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for (impl, N), g in d.groupby(["impl", "N"]):
        lab = f"{impl}, N={N}"
        a1.plot(g.P, g.S, marker="o", label=lab)
        a2.plot(g.P, g.E, marker="o", label=lab)
    a1.plot(Ps, Ps, ls="--", color="gray", lw=1, label="ideal")
    a2.axhline(1, ls="--", color="gray", lw=1)
    for a in (a1, a2):
        _plog(a, Ps)
        a.set_xlabel("ranks P")
    a1.set_yscale("log", base=2)
    a1.set_ylabel("speedup S = T(P0)·P0 / T(P)")
    a2.set_ylabel("parallel efficiency E = S / P")
    a2.set_ylim(0, 1.1)
    a1.legend(fontsize=8)
    fig.suptitle(f"Speedup ({tag})")
    _save(fig, figs, f"speedup_{tag}.png")


def plot_times(s, figs, tag):
    d = s[s.tag == tag]
    if d.empty:
        return
    impls = sorted(d.impl.unique())
    fig, axes = plt.subplots(1, len(impls), figsize=(4.2 * len(impls), 3.4),
                             squeeze=False, sharey=True)
    for ax, impl in zip(axes[0], impls):
        for N, g in d[d.impl == impl].groupby("N"):
            ax.plot(g.P, g["T"], marker="o", label=f"total (N={N})")
            ax.plot(g.P, g.t_compute_max, marker="s", label=f"compute, max rank (N={N})")
            ax.plot(g.P, g.t_comm_max, marker="^", label=f"communication, max rank (N={N})")
        ax.set_yscale("log")
        _plog(ax, sorted(d.P.unique()))
        ax.set_title(impl)
        ax.set_xlabel("ranks P")
    axes[0][0].set_ylabel("time [s]")
    axes[0][-1].legend(fontsize=7)
    _save(fig, figs, f"times_{tag}.png")


def plot_overhead(s, figs, tag):
    d = s[s.tag == tag]
    if d.empty:
        return
    keys = ["f_compute", "f_comm_p2p", "f_comm_coll", "f_idle", "f_other"]
    labels = ["compute", "point-to-point comm", "collective comm", "idle (waiting for slowest)", "other"]
    groups = list(d.groupby(["impl", "N"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.4),
                             squeeze=False, sharey=True)
    for ax, ((impl, N), g) in zip(axes[0], groups):
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for i, (k, lab) in enumerate(zip(keys, labels)):
            v = g[k].clip(lower=0).values
            ax.bar(x, v, bottom=bottom, color=f"C{i}", width=0.7, label=lab)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(g.P.astype(str))
        ax.set_xlabel("ranks P")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{impl}, N={N}")
    axes[0][0].set_ylabel("fraction of P·T (core time)")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, figs, f"overhead_{tag}.png")


def plot_imbalance(s, figs, tag):
    d = s[(s.tag == tag) & (s.P > 1)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for (impl, N), g in d.groupby(["impl", "N"]):
        ax.plot(g.P, g.lam, marker="o", label=f"{impl}, N={N}: measured")
        ax.plot(g.P, g.lam_work, marker="o", ls=":", label=f"{impl}, N={N}: from point count")
    _plog(ax, sorted(d.P.unique()))
    ax.set_xlabel("ranks P")
    ax.set_ylabel("λ = max / mean compute time − 1")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    _save(fig, figs, f"imbalance_{tag}.png")


def plot_rank_timings(fname, figs):
    d = load_run(fname)
    t, P = d["t"], int(d["P"])
    r = np.arange(P)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.35 * P + 2), 3.4))
    bottom = np.zeros(P)
    for i, (k, lab) in enumerate([("compute", "compute"), ("comm_p2p", "point-to-point comm"),
                                  ("comm_coll", "collective comm"), ("other", "other")]):
        v = np.clip(t[k], 0, None)
        ax.bar(r, v, bottom=bottom, color=f"C{i}", width=0.75, label=lab)
        bottom += v
    ax.axhline(t["total"].max(), color="gray", ls="--", lw=1, label="wall time T")
    ax.set_xticks(r)
    ax.set_xlabel("rank")
    ax.set_ylabel("time [s]")
    m = d["meta"]
    ax.set_title(f"{m['impl']}, N={int(d['N'])}, P={P}, dims={'x'.join(map(str, m['dims']))}")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, figs, "rank_timings.png")


def plot_bandwidth(runs, figs):
    app = runs[runs.P > 1].dropna(subset=["bw_eff"])
    if app.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for impl, g in app.groupby("impl"):
        ax.scatter(g.msg_size, g.bw_eff / 1e9, label=f"{impl} halo (effective)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("message size [bytes]")
    ax.set_ylabel("bandwidth [GB/s]")
    ax.legend(fontsize=7)
    _save(fig, figs, "bandwidth.png")


def plot_convergence(s, figs, tag="conv"):
    d = s[s.tag == tag]
    if d.N.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for impl, g in d.groupby("impl"):
        g = g.groupby("N").median(numeric_only=True).reset_index()   # error is P-independent
        p = float(np.polyfit(np.log(g.h), np.log(g.error_Linf), 1)[0])
        ax.plot(g.h, g.error_Linf, marker="o", label=f"{impl} max-norm (order {p:.2f})")
    h = np.sort(d.h.unique())
    ref = d.sort_values("h").error_Linf.iloc[-1] * (h / h[-1]) ** 2
    ax.plot(h, ref, ls="--", color="gray", lw=1, label="O(h²) reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("max-norm error vs exact solution")
    ax.legend(fontsize=7)
    _save(fig, figs, "convergence.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(HERE, "..", "..", "results", "sequential"))
    ap.add_argument("--figs", default=os.path.join(HERE, "..", "..", "results", "figs"))
    ap.add_argument("--run", help="run file for the per-rank plot")
    a = ap.parse_args()

    runs = load_runs(a.results)
    if runs.empty:
        raise SystemExit(f"no N*_P*_rep*.npz files under {a.results}")
    s = summarize(runs)
    for tag in s.tag.unique():
        if tag == "conv":
            continue
        plot_speedup(s, a.figs, tag)
        plot_times(s, a.figs, tag)
        plot_overhead(s, a.figs, tag)
        plot_imbalance(s, a.figs, tag)
    plot_convergence(s, a.figs)
    noconv = runs[runs.tag != "conv"]
    if a.run or not noconv.empty:
        plot_rank_timings(a.run or noconv.sort_values(["P", "N"]).file.iloc[-1], a.figs)
    plot_bandwidth(runs, a.figs)

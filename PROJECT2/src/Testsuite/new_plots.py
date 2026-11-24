"""All report plots from a results folder (see README.md for the file format).

    python plots.py --results ../sequential/results --figs figs [--run <one .npz>]

Figures (each skipped if its data is missing):
    speedup_<tag>.png                  speedup S and efficiency E vs P (ideal dashed)
    times_<tag>.png                    total / compute / comm time vs P
    overhead_<tag>.png                 fractions of P*T: compute, p2p, collective, idle, other
    imbalance_<tag>.png                load imbalance lambda vs P
    throughput_<tag>.png               aggregate GFLOP/s and memory ops/s vs P
    mapping_<tag>_<impl>_N<N>_P<P>.png fixed-(N, P) comparison across MPI rank mappings
    rank_timings.png                   per-rank stacked phases for one run (default: largest P)
    bandwidth.png                      effective halo bandwidth vs message size
    convergence.png                    error vs h, log-log, with fitted order (tag "conv")
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

GROUP = ["impl", "mapping", "tag", "N", "P"]
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 12})


def _label(impl, mapping, N=None):
    base = f"{impl} ({mapping})" if mapping and mapping != "default" else impl
    return f"{base}, N={N}" if N is not None else base


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

    flops_tot = float(d["flops"].sum()) if "flops" in d else np.nan
    mem_tot = float(d["mem_ops"].sum()) if "mem_ops" in d else np.nan

    return dict(
        impl=m["impl"],
        mapping=m.get("mapping", "default"),
        tag=m["tag"],
        N=int(d["N"]),
        P=P,
        rep=int(d["rep"]),
        h=float(d["h"]),
        T=T,
        t_compute_max=float(t["compute"].max()),
        t_comm_max=float(comm.max()),
        lam=float(t["compute"].max() / t["compute"].mean() - 1) if t["compute"].mean() > 0 else np.nan,
        lam_work=float(d["work"].max() / d["work"].mean() - 1),
        f_compute=t["compute"].sum() / core,
        f_comm_p2p=t["comm_p2p"].sum() / core,
        f_comm_coll=t["comm_coll"].sum() / core,
        f_other=t["other"].sum() / core,
        f_idle=(T - t["total"]).sum() / core,
        gflops=flops_tot / T / 1e9 if T > 0 else np.nan,
        mem_rate=mem_tot / T / 1e9 if T > 0 else np.nan,
        msg_size=float(d["bytes_sent"].sum() / max(1, d["n_msgs"].sum())),
        bw_eff=float(d["bytes_sent"].sum() / t["comm_p2p"].sum()) if t["comm_p2p"].sum() > 0 else np.nan,
        error_Linf=float(d["error_Linf"]),
        file=fname,
    )


def load_runs(root):
    files = sorted(glob.glob(os.path.join(root, "**", "N*_P*_rep*.npz"), recursive=True))
    return pd.DataFrame([metrics(load_run(f), f) for f in files])


def summarize(runs):
    """Median over repetitions, then speedup and efficiency per (impl, mapping, tag, N)."""
    num = runs.select_dtypes("number").columns.drop(["N", "P", "rep"], errors="ignore")
    s = runs.groupby(GROUP)[list(num)].median().reset_index()
    s = s.sort_values(GROUP).reset_index(drop=True)
    first = s.groupby(["impl", "mapping", "tag", "N"])
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
    for (impl, mapping, N), g in d.groupby(["impl", "mapping", "N"]):
        lab = _label(impl, mapping, N)
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
    if d.P.nunique() < 2:
        return
    series = list(d.groupby(["impl", "mapping"]))
    fig, axes = plt.subplots(1, len(series), figsize=(4.2 * len(series), 3.4),
                             squeeze=False, sharey=True)
    for ax, ((impl, mapping), dg) in zip(axes[0], series):
        for N, g in dg.groupby("N"):
            ax.plot(g.P, g["T"], marker="o", label=f"total (N={N})")
            ax.plot(g.P, g.t_compute_max, marker="s", label=f"compute, max rank (N={N})")
            ax.plot(g.P, g.t_comm_max, marker="^", label=f"communication, max rank (N={N})")
        ax.set_yscale("log")
        _plog(ax, sorted(d.P.unique()))
        ax.set_title(_label(impl, mapping))
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
    groups = list(d.groupby(["impl", "mapping", "N"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.4),
                             squeeze=False, sharey=True)
    for ax, ((impl, mapping, N), g) in zip(axes[0], groups):
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
        ax.set_title(_label(impl, mapping, N))
    axes[0][0].set_ylabel("fraction of P·T (core time)")
    axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, figs, f"overhead_{tag}.png")


def plot_imbalance(s, figs, tag):
    d = s[(s.tag == tag) & (s.P > 1)]
    if d.P.nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for (impl, mapping, N), g in d.groupby(["impl", "mapping", "N"]):
        lab = _label(impl, mapping, N)
        ax.plot(g.P, g.lam, marker="o", label=f"{lab}: measured")
        ax.plot(g.P, g.lam_work, marker="o", ls=":", label=f"{lab}: from point count")
    _plog(ax, sorted(d.P.unique()))
    ax.set_xlabel("ranks P")
    ax.set_ylabel("λ = max / mean compute time − 1")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    _save(fig, figs, f"imbalance_{tag}.png")


def plot_throughput(s, figs, tag):
    d = s[s.tag == tag].dropna(subset=["gflops", "mem_rate"], how="all")
    if d.empty or d.P.nunique() < 2:
        return
    Ps = sorted(d.P.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for (impl, mapping, N), g in d.groupby(["impl", "mapping", "N"]):
        lab = _label(impl, mapping, N)
        if g.gflops.notna().any():
            a1.plot(g.P, g.gflops, marker="o", label=lab)
        if g.mem_rate.notna().any():
            a2.plot(g.P, g.mem_rate, marker="o", label=lab)
    for a in (a1, a2):
        _plog(a, Ps)
        a.set_xlabel("ranks P")
    a1.set_ylabel("throughput [GFLOP/s]")
    a2.set_ylabel("memory rate [G-ops/s]")
    a1.legend(fontsize=8)
    fig.suptitle(f"Throughput ({tag})")
    _save(fig, figs, f"throughput_{tag}.png")


def plot_mapping_comparison(s, figs, tag):
    """Compare MPI rank mappings side-by-side for each fixed (impl, N, P)."""
    d = s[s.tag == tag]
    if d.empty or d["mapping"].nunique() < 2:
        return
    for (impl, N, P), g in d.groupby(["impl", "N", "P"]):
        if g["mapping"].nunique() < 2:
            continue
        g = g.sort_values("mapping")
        x = np.arange(len(g))
        has_tput = g[["gflops", "mem_rate"]].notna().any().any()
        ncols = 3 if has_tput else 2
        fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.6))

        # 1. Time breakdown (max rank)
        axes[0].bar(x, g.t_compute_max, label="compute (max)")
        axes[0].bar(x, g.t_comm_max, bottom=g.t_compute_max, label="comm (max)")
        axes[0].set_ylabel("time [s]")
        axes[0].set_title("Time breakdown")
        axes[0].legend(fontsize=7)

        # 2. Load imbalance
        axes[1].bar(x, g.lam, color="C2", label="measured")
        axes[1].set_ylabel("λ = max / mean − 1")
        axes[1].set_ylim(bottom=0)
        axes[1].set_title("Load imbalance")

        # 3. Compute & memory throughput (if present)
        if has_tput:
            w = 0.35
            if g.gflops.notna().any():
                axes[2].bar(x - w / 2, g.gflops.fillna(0), width=w, label="GFLOP/s")
            if g.mem_rate.notna().any():
                axes[2].bar(x + w / 2, g.mem_rate.fillna(0), width=w, label="Mem G-ops/s")
            axes[2].set_ylabel("rate")
            axes[2].set_title("Throughput")
            axes[2].legend(fontsize=7)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(g.mapping, rotation=20, ha="right")

        fig.suptitle(f"MPI Mapping Comparison — {impl}, N={N}, P={P} ({tag})")
        _save(fig, figs, f"mapping_{tag}_{impl}_N{N}_P{P}.png")


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
    title = f"{_label(m['impl'], m.get('mapping', 'default'), int(d['N']))}, P={P}"
    if "dims" in m:
        title += f", dims={'x'.join(map(str, m['dims']))}"
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    _save(fig, figs, "rank_timings.png")


def plot_bandwidth(runs, figs):
    app = runs[runs.P > 1].dropna(subset=["bw_eff"])
    if app.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (impl, mapping), g in app.groupby(["impl", "mapping"]):
        ax.scatter(g.msg_size, g.bw_eff / 1e9, label=f"{_label(impl, mapping)} halo (effective)")
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
        plot_throughput(s, a.figs, tag)
        plot_mapping_comparison(s, a.figs, tag)
    plot_convergence(s, a.figs)
    noconv = runs[runs.tag != "conv"]
    if a.run or not noconv.empty:
        plot_rank_timings(a.run or noconv.sort_values(["P", "N"]).file.iloc[-1], a.figs)
    plot_bandwidth(runs, a.figs)
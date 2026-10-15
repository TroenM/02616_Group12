import os
import numpy as np

"""
Shared saving of timing data, following the datafile layouts in the report.

Files are saved to ../data relative to this file:
    summary_<JOB>_<PIXELCOUNT>_<XLIM>_<YLIM>_<NUMGRIDS>_<NUMWORKERS>.npy
    rankstats_<JOB>_<PIXELCOUNT>_<XLIM>_<YLIM>_<NUMGRIDS>_<NUMWORKERS>.npy
ex: summary_staticGrid_1000x1000_-2,2x0,75_-1,3x1,3_16_4.npy

Summary layout (1D array, see SUMMARY_FIELDS for the index of each value)
RankStats layout (2D array, idx is rank): [time rank computation, rank idle time]
"""

DATA_DIR = os.path(r"")

SUMMARY_FIELDS = (
    "total_time",
    "master_send_time",
    "master_wait_time",
    "total_computation_time",
    "total_idle_time",
    "size_x",
    "size_y",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "num_grids",
    "num_workers",
)


def _fmt_lim(lim):
    # -2.2, 0.75 -> "-2,2x0,75"
    return "x".join(f"{v:g}".replace(".", ",") for v in lim)


def data_name(job, size, xlim, ylim, num_grids, num_workers):
    return (
        f"{job}_{size[0]}x{size[1]}_{_fmt_lim(xlim)}_{_fmt_lim(ylim)}_{num_grids}_{num_workers}"
    )


def save_run_data(
    job,
    size,
    xlim,
    ylim,
    num_grids,
    num_workers,
    total_time,
    master_send_time,
    master_wait_time,
    rankstats,
):
    rankstats = np.asarray(rankstats, dtype=float)

    summary = np.array(
        [
            total_time,
            master_send_time,
            master_wait_time,
            rankstats[:, 0].sum(),
            rankstats[:, 1].sum(),
            size[0],
            size[1],
            xlim[0],
            xlim[1],
            ylim[0],
            ylim[1],
            num_grids,
            num_workers,
        ],
        dtype=float,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    name = data_name(job, size, xlim, ylim, num_grids, num_workers)
    np.save(os.path.join(DATA_DIR, f"summary_{name}.npy"), summary)
    np.save(os.path.join(DATA_DIR, f"rankstats_{name}.npy"), rankstats)

    print(f"Saved data files for {name} in {os.path.normpath(DATA_DIR)}")
    return summary

import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")
)
from run_data import save_run_data, _fmt_lim

total_time = -perf_counter()

"""
ABOUT THE IMPLEMENTATION:

This version takes input 
    number of chunks per axis
    number of ranks

Then divides the work out to each ranks
if the number of chunks is not divisible by number of ranks, then it divides the work out on the other ranks starting from 1.

"""


_help = f"""\
{sys.argv[0]} [chunks-per-axis] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with 4x4 = 16 chunks
$ mpirun -n 4 python {sys.argv[0]} 4

Call it with 8x8 chunks and image size of 100 by 500 pixels
$ mpirun -n 4 python {sys.argv[0]} 8 100x500
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

if size_mpi < 2:
    print("Need at least 2 ranks (rank 0 only distributes work)")
    sys.exit(1)

# Defaults
chunk_per_axis = 4
n_chunks = chunk_per_axis**2
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Arguments
argv = sys.argv[1:]
if argv:
    chunk_per_axis = int(argv.pop(0))
    n_chunks = chunk_per_axis**2
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

if rank == 0:
    print(f"""\
Calculating the Mandelbrot set with these arguments:
{n_chunks = }
{size = }
{xlim = }
{ylim = }
{size_mpi = }
""")

# Convert to numpy arrays
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
image = np.zeros(size)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

n_x = size[0]
n_y = size[1]

# Same on every rank, so sender and receiver agree on buffer size
n_workers = size_mpi - 1
n_even_distributed = n_chunks // n_workers
n_mismatch = n_chunks % n_workers
workload = n_even_distributed + 1  # rows per message (last row may be zero padding)

total_comparable_time = -perf_counter()
if rank == 0:
    chunk_map = np.zeros((n_chunks, 5))
    master_send_time = 0
    master_wait_time = 0

    for idx in range(chunk_map.shape[0]):
        x_coord = idx // chunk_per_axis
        y_coord = idx % chunk_per_axis

        start_x = x_coord * n_x // chunk_per_axis
        end_x = (x_coord + 1) * n_x // chunk_per_axis

        start_y = y_coord * n_y // chunk_per_axis
        end_y = (y_coord + 1) * n_y // chunk_per_axis

        chunk_map[idx] = [idx, start_x, end_x, start_y, end_y]

    # send work - now changed to outside the idx loop
    z = 0
    sent = {}
    for r in range(1, size_mpi):
        rank_data = np.zeros((workload, 5))
        # added "* n_even_distributed" to avoid overlapping intervals
        rank_data[:n_even_distributed] = chunk_map[
            (r - 1) * n_even_distributed : r * n_even_distributed
        ]

        if r <= n_mismatch:
            rank_data[-1] = chunk_map[n_even_distributed * n_workers + z]
            z += 1

        my_rows = workload if r <= n_mismatch else workload - 1
        sent[r] = rank_data[:my_rows]

        master_send_time -= perf_counter()
        comm.Send(rank_data, dest=r)
        master_send_time += perf_counter()

    print(f"MPI grid: {chunk_per_axis} x {chunk_per_axis}")

    # added method for rank 0 to recieving images and timings from the workers
    rank_times = np.zeros((n_workers, workload, 5))

    for other_rank in range(1, size_mpi):
        chunks = sent[other_rank]
        n_pix = int(sum((ex - sx) * (ey - sy) for _, sx, ex, sy, ey in chunks))
        packed = np.empty(n_pix, dtype=np.float64)

        master_wait_time -= perf_counter()
        comm.Recv(packed, source=other_rank)
        master_wait_time += perf_counter()

        # unpack chunk by chunk.
        offset = 0
        for _, sx, ex, sy, ey in chunks:
            sx, ex, sy, ey = int(sx), int(ex), int(sy), int(ey)
            n = (ex - sx) * (ey - sy)
            image[sx:ex, sy:ey] = packed[offset:offset + n].reshape(ex - sx, ey - sy)
            offset += n

        master_wait_time -= perf_counter()
        comm.Recv(rank_times[other_rank - 1], source=other_rank)
        master_wait_time += perf_counter()

    rank_times = rank_times.reshape(-1, 5)

    # Master rank does no computation
    local_stats = np.array([0.0, master_send_time, master_wait_time])

else:
    rank_recv_time = 0
    rank_send_time = 0
    rank_comp_time = 0
    mem = np.zeros((workload, 5))

    rank_recv_time -= perf_counter()
    comm.Recv(mem, source=0)
    rank_recv_time += perf_counter()

    # drop the extra space row if this rank did not get an extra chunk
    my_workload = workload if rank <= n_mismatch else workload - 1
    mem = mem[:my_workload]
    local_image = np.zeros(size)

    t_rank_computaion = -perf_counter()  # Keep this generating the heatmap
    rank_comp_time -= perf_counter()  # For computing load imbalancing

    chunk_times = np.zeros(my_workload)  # time spent on each of this rank's chunks

    for k, (_, start_x, end_x, start_y, end_y) in enumerate(mem):
        t_chunk = -perf_counter()
        for x in range(int(start_x), int(end_x)):
            cx = complex(xlim[0] + x * xconst, 0)

            for y in range(int(start_y), int(end_y)):
                c = cx + complex(0, ylim[0] + y * yconst)
                z = 0

                for i in range(100):
                    z = z * z + c

                    if abs(z) > 2:
                        local_image[x, y] = i
                        break
        chunk_times[k] = t_chunk + perf_counter()

    rank_comp_time += perf_counter()
    t_rank_computaion += perf_counter()
    # print(f"Rank {rank}: {my_workload} chunks, chunk ids {mem[:, 0].astype(int)}, {t_rank_computaion:.3f} s")
    packed = np.concatenate([
        local_image[int(sx):int(ex), int(sy):int(ey)].ravel()
        for _, sx, ex, sy, ey in mem
    ])
    # Send information from worker to rank 0
    rank_send_time -= perf_counter()
    comm.Send(packed, dest=0)
    rank_send_time += perf_counter()

    time_pckg = np.zeros((workload, 5))
    time_pckg[:my_workload] = mem
    time_pckg[:my_workload, 0] = chunk_times
    rank_send_time -= perf_counter()
    comm.Send(time_pckg, dest=0)
    rank_send_time += perf_counter()

    local_stats = np.array([rank_comp_time, rank_send_time, rank_recv_time])

# Gather the local data and save it
rankstats = np.zeros((size_mpi, 3)) if rank == 0 else None
comm.Gather(local_stats, rankstats, root=0)

MPI.Finalize()

total_time += perf_counter()
total_comparable_time += perf_counter()

if rank == 0:
    # Every rank computes one block, so grids = workers = size_mpi
    summary = save_run_data(
        "staticGrid",
        size,
        xlim,
        ylim,
        n_chunks,
        size_mpi,
        total_time,
        master_send_time,
        master_wait_time,
        rankstats,
        subdir="Grid",
    )
    print(f"""
    Total time =            {total_time:.4f}
    Total comparable Time = {total_comparable_time:.4f}
    Master send time =      {master_send_time:.4f}
    Master wait time =      {master_wait_time:.4f}
    Total idle time =       {summary[4]:.4f}
    Computation time =      {summary[3]:.4f}""")


# Plot the complete image
if rank == 0 and size[0] <= 5000:
    import matplotlib.pyplot as plt

    # 1. Create exact X and Y coordinates matching your loops
    x_vals = xlim[0] + np.arange(size[0]) * xconst
    y_vals = ylim[0] + np.arange(size[1]) * yconst
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")

    # 2. Build a 2D grid for the computation times
    time_grid = np.zeros(size)

    for row in rank_times:  # padding rows are all zeros -> empty slice, no effect
        t, chunk_start_x, chunk_end_x, chunk_start_y, chunk_end_y = row

        time_grid[
            int(chunk_start_x) : int(chunk_end_x), int(chunk_start_y) : int(chunk_end_y)
        ] = t

    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # imshow keeps each block a sharp square (contourf would blur the edges)
    time_plot = ax.imshow(
        time_grid.T,
        extent=np.concatenate([xlim, ylim]),
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )
    cbar = fig.colorbar(time_plot, ax=ax)
    cbar.set_label("Computation time per block (seconds)")

    ax.contour(X.T, Y.T, image.T, levels=[0.5], colors="black", linewidths=1.5)

    ax.set_title(f"Mandelbrot Set & MPI Rank Computation Times ({size_mpi} Ranks)")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")

    # Save and display the result
    plt.savefig(f"../figures/Blocking/Grid_Blocking_{size_mpi}_grid.png", dpi=300)
    plt.show()

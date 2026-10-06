import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter

_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with a chunk-size of 10
$ mpirun -n 4 python {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ mpirun -n 4 python {sys.argv[0]} 10 100x500
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

# Defaults
chunk_size = 10
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

if rank == 0:
    print(f"""\
Calculating the Mandelbrot set with these arguments:

{chunk_size = }
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

# Divide into grid of equal chunks

n_x = size[0]
n_y = size[1]

#size_mpi must be perfect square.
n_sub = int(np.sqrt(size_mpi))

if n_sub * n_sub != size_mpi:
    if rank == 0:
        print(
            f"Error: number of MPI ranks ({size_mpi}) must be a perfect square"
        )
    sys.exit(1)

x_coord = rank // n_sub
y_coord = rank % n_sub

start_x = x_coord * n_x // n_sub
end_x   = (x_coord + 1) * n_x // n_sub

start_y = y_coord * n_y // n_sub
end_y   = (y_coord + 1) * n_y // n_sub

# Size of this rank's local image
local_nx = end_x - start_x
local_ny = end_y - start_y

local_image = np.zeros((local_nx, local_ny))

if rank == 0:
    print(f"MPI grid: {n_sub} x {n_sub}")

#print each rank's assigned region
print(
    f"Rank {rank}: "
    f"x = [{start_x}, {end_x}), "
    f"y = [{start_y}, {end_y}), "
    f"local size = {local_nx} x {local_ny}"
)

# Each row contains [computation_time, start_idx, end_idx], row_idx = rank
rank_times = np.ones((size_mpi, 5)) * (-1) #insert -1 catch missing values (easier debugging)

# Compute this rank's part of the Mandelbrot set
t_rank_computaion = -perf_counter()

for x in range(start_x, end_x):
    cx = complex(xlim[0] + x * xconst, 0)

    for y in range(start_y, end_y):
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0

        for i in range(100):
            z = z * z + c

            if np.abs(z) > 2:
                local_image[x - start_x, y - start_y] = i
                break


t_rank_computaion += perf_counter()

# Send all local images to rank 0
if rank == 0:

    image[
        start_x:end_x,
        start_y:end_y
    ] = local_image

    rank_times[0, :] = np.array([
        t_rank_computaion,
        start_x,
        end_x,
        start_y,
        end_y
    ])

    #recieve from the others
    for other_rank in range(1, size_mpi):

        #compute other rank's position in the grid
        other_x_coord = other_rank // n_sub
        other_y_coord = other_rank % n_sub

        other_start_x = (
            other_x_coord * n_x // n_sub
        )

        other_end_x = (
            (other_x_coord + 1) * n_x // n_sub
        )

        other_start_y = (
            other_y_coord * n_y // n_sub
        )

        other_end_y = (
            (other_y_coord + 1) * n_y // n_sub
        )


        # Size of the block we expect
        other_nx = other_end_x - other_start_x
        other_ny = other_end_y - other_start_y


        other_image = np.zeros(
            (other_nx, other_ny),
            dtype=np.float64
        )

        comm.Recv(
            other_image,
            source=other_rank
        )

        #copy the received block into global image

        image[
            other_start_x:other_end_x,
            other_start_y:other_end_y
        ] = other_image

        #get timing information

        comm.Recv(
            rank_times[other_rank, :],
            source=other_rank
        )


else:

    #Send information from other to rank 0
    comm.Send(
        local_image,
        dest=0
    )

    time_pckg = np.array([
        t_rank_computaion,
        start_x,
        end_x,
        start_y,
        end_y
    ], dtype=np.float64)

    comm.Send(
        time_pckg,
        dest=0
    )

# Plot the complete image

if rank == 0:

    import matplotlib.pyplot as plt

    # 1. Create exact X and Y coordinates matching your loops
    x_vals = xlim[0] + np.arange(size[0]) * xconst
    y_vals = ylim[0] + np.arange(size[1]) * yconst
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

    # 2. Build a 2D grid for the computation times
    time_grid = np.zeros(size)

    for row in rank_times:
        t, chunk_start_x, chunk_end_x, chunk_start_y, chunk_end_y = row

        time_grid[
            int(chunk_start_x):int(chunk_end_x),
            int(chunk_start_y):int(chunk_end_y)
        ] = t


    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Transpose the arrays (.T) so the X-axis is horizontal and Y-axis is vertical
    # Use contourf for the computation times
    time_plot = ax.contourf(X.T, Y.T, time_grid.T, levels=50, cmap='viridis')
    cbar = fig.colorbar(time_plot, ax=ax)
    cbar.set_label('Computation Time (seconds)')

    # Draw the Mandelbrot set boundary as a black line
    # The set is 0, the outside is >= 1. 0.5 perfectly captures the border.
    ax.contour(X.T, Y.T, image.T, levels=[0.5], colors='black', linewidths=1.5)

    ax.set_title(f'Mandelbrot Set & MPI Rank Computation Times ({size_mpi} Ranks)')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    
    # Save and display the result
    plt.savefig(f'../figures/Blocking_Times_N{size_mpi}_grid.png', dpi=300)
    plt.show()


    ####### Standard plot ############
    fig, ax = plt.subplots()


    plt.rcParams.update({
        "font.size": 10,
    })

    plt.imshow(
        image.T,
        extent=np.concatenate([xlim, ylim])
    )

    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    plt.margins(0, 0)

    plt.savefig(
        "../figures/Figure_1.png",
        bbox_inches="tight",
        pad_inches=0
    )






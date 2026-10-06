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

if rank == 0:
    # TODO: use Chistians grid map to distribute (glb_idx, start_x, end_x, start_y, end_y)
    pass
else:
    # TODO: Remove hard coded nx ny
    nx, ny = 5, 5
    # Prepare buffers
    NUM_BUFFERS = 3
    num_calls = 0  # Used for choosing the buffer dimension
    image_buffer = np.zeros(
        (NUM_BUFFERS, nx, ny)
    )  # Create three dimensions (each is an image)
    info_buffer = np.zeros((5, NUM_BUFFERS))

    # TODO: Handle double sends for fisrt comm (probably done in rank 0)
    while True:
        # Choose the current buffers to use
        current_image = image_buffer[num_calls % NUM_BUFFERS]
        current_info = info_buffer[num_calls % NUM_BUFFERS]

        # Recieve work (tag = 0 for work)
        comm.Irecv(info_buffer, source=0, tag=0)
        glb_idx, start_x, end_x, start_y, end_y = info_buffer.astype(int)

        # Work is done
        # TODO: Get num_grids from somewhere
        if glb_idx >= num_grids:
            break

        # Perform computation
        t_rank_computaion = -perf_counter()
        for x in range(start_x, end_x):
            cx = complex(xlim[0] + x * xconst, 0)

            for y in range(start_y, end_y):
                c = cx + complex(0, ylim[0] + y * yconst)
                z = 0

                for i in range(100):
                    z = z * z + c

                    if np.abs(z) > 2:
                        current_image[x - start_x, y - start_y] = i
                        break
        t_rank_computaion += perf_counter()

        comm.Isend(current_image, dest=0)
        num_calls += 1

    # TODO: Shutdown worker

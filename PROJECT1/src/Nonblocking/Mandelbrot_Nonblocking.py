import sys
import numpy as np
from mpi4py import MPI

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

def free(req):
    try:
        req.free()
    except: pass

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

# Divide the x-direction into equal chunks
n_x = size[0]

# Determine which x-values belong to this rank
start = rank * n_x // size_mpi
end = (rank + 1) * n_x // size_mpi

local_image = np.zeros((end - start, size[1]))

# Compute this rank's part of the Mandelbrot set

for x in range(start, end):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z * z + c
            if np.abs(z) > 2:
                local_image[x - start, y] = i
                break

# Send all local images to rank 0

if rank == 0:
    # Rank 0 already created its own work
    image[start:end, :] = local_image
    # Receive the other ranks
    for other_rank in range(1, size_mpi):
        other_start = other_rank * n_x // size_mpi
        other_end = (other_rank + 1) * n_x // size_mpi
        req = comm.Irecv(
            image[other_start:other_end, :],
            source=other_rank
        )
        req.Wait()
        free(req)
else:
    # Blocking send
    req = comm.Isend(
        local_image,
        dest=0
    )
    req.Wait()
    free(req)
# ---------------------------------------------------------
# Plot the complete image
# ---------------------------------------------------------


if rank == 0:

    import matplotlib.pyplot as plt

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
        "/zhome/f3/8/187578/Desktop/02616_LSM/02616_Group12/PROJECT1/figures/Nonblocking/Figure_Nonblocking.png",
        bbox_inches="tight",
        pad_inches=0
    )
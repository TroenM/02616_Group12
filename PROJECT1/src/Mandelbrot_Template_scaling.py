import os
import sys
import numpy as np
from time import perf_counter

"""
Serial (1 rank) baseline for the scaling test.

Same computation as Mandelbrot_Template.py, but timed and saved in the same
format as Blocking/Mandelbrot_Blocking_scaleTest.py, so Plotting/scaling.py
can compute speedup as T_1 / T_N.

Note: uses abs(z) instead of np.abs(z) (as the MPI versions do), since
np.abs on a Python complex is much slower and would inflate the baseline.
"""

_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ python {sys.argv[0]} 10 100x500
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# Defaults (must match the MPI scaling runs)
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

print(f"""\
Calculating the Mandelbrot set with these arguments:

{chunk_size = }
{size = }
{xlim = }
{ylim = }
""")

# Convert to numpy arrays
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
image = np.zeros(size)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

t_total = - perf_counter()
for x in range(size[0]):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z * z + c
            if abs(z) > 2:
                image[x, y] = i
                break
t_total += perf_counter()

print(f"Serial wall time: {t_total:.3f} s")

# ---------------------------------------------------------
# Save stuff (same layout as the Blocking/Nonblocking results)
# ---------------------------------------------------------
results_dir = "../results/Serial"
os.makedirs(results_dir, exist_ok=True)
results = {
    "variant": "Serial",
    "n_ranks": 1,
    "size": size,
    "xlim": xlim,
    "ylim": ylim,
    "t_total": t_total,  # wall time (s), no communication
    "t_comp": np.array([t_total]),  # compute time per rank (s)
    "start": np.array([0]),
    "end": np.array([size[0]]),
}
np.save(f"{results_dir}/Serial_N1.npy", results)

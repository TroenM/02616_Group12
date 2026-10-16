import matplotlib.pyplot as plt
import numpy as np
import os

print(os.getcwd())

image = np.load(
    "../results/Dynamic/image_dynamicGrid_5000x5000_-2,2x0,75_-1,3x1,3_65536_71.npy"
)
rank_times = np.load(
    "../results/Dynamic/rankstats_dynamicGrid_5000x5000_-2,2x0,75_-1,3x1,3_65536_71.npy"
)

xlim = [-2.2, 0.75]
ylim = [-1.3, 1.3]
size = (5000, 5000)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

if rank_times.shape[1] == 3:
#TODO: Dummy positions

# 1. Create exact X and Y coordinates matching your loops
x_vals = xlim[0] + np.arange(size[0]) * xconst
y_vals = ylim[0] + np.arange(size[1]) * yconst
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# 2. Build a 2D grid for the computation times
time_grid = np.zeros(size)

for row in rank_times:  # padding rows are all zeros -> empty slice, no effect
    t, chunk_start_x, chunk_end_x, chunk_start_y, chunk_end_y = row

    time_grid[
        int(chunk_start_x):int(chunk_end_x),
        int(chunk_start_y):int(chunk_end_y)
    ] = t

# 3. Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# imshow keeps each block a sharp square (contourf would blur the edges)
time_plot = ax.imshow(time_grid.T, extent=np.concatenate([xlim, ylim]), origin='lower',
                    cmap='viridis', interpolation='nearest', aspect='auto')
cbar = fig.colorbar(time_plot, ax=ax)
cbar.set_label('Computation time per block (seconds)')

ax.contour(X.T, Y.T, image.T, levels=[0.5], colors='black', linewidths=1.5)

ax.set_title(f'Mandelbrot Set & MPI Rank Computation Times ({size_mpi} Ranks)')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')

    # Save and display the result
    plt.savefig(f'PROJECT1/figures/Blocking/Grid_Blocking_{size_mpi}_grid.png', dpi=300)
    plt.show()

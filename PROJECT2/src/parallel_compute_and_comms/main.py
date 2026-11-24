import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from parallel_compute_and_comms.arguments import get_args
from parallel_compute_and_comms.functions import jacobi_range, sol
from parallel_compute_and_comms.mpi_helper import (
    create_comm_datatypes,
    get_local_extent,
)

# Read arguments
args = get_args()
NUM_DECOMPS_PER_AXIS = args.dim
Nx, Ny, Nz = args.nodes
MAXIT = args.maxit
TOL = args.tol

# Define cartesian communicator
comm = MPI.COMM_WORLD
dims = MPI.Compute_dims(comm.Get_size(), NUM_DECOMPS_PER_AXIS)
cart_comm = comm.Create_cart(dims, periods=[False] * 3, reorder=True)
coords = cart_comm.Get_coords(comm.rank)

if comm.rank == 0:
    print(f"{MAXIT=}, {TOL=}")


# Link communications across ranks
neighbors = {}
for axis in range(3):
    backwards, forwards = cart_comm.Shift(axis, 1)
    neighbors[axis] = {"forward": forwards, "backward": backwards}

# Get local sizes and limits
nx, x_start, x_end = get_local_extent(Nx, dims[0], coords[0])
ny, y_start, y_end = get_local_extent(Ny, dims[1], coords[1])
nz, z_start, z_end = get_local_extent(Nz, dims[2], coords[2])

# Define derived constants
deltax, deltay, deltaz = (
    2 / (Nx + 1),
    2 / (Ny + 1),
    2 / (Nz + 1),
)  # Assuming domain is [-1, 1] in all directions
delta = (deltax, deltay, deltaz)

# Get the physical orign of the rank
phys_origin = (-1 + x_start * deltax, -1 + y_start * deltay, -1 + z_start * deltaz)


# Define the communication datatypes
datatypes = create_comm_datatypes(nx, ny, nz, MPI.DOUBLE)

# Define solution buffers with halo regions
u_curr = np.zeros((nx + 2, ny + 2, nz + 2))
u_new = np.zeros_like(u_curr)

# Main loop
# TODO: move to a seperate function once it works

for iter in range(MAXIT):
    # Post boundary send/recv requests
    recv_reqs = []
    send_reqs = []
    for axis in range(3):
        recv_reqs.append(
            cart_comm.Irecv(
                [u_curr, 1, datatypes[axis]["forward_recv"]], neighbors[axis]["forward"]
            )
        )
        recv_reqs.append(
            cart_comm.Irecv(
                [u_curr, 1, datatypes[axis]["backward_recv"]],
                neighbors[axis]["backward"],
            )
        )
        send_reqs.append(
            cart_comm.Isend(
                [u_curr, 1, datatypes[axis]["forward_send"]], neighbors[axis]["forward"]
            )
        )
        send_reqs.append(
            cart_comm.Isend(
                [u_curr, 1, datatypes[axis]["backward_send"]],
                neighbors[axis]["backward"],
            )
        )

    # Compute interior points
    res_sq = 0
    res_sq += jacobi_range(u_curr, u_new, (2, nx), (2, ny), (2, nz), delta, phys_origin)

    # TODO: Consider whether we need to wait for all recv_reqs to finish before fitting boundaries
    MPI.Request.Waitall(recv_reqs)

    # Left boundary
    res_sq += jacobi_range(
        u_curr, u_new, (1, 2), (1, ny + 1), (1, nz + 1), delta, phys_origin
    )

    # Top boundary
    res_sq += jacobi_range(
        u_curr, u_new, (1, nx + 1), (1, 2), (1, nz + 1), delta, phys_origin
    )

    # Front boundary
    res_sq += jacobi_range(
        u_curr, u_new, (1, nx + 1), (1, ny + 1), (1, 2), delta, phys_origin
    )

    # Right boundary
    if nx > 1:
        res_sq += jacobi_range(
            u_curr, u_new, (nx, nx + 1), (1, ny + 1), (1, nz + 1), delta, phys_origin
        )
    # Bottom boundary
    if ny > 1:
        res_sq += jacobi_range(
            u_curr, u_new, (1, nx + 1), (ny, ny + 1), (1, nz + 1), delta, phys_origin
        )

    # Back boundary
    if nz > 1:
        res_sq += jacobi_range(
            u_curr, u_new, (1, nx + 1), (1, ny + 1), (nz, nz + 1), delta, phys_origin
        )

    # Ensure neighbors have recieved boundaries before overwriting
    MPI.Request.Waitall(send_reqs)

    # Flip solution pointers
    u_curr, u_new = u_new, u_curr

    # Collect residuals
    global_res_sq = cart_comm.allreduce(res_sq, op=MPI.SUM)
    resiudal = np.sqrt(global_res_sq)
    if resiudal < TOL:
        # All requests have been finished in the final wait all
        # So safe to break and finalize
        break

# Esnure all reqs are completed before finalizing
MPI.Request.Waitall(recv_reqs)
MPI.Request.Waitall(send_reqs)

# MPI.Finalize()

# Run local test of solution
local_x = phys_origin[0] + np.arange(1, nx + 1) * deltax
local_y = phys_origin[1] + np.arange(1, ny + 1) * deltay
local_z = phys_origin[2] + np.arange(1, nz + 1) * deltaz
X, Y, Z = np.meshgrid(local_x, local_y, local_z, indexing="ij")
true_sol = sol(X, Y, Z)

u_interior = u_curr[1:-1, 1:-1, 1:-1]

# Global max error across all ranks
local_max_err = np.max(np.abs(true_sol - u_interior))
global_max_err = cart_comm.allreduce(local_max_err, op=MPI.MAX)

if comm.rank == 0:
    print(f"Residual: {resiudal:.6e}")
    print(f"Global Max Error: {global_max_err:.6e}")

    fig, ax = plt.subplots(1, 2)
    img1 = ax[0].imshow(u_interior[:, :, nz // 2])
    img2 = ax[1].imshow(true_sol[:, :, nz // 2])

    ax[0].set_title("FDM")
    ax[1].set_title("True solution")
    fig.colorbar(img1, ax=ax[0])
    fig.colorbar(img2, ax=ax[1])
    fig.tight_layout()
    plt.show()

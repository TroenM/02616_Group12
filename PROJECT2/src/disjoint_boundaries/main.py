import json
import os
from time import perf_counter

import numpy as np
# Need to import njit before mpi to avoid segfaults


from disjoint_boundaries.arguments import get_args # isort: skip
from disjoint_boundaries.functions import jacobi_iteration, FunctionHandler # isort: skip
from disjoint_boundaries.mpi_helper import (
    create_comm_datatypes,
) # isort: skip
from mpi4py import MPI  # isort: skip

# Read arguments
t_setup = -perf_counter()
args = get_args()
NUM_DECOMPS_PER_AXIS = args.dim
Nx, Ny, Nz = args.nodes
MAXIT = args.maxit
TOL = args.tol

# Define cartesian communicator
comm = MPI.COMM_WORLD
dims = MPI.Compute_dims(comm.Get_size(), NUM_DECOMPS_PER_AXIS)
cart_comm = comm.Create_cart(dims, periods=[False] * 3, reorder=True)
rank = cart_comm.rank
P = cart_comm.Get_size()
coords = cart_comm.Get_coords(rank)

if rank == 0:
    print(f"{MAXIT=}, {TOL=}, {dims=}, {Nx=}")

# Link communications across ranks
neighbors = {}
for axis in range(3):
    backwards, forwards = cart_comm.Shift(axis, 1)
    neighbors[axis] = {"forward": forwards, "backward": backwards}

functions_handler = FunctionHandler((Nx, Ny, Nz), dims, coords)
nx, ny, nz = functions_handler.get_local_n()
detlasq_rhs = functions_handler.get_deltasq_rhs().astype(np.float32)

# Define the communication datatypes
datatypes = create_comm_datatypes(nx, ny, nz, MPI.FLOAT)

# Define solution buffers with halo regions
u_curr = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float32)
u_new = np.zeros_like(u_curr)

# Warm-up JIT with the exact argument types used in the loop
jacobi_iteration(u_curr, u_new, detlasq_rhs, (1, 2), (1, 2), (1, 2))

# Main loop
recv_reqs = []
send_reqs = []
residual = -1
t_setup += perf_counter()
t_compute = 0.0
t_comm_p2p = 0.0
t_comm_coll = 0.0
cart_comm.Barrier()          # all ranks start the clock together
t_total = -perf_counter()

n_iter = 0
for iter in range(MAXIT):
    n_iter += 1
    # Post boundary send/recv requests
    recv_reqs = []
    send_reqs = []
    t_comm_p2p -= perf_counter()
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
    t_comm_p2p += perf_counter()

    # Compute interior points
    res_sq = 0
    t_compute -= perf_counter()
    res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (2, nx), (2, ny), (2, nz))
    t_compute += perf_counter()

    # TODO: Consider whether we need to wait for all recv_reqs to finish before fitting boundaries
    t_comm_p2p -= perf_counter()
    MPI.Request.Waitall(recv_reqs)
    t_comm_p2p += perf_counter()

    # X faces (full j and k)
    t_compute -= perf_counter()
    res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (1, 2), (1, ny + 1), (1, nz + 1))
    if nx > 1:
        res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (nx, nx + 1), (1, ny + 1), (1, nz + 1))

    # Y faces (interior i, full k)
    res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (2, nx), (1, 2), (1, nz + 1))
    if ny > 1:
        res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (2, nx), (ny, ny + 1), (1, nz + 1))

    # Z faces (interior i and j)
    res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (2, nx), (2, ny), (1, 2))
    if nz > 1:
        res_sq += jacobi_iteration(u_curr, u_new, detlasq_rhs, (2, nx), (2, ny), (nz, nz + 1))

    t_compute += perf_counter()

    # Ensure neighbors have recieved boundaries before overwriting
    t_comm_p2p -= perf_counter()
    MPI.Request.Waitall(send_reqs)
    t_comm_p2p += perf_counter()

    # Flip solution pointers
    u_curr, u_new = u_new, u_curr

    # Collect residuals
    t_comm_coll -= perf_counter()
    global_res_sq = cart_comm.allreduce(res_sq, op=MPI.SUM)
    t_comm_coll += perf_counter()
    residual = np.sqrt(global_res_sq)
    if residual < TOL:
        break

t_total += perf_counter()
t_other = t_total - (t_compute + t_comm_p2p + t_comm_coll)

###########################################
#### Computation done packaging output ####
###########################################

# Run local test of solution
true_sol = functions_handler.get_sol()
u_interior = u_curr[1:-1, 1:-1, 1:-1]

# Halo traffic per iteration (only real neighbors)
face_elems = (ny * nz, nx * nz, nx * ny)
msgs_per_iter = 0
bytes_per_iter = 0
for axis in range(3):
    for direction in ("forward", "backward"):
        if neighbors[axis][direction] != MPI.PROC_NULL:
            msgs_per_iter += 1
            bytes_per_iter += face_elems[axis] * u_curr.itemsize

FLOPS_PER_POINT = 9
MEM_OPS_PER_POINT = 11

bytes_sent_local = bytes_per_iter * n_iter
n_msgs_local = msgs_per_iter * n_iter
work_local = nx * ny * nz
flops_local = work_local * n_iter * FLOPS_PER_POINT
mem_ops_local = work_local * n_iter * MEM_OPS_PER_POINT

# Global error_Linf and u_norm (RMS of u)
local_max_err = float(np.max(np.abs(true_sol - u_interior)))
error_Linf = cart_comm.allreduce(local_max_err, op=MPI.MAX)

# TODO: Ask Simon why we want to look at the norm of the solution?
local_u_sq = float(np.sum(u_interior**2))
global_u_sq = cart_comm.allreduce(local_u_sq, op=MPI.SUM)
u_norm = np.sqrt(global_u_sq / (Nx * Ny * Nz))

# Gather per-rank arrays to rank 0
local_timers = np.array(
    [t_total, t_compute, t_comm_p2p, t_comm_coll, t_other], dtype=np.float64
)
timers_all = cart_comm.gather(local_timers, root=0)
t_setup_all = cart_comm.gather(t_setup, root=0)
work_all = cart_comm.gather(work_local, root=0)
bytes_sent_all = cart_comm.gather(bytes_sent_local, root=0)
n_msgs_all = cart_comm.gather(n_msgs_local, root=0)
flops_all = cart_comm.gather(flops_local, root=0)
mem_ops_all = cart_comm.gather(mem_ops_local, root=0)
coords_all = cart_comm.gather(coords, root=0)

# Clean up
for axis_dts in datatypes.values():
    for dt in axis_dts.values():
        dt.Free()
cart_comm.Free()

# Save everything
if rank == 0:
    mapping = getattr(args, "mapping", "default")
    out_dir = os.path.join(args.outdir, args.tag, args.impl, mapping)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"N{Nx}_P{P}_rep{args.rep}.npz")

    meta = json.dumps(
        {
            "impl": args.impl,
            "mapping": mapping,
            "tag": args.tag,
            "dims": list(dims),
            "nodes": [Nx, Ny, Nz],
        }
    )

    np.savez(
        out_path,
        timers=np.array(timers_all, dtype=np.float64),
        timer_names=np.array(["total", "compute", "comm_p2p", "comm_coll", "other"]),
        t_setup=np.array(t_setup_all, dtype=np.float64),
        work=np.array(work_all, dtype=np.int64),
        bytes_sent=np.array(bytes_sent_all, dtype=np.int64),
        n_msgs=np.array(n_msgs_all, dtype=np.int64),
        flops=np.array(flops_all, dtype=np.int64),
        mem_ops=np.array(mem_ops_all, dtype=np.int64),
        coords=np.array(coords_all, dtype=np.int64),
        N=np.int64(Nx),
        h=np.float64(2 / (Nx + 1)),
        P=np.int64(P),
        n_iter=np.int64(n_iter),
        rep=np.int64(args.rep),
        residual=np.float64(residual),
        error_Linf=np.float64(error_Linf),
        u_norm=np.float64(u_norm),
        meta=meta,
    )
    print(f"Saved {out_path}")
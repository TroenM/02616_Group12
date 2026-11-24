"""

example run: mpirun -np 12 python3.12 poisson_mpi.py 100 2 3 2

Each iteration (pseudo code)
  1. halo_exchange(u)                  -> ghosts hold the neighbours' current values
  2. u_new = jacobi_iteration_block()  -> update this rank's block
  3. every `check_every` iterations: local sum (u_new - u)^2 over owned points,
     allreduce -> global residual, identical on every rank -> same stop decision
  4. u = u_new
"""
import json
import os
import sys
from time import perf_counter

import numpy as np
from mpi4py import MPI

import functions as func
from blocks import CartDecomposition

IMPL = "jacobi_cart"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "sequential")
check_every = 10


if __name__ == "__main__":
    N = int(sys.argv[1])
    maxit = int(sys.argv[2])
    tol = float(sys.argv[3])
    tag = sys.argv[4]
    rep = int(sys.argv[5])
    dims = tuple(int(a) for a in sys.argv[6:9]) if len(sys.argv) >= 9 else (0, 0, 0)

    comm = MPI.COMM_WORLD
    x = np.linspace(-1, 1, N)
    delta = x[1] - x[0]

    # setup
    t0 = perf_counter()
    dec = CartDecomposition(comm, N, dims)
    t_setup = perf_counter() - t0

    # halo traffic per iteration (one float64 plane per existing neighbour)
    halo_bytes, halo_msgs = 0, 0
    for axis in range(3):
        plane = int(np.prod(dec.shape)) // dec.shape[axis]
        for nb in dec.neighbours[axis]:
            if nb != MPI.PROC_NULL:
                halo_bytes += 8 * plane
                halo_msgs += 1
    work = int(np.prod([e - s for s, e in dec.owned]))

    # solve
    u = dec.allocate()
    comm.Barrier()
    t0 = perf_counter()
    u, iters, residual, t = func.jacobi_mpi(dec, u, delta, maxit, tol, check_every)
    t_total = perf_counter() - t0
    t_other = t_total - t["compute"] - t["comm_p2p"] - t["comm_coll"]

    # collect on rank 0 (untimed)
    row = [t_total, t["compute"], t["comm_p2p"], t["comm_coll"], t_other]
    local = (row, t_setup, work, halo_bytes * iters, halo_msgs * iters, list(dec.coords))
    allr = comm.gather(local, root=0)
    hosts = comm.gather(MPI.Get_processor_name(), root=0)
    u_approx = dec.gather(u, root=0)

    if dec.rank == 0:
        P = comm.Get_size()
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        e = u_approx - func.sol(X, Y, Z)
        out = dict(
            timers=np.array([a[0] for a in allr]),
            timer_names=np.array(["total", "compute", "comm_p2p", "comm_coll", "other"]),
            t_setup=np.array([a[1] for a in allr]),
            work=np.array([a[2] for a in allr], dtype=np.int64),
            bytes_sent=np.array([a[3] for a in allr], dtype=np.int64),
            n_msgs=np.array([a[4] for a in allr], dtype=np.int64),
            coords=np.array([a[5] for a in allr], dtype=np.int64),
            N=N, h=delta, P=P, n_iter=iters, rep=rep,
            residual=residual,
            error_Linf=float(np.max(np.abs(e))),
            u_norm=float(np.sqrt(np.mean(u_approx * u_approx))),
            meta=json.dumps(dict(impl=IMPL, tag=tag, dims=list(dec.dims),
                                 nodes=len(set(hosts)))),
        )
        d = os.path.join(RESULTS, tag, IMPL)
        os.makedirs(d, exist_ok=True)
        fname = os.path.join(d, f"N{N}_P{P}_rep{rep}.npz")
        np.savez(fname, **out)
        print(f"N={N} P={P} iters={iters} T={max(a[0][0] for a in allr):.3f}s "
              f"Linf_err={out['error_Linf']:.3e} -> {fname}")

    dec.free()

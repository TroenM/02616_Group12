"""

example run: mpirun -np 12 python3.12 poisson_mpi.py 100 2 3 2

Each iteration (pseudo code)
  1. halo_exchange(u)                  -> ghosts hold the neighbours' current values
  2. u_new = jacobi_iteration_block()  -> update this rank's block
  3. every `check_every` iterations: local sum (u_new - u)^2 over owned points,
     allreduce -> global residual, identical on every rank -> same stop decision
  4. u = u_new
"""
import sys
from time import perf_counter

import numpy as np
from mpi4py import MPI

import functions as func
from blocks import CartDecomposition


if __name__ == "__main__":
    N = int(sys.argv[1])
    maxit = 30 #int(sys.argv[2])
    print("Max iterations", maxit)
    tol = 1e-4 #float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    print("Tolerance", tol)
    check_every = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    dims = tuple(int(a) for a in sys.argv[5:8]) if len(sys.argv) >= 8 else (0, 0, 0)

    x = np.linspace(-1, 1, N)
    delta = x[1] - x[0]

    dec = CartDecomposition(MPI.COMM_WORLD, N, dims)
    if dec.rank == 0:
        print(f"ranks={dec.cart.Get_size()} dims={tuple(dec.dims)} N={N} delta={delta:.4f}")

    u = dec.allocate() 
    u, iters, residual = func.jacobi_mpi(dec, u, delta, maxit, tol, check_every)
    u_approx = dec.gather(u, root=0)

    if dec.rank == 0:
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        u_true = func.sol(X, Y, Z)
        err = np.max(np.abs(u_approx - u_true))
        print(f"iterations: {iters}, residual: {residual:.4e}, max error vs true: {err:.4e}")
        np.save("u_approx.npy", u_approx)
        func.plot_poisson_solution(N, X, Y, Z, u_true, u_approx, "jacobi")

    dec.free()
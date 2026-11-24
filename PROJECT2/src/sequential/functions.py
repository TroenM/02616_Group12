import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from blocks import CartDecomposition
from mpi4py import MPI

def sol(x,y,z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

def rhs(x, y, z):
    return 3 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

def jacobi_iteration(u: np.ndarray, delta: float, offset = [0, 0, 0]) -> np.ndarray:
    dims = u.shape
    u_new = u.copy()
    residual = 0

    delta_squared = delta ** 2
    ox, oy, oz = offset
    
    for i in range(1, dims[0] -1):
        for j in range(1, dims[1] - 1):
            for k in range(1, dims[2] - 1):
                u_new[i,j,k] = 1/6 * (u[i-1, j, k] + u[i+1, j, k] 
                                  + u[i, j-1, k] + u[i, j+1, k]
                                  + u[i, j, k-1] + u[i, j, k+1]
                                  + delta_squared * rhs(-1 + (ox + i)*delta, -1 + (oy + j)*delta, -1 + (oz + k)*delta))
                residual += (u[i,j,k] - u_new[i,j,k])**2

    return u_new, residual



def jacobi_mpi(dec: CartDecomposition, u: np.ndarray, delta: float,
               maxit: int = 100, tol: float = 1e-4, check_every: int = 10):

    cart = dec.cart
    offset = [s for s, _ in dec.local]
    owned = tuple(slice(os - ls, oe - ls)
                  for (os, oe), (ls, _) in zip(dec.owned, dec.local))

    residual = np.inf
    time_start = perf_counter()

    for it in range(maxit):
        dec.halo_exchange(u, mode = "non-blocking")
        u_new, local_sq = jacobi_iteration(u, delta, offset)

        if it % check_every == 0 or it == maxit - 1:
            residual = np.sqrt(cart.allreduce(local_sq, op=MPI.SUM))
            if dec.rank == 0:
                print(f"iter: {it}/{maxit}, residual: {residual:.4e}, "
                      f"time spent: {perf_counter() - time_start:.2f}", end="\r")
            if residual <= tol:  # same value on all ranks -> all stop together
                u = u_new
                break

        u = u_new

    if dec.rank == 0:
        print(f"\nFinal time: {perf_counter() - time_start:.2f}")
    return u, it + 1, residual


def plot_poisson_solution(N, X, Y, Z, u_true, u_approx, name):
    fig, ax = plt.subplots()

    fig, ax = plt.subplots(2, 1)

    approx = ax[0].contourf(X[:, :, N//2], Y[:, :, N//2], u_approx[:, :, N//2], levels = 50)
    fig.colorbar(approx)
    ax[0].set_title("Approx")
    true_sol = ax[1].contourf(X[:, :, N//2], Y[:, :, N//2], u_true[:, :, N//2], levels = 50)
    fig.colorbar(true_sol)
    ax[1].set_title("True")
    fig.tight_layout()
    fig.savefig(f"slice_{name}.png")

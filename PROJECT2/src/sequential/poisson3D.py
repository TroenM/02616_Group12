import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from time import perf_counter

@jit
def rhs(x, y, z):
    return 3 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

@jit
def sol(x,y,z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)

@jit
def jacobi_iteration(u: np.ndarray, delta: float) -> np.ndarray:
    dims = u.shape
    res = u.copy()

    delta_squared = delta ** 2
    
    for i in range(1, dims[0] -1):
        for j in range(1, dims[1] - 1):
            for k in range(1, dims[2] - 1):
                res[i,j,k] = 1/6 * (u[i-1, j, k] + u[i+1, j, k] 
                                  + u[i, j-1, k] + u[i, j+1, k]
                                  + u[i, j, k-1] + u[i, j, k+1]
                                  + delta_squared * rhs(-1 + i*delta, -1 * j*delta, -1 * k*delta))

    return res

def jacobi(u: np.ndarray, delta: float, maxit: int = 100, tol: float = 1e-4) -> np.ndarray:

    u_new = u.copy()
    residual = np.inf

    time_start = perf_counter()
    for iter in range(maxit):
        u_new = jacobi_iteration(u, delta)

        if (iter % 10) == 0 or iter == maxit:
            residual = np.linalg.norm(u_new - u)
            intermediate_time = perf_counter() - time_start
            print(f"iter: {iter}/{maxit}, residaul: {residual:.4f}, time spend: {intermediate_time:.2f}", end = "\r")
            if residual <= tol:
                break

        u = u_new.copy()

    final_time = perf_counter() - time_start
    print(f"\nFinal time: {final_time:.2f}")

    return u


if __name__ == "__main__":
    N = 100
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    z = np.linspace(-1, 1, N)

    delta = x[1] - x[0]
    print(f"delta: {delta}")

    u = np.zeros((N,N,N))


    u = jacobi(u, delta, maxit=10000)

    fig, ax = plt.subplots()


    X, Y, Z = np.meshgrid(x, y, z, indexing = "ij")
    f = sol(X, Y, Z)

    fig, ax = plt.subplots(2, 1)

    approx = ax[0].contourf(X[:, :, N//2], Y[:, :, N//2], u[:, :, N//2], levels = 50)
    fig.colorbar(approx)
    ax[0].set_title("Approx")
    true_sol = ax[1].contourf(X[:, :, N//2], Y[:, :, N//2], f[:, :, N//2], levels = 50)
    fig.colorbar(true_sol)
    ax[1].set_title("True")
    fig.tight_layout()
    fig.savefig("../results/sequential/slice.png")

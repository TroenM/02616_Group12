import numpy as np
from numba import jit


@jit
def sol(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


@jit
def rhs(x, y, z):
    return 3 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


@jit
def jacobi_iteration(u: np.ndarray, delta: float) -> np.ndarray:
    dims = u.shape
    u_new = u.copy()
    residual = 0

    delta_squared = delta**2
    one_over_six = 1 / 6  # ruff is formatting weird, so this is defined here

    for i in range(1, dims[0] - 1):
        for j in range(1, dims[1] - 1):
            for k in range(1, dims[2] - 1):
                u_new[i, j, k] = one_over_six * (
                    u[i - 1, j, k]
                    + u[i + 1, j, k]
                    + u[i, j - 1, k]
                    + u[i, j + 1, k]
                    + u[i, j, k - 1]
                    + u[i, j, k + 1]
                    + delta_squared
                    * rhs(-1 + i * delta, -1 + j * delta, -1 + k * delta)
                )
                residual += (u[i, j, k] - u_new[i, j, k]) ** 2

    return u_new, residual

import numpy as np
from numba import jit


@jit
def sol(x, y, z) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


@jit
def rhs(x, y, z) -> float:
    return 3 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


@jit
def jacobi_range(
    u_curr: np.ndarray,
    u_new: np.ndarray,
    i_range: tuple[int, int],
    j_range: tuple[int, int],
    k_range: tuple[int, int],
    delta: tuple[float, float, float],
    phys_origin: tuple[float, float, float],
) -> float:
    residual_sq = 0.0
    dx, dy, dz = delta
    x0, y0, z0 = phys_origin
    delta_squared = (dx * dy * dz) ** (2.0 / 3.0)
    one_over_six = 1.0 / 6.0  # Ruf formats this weirdly, so it is precomputed here

    for i in range(i_range[0], i_range[1]):
        x = x0 + i * dx
        for j in range(j_range[0], j_range[1]):
            y = y0 + j * dy
            for k in range(k_range[0], k_range[1]):
                z = z0 + k * dz
                u_new[i, j, k] = one_over_six * (
                    u_curr[i - 1, j, k]
                    + u_curr[i + 1, j, k]
                    + u_curr[i, j - 1, k]
                    + u_curr[i, j + 1, k]
                    + u_curr[i, j, k - 1]
                    + u_curr[i, j, k + 1]
                    + delta_squared * rhs(x, y, z)
                )
                diff = u_curr[i, j, k] - u_new[i, j, k]
                residual_sq += diff * diff

    return residual_sq

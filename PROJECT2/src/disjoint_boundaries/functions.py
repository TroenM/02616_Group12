import numpy as np
from numba import njit


@njit
def sol(x, y, z) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


@njit
def _rhs(x, y, z) -> float:
    return 3 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


@njit
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
    one_over_six = 1.0 / 6.0  # Ruff formats this weirdly, so it is precomputed here

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
                    + delta_squared * _rhs(x, y, z)
                )
                diff = u_curr[i, j, k] - u_new[i, j, k]
                residual_sq += diff * diff

    return residual_sq


@njit
def jacobi_iteration(
    u_curr: np.ndarray,
    u_new: np.ndarray,
    deltasq_rhs: np.ndarray,
    i_range: tuple[int, int],
    j_range: tuple[int, int],
    k_range: tuple[int, int],
):
    residual_sq = 0.0 
    one_over_six = 1.0 / 6.0  # Ruff formats this weirdly, so it is precomputed here

    for i in range(*i_range):
        for j in range(*j_range):
            for k in range(*k_range):
                u_new[i, j, k] = one_over_six * (
                    u_curr[i - 1, j, k]
                    + u_curr[i + 1, j, k]
                    + u_curr[i, j - 1, k]
                    + u_curr[i, j + 1, k]
                    + u_curr[i, j, k - 1]
                    + u_curr[i, j, k + 1]
                    + deltasq_rhs[i-1,j-1,k-1]
                )
                diff = u_curr[i, j, k] - u_new[i, j, k]
                residual_sq += diff * diff

    return residual_sq


class FunctionHandler:
    def __init__(self, N: tuple[int, int, int], dims: tuple[int, int, int], coords: tuple[int, int, int]):
        self.Nx = N[0]
        self.Ny = N[1]
        self.Nz = N[2]

        self.dimx = dims[0]
        self.dimy = dims[1]
        self.dimz = dims[2]

        self.coordx = coords[0]
        self.coordy = coords[1]
        self.coordz = coords[2]

        self.nx, self.x_start, self.x_end = self._get_local_extent(self.Nx, self.dimx, self.coordx)
        self.ny, self.y_start, self.y_end = self._get_local_extent(self.Ny, self.dimy, self.coordy)
        self.nz, self.z_start, self.z_end = self._get_local_extent(self.Nz, self.dimz, self.coordz)

    def get_local_n(self) -> tuple[int, int, int]:
        return self.nx, self.ny, self.nz

    def get_deltasq_rhs(self) -> np.ndarray:
        # Define derived constants
        deltax, deltay, deltaz = (
            2 / (self.Nx + 1),
            2 / (self.Ny + 1),
            2 / (self.Nz + 1),
        )  # Assuming domain is [-1, 1] in all directions

        # Get the physical orign of the rank
        phys_origin = (-1 + self.x_start * deltax, -1 + self.y_start * deltay, -1 + self.z_start * deltaz)

        local_x = phys_origin[0] + np.arange(1, self.nx + 1) * deltax
        local_y = phys_origin[1] + np.arange(1, self.ny + 1) * deltay
        local_z = phys_origin[2] + np.arange(1, self.nz + 1) * deltaz
        X, Y, Z = np.meshgrid(local_x, local_y, local_z, indexing="ij")

        delta_sq = (deltax * deltay * deltaz)**(2/3)
        deltasq_rhs = delta_sq *_rhs(X, Y, Z)

        return deltasq_rhs

    def get_sol(self) -> np.ndarray:
        # Define derived constants
        deltax, deltay, deltaz = (
            2 / (self.Nx + 1),
            2 / (self.Ny + 1),
            2 / (self.Nz + 1),
        )  # Assuming domain is [-1, 1] in all directions

        # Get the physical orign of the rank
        phys_origin = (-1 + self.x_start * deltax, -1 + self.y_start * deltay, -1 + self.z_start * deltaz)

        local_x = phys_origin[0] + np.arange(1, self.nx + 1) * deltax
        local_y = phys_origin[1] + np.arange(1, self.ny + 1) * deltay
        local_z = phys_origin[2] + np.arange(1, self.nz + 1) * deltaz
        X, Y, Z = np.meshgrid(local_x, local_y, local_z, indexing="ij")

        return sol(X, Y, Z)


    @staticmethod
    def _get_local_extent(global_size, dim_procs, coord):
        base_size = global_size // dim_procs
        remainder = global_size % dim_procs

        local_size = base_size + 1 if coord < remainder else base_size

        # Calculate the global starting index for this rank
        global_start = (coord * base_size) + min(coord, remainder)
        global_end = global_start + local_size

        return local_size, global_start, global_end
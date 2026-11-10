import numpy as np
import matplotlib.pyplot as plt
from numba import jit

LOGLEVEL = 1

@jit
def point_xyz(i,j,k, delta):
    
    return boundary[0, 0] + i*delta, boundary[1,0] + j*delta, boundary[2,0] + k*delta

@jit
def jacobi_iteration(domain, rhs, deltas, delta_sq):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N-1):
                domain[i,j,k] = 1/6 * ( 
                    domain[i -1, j, k] + domain[i + 1, j, k] +\
                    domain[i, j-1, k] + domain[i, j+1, k] +\
                    domain[i, j, k-1] + domain[i, j, k+1] +\
                    delta_sq * rhs(*point_xyz(i,j,k, deltas))
                )

def jacobi(domain, rhs, max_iter: int = 1000, tol = 1e-4, true_sol = None):
    for _ in range(max_iter):
        if LOGLEVEL > 0:
            print(f"Computing iteration {_}", end="\r")
        jacobi_iteration(domain, rhs, deltas, delta_sq)
        d = np.max(np.abs(true_sol - domain)) if true_sol is not None else np.inf
        if d < tol: break

if __name__ == '__main__':
    import sys
    # Parameters for Poisson problem
    @jit
    def f(x,y,z):
        return 3 * np.pi ** 2 *np.sin(np.pi * x) * np.sin(np.pi*y) * np.sin(np.pi * z)

    @jit
    def u_true(x,y,z):
        return np.sin(np.pi * x) * np.sin(np.pi*y) * np.sin(np.pi * z)
    
    boundary = np.array([
        [-1,1],
        [-1,1],
        [-1,1]
    ])

    # Discretization
    
    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 100
    max_iterations = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000

    deltas = np.diff(boundary, axis=1)[0,0] / N
    delta_sq = deltas ** 2

    domain = np.zeros((N,N,N))

    x,y,z = np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N)
    X,Y,Z = np.meshgrid(x,y,z)

    sol = u_true(X,Y,Z)

    jacobi(domain, f, max_iter = max_iterations, true_sol = sol)

    # Compare the computed and exact solutions on the central z-slice.
    mid = N // 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, values, title in zip(
        axes,
        (domain[:, :, mid], sol[:, :, mid]),
        ('Jacobi solution', 'Exact solution'),
    ):
        image = ax.imshow(values, extent=(-1, 1, -1, 1), origin='lower')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(image, ax=ax)
    plt.savefig("../results/sequential.png")

    



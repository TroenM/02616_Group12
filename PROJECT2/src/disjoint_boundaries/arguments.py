import argparse


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def dim_type(strings):
    if len(strings.split(",")) == 1:
        return int(strings)
    else:
        return tuple_type(strings)


def n_type(strings):
    if len(strings.split(",")) == 1:
        return [int(strings)] * 3
    else:
        return tuple_type(strings)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="disjoint_boundaries/main",
        description="Ranks compute interior points while waiting for boundary conditions",
    )

    parser.add_argument(
        "-d",
        "--dim",
        help="The number of decompositions in each dimension (x, y, z), or number of dimensions: passed as string '(x,y,z)' or int N",
        type=dim_type,
        default=3,
    )

    parser.add_argument(
        "-N",
        "--nodes",
        help="Tuple '(Nx, Ny, Nz)', or int N -> Nx = Ny = Nz = N",
        type=n_type,
        default=[100] * 3,
    )

    parser.add_argument(
        "--maxit", help="Max number of jacobi interations", type=int, default=100
    )

    parser.add_argument(
        "--tol",
        help="Residual tolerence for converged solution",
        type=float,
        default=1e-4,
    )

    parser.add_argument("--tag", type=str, default="scaling")
    parser.add_argument("--impl", type=str, default="disjoint_boundaries")
    parser.add_argument("--mapping", type=str, default="default")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="results")

    args = parser.parse_args()
    return args

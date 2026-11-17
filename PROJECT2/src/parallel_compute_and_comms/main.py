from mpi4py import MPI
import argparse
from parallel_compute_and_comms.functions import sol, rhs, jacobi_iteration

comm = MPI.COMM_WORLD

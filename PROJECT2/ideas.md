1. Check rank residuals (computed lhs vs know rhs), when a tolerence is achieved skip computation and broadcast boundaries. Probably only interesting if we have more work than workers.

2. Pointwise residual computation

3. Threading with jit. Bind threads rank, with ranks*threads = nprocs
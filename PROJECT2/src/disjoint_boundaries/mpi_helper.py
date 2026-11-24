from mpi4py import MPI

def create_comm_datatypes(
    nx: int, ny: int, nz: int, mpi_base_type: MPI.Datatype = MPI.DOUBLE
) -> dict[int, dict[str, MPI.Datatype]]:
    """
    Generates datatypes[axis][direction_action] = MPI.Datatype
    """
    full_size = (nx + 2, ny + 2, nz + 2)
    interior_dims = (nx, ny, nz)
    datatypes = {0: {}, 1: {}, 2: {}}

    # The logical progression along an axis from index 0 to N+1
    dt_keys = [
        "backward_recv",  # Inject into left halo (index 0)
        "backward_send",  # Extract left interior (index 1)
        "forward_send",  # Extract right interior (index N)
        "forward_recv",  # Inject into right halo (index N+1)
    ]

    for axis in range(3):
        sub_size = list(interior_dims)
        curr_axis_size = sub_size[axis]
        sub_size[axis] = 1

        # Following the order of dt_keys
        start_offsets = [0, 1, curr_axis_size, curr_axis_size + 1]

        for start_offset, dt_key in zip(start_offsets, dt_keys):
            start = [1, 1, 1]
            start[axis] = start_offset

            dt = mpi_base_type.Create_subarray(
                sizes=full_size, subsizes=sub_size, starts=start
            )

            dt.Commit()
            datatypes[axis][dt_key] = dt

    return datatypes

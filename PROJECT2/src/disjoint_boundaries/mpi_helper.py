from mpi4py import MPI


def get_local_extent(global_size, dim_procs, coord):
    base_size = global_size // dim_procs
    remainder = global_size % dim_procs

    local_size = base_size + 1 if coord < remainder else base_size

    # Calculate the global starting index for this rank
    global_start = (coord * base_size) + min(coord, remainder)
    global_end = global_start + local_size

    return local_size, global_start, global_end


def __create_halo_subarray(
    mpi_base_type: MPI.Datatype,
    full_size: tuple[int, int, int],
    sub_size: tuple[int, int, int],
    starts: tuple[int, int, int],
) -> MPI.Datatype:
    """
    Creates a halo subarray for a given direction
    used as a helper for generating send/recv datatypes
    """
    return mpi_base_type.Create_subarray(full_size, sub_size, starts)


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

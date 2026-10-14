import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter, sleep

"""
ABOUT THE IMPLEMENTATION:

This version takes input 
    number of chunks per axis
    number of ranks

Then divides the work out to each ranks
if the number of chunks is not divisible by number of ranks, then it divides the work out on the other ranks starting from 1.

"""


_help = f"""\
{sys.argv[0]} [chunks-per-axis] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with 4x4 = 16 chunks
$ mpirun -n 4 python {sys.argv[0]} 4

Call it with 8x8 chunks and image size of 100 by 500 pixels
$ mpirun -n 4 python {sys.argv[0]} 8 100x500
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

if size_mpi < 2:
    print("Need at least 2 ranks (rank 0 only distributes work)")
    sys.exit(1)

# Defaults
chunk_per_axis = 4
n_chunks = chunk_per_axis**2
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Arguments
argv = sys.argv[1:]
if argv:
    chunk_per_axis = int(argv.pop(0))
    n_chunks = chunk_per_axis**2
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

if rank == 0:
    print(f"""\
Calculating the Mandelbrot set with these arguments:
{n_chunks = }
{size = }
{xlim = }
{ylim = }
{size_mpi = }
""")

# Convert to numpy arrays
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

n_x = size[0]
n_y = size[1]

# Same on every rank, so sender and receiver agree on buffer size
n_workers = size_mpi - 1
n_even_distributed = n_chunks // n_workers
n_mismatch = n_chunks % n_workers
workload = n_even_distributed + 1  # rows per message (last row may be zero padding)

# TODO: Reconcile Christians setup ^ with the constructed boiler plate

# TODO: Remove hard coded test values
NUM_GRIDS = 50
nx, ny = 100, 100

# Tag convertion:
#   0: Info data [glb_idx, start_x, end_x, start_y, end_y]
#   1: Computed data [local_image.flatten(), time_data]
if rank == 0:
    # -------- Master rank ----------
    image = np.zeros(size)
    rank_times = np.ones((NUM_GRIDS, 6)) * (-1)
    master_send_time = 0
    master_wait_time = 0

    # TODO: use Chistians grid map to distribute (glb_idx, start_x, end_x, start_y, end_y)
    tasks = [
        np.array([i, 0, nx, 0, ny], dtype=int) for i in range(NUM_GRIDS)
    ]  # Dummy data

    # Stuff for keeping track of how far we are
    tasks_sent = 0
    tasks_completed = 0

    # Queue used for remembering what we sent to who
    worker_queue = {w: [] for w in range(1, size_mpi)}
    worker_task_counts = {w: 0 for w in range(1, size_mpi)}

    # Dedicated contiguous receive buffer for each worker (0 is a dummy for rank 0)
    recv_buffers = [np.zeros((nx * ny + 6), dtype=float) for _ in range(size_mpi)]
    recv_reqs = [MPI.REQUEST_NULL] * size_mpi

    # Send initial 2 grids to all workers
    for worker in range(1, size_mpi):
        for _ in range(2):
            if tasks_sent < NUM_GRIDS:
                task = tasks[tasks_sent]
                master_send_time -= perf_counter()
                comm.Send(
                    task, dest=worker, tag=0
                )  # Use blocking send (small package and the recieve is ready)
                master_send_time += perf_counter()
                worker_queue[worker].append(task)
                tasks_sent += 1

        # Post the first non-blocking recieve for this worker
        if len(worker_queue[worker]) > 0:
            recv_reqs[worker] = comm.Irecv(recv_buffers[worker], source=worker, tag=1)

    # Dynamic scheduling loop
    while tasks_completed < NUM_GRIDS:
        # Wait for any worker to finish
        master_wait_time -= perf_counter()
        worker = MPI.Request.Waitany(recv_reqs)
        master_wait_time += perf_counter()

        glb_idx, start_x, end_x, start_y, end_y = worker_queue[worker].pop(0)

        # Unpack the recieved image
        flat_data = recv_buffers[worker]
        image_data = flat_data[:-6].reshape((nx, ny))
        time_data = flat_data[-6:]

        # Insert the data into the result structures
        x_len, y_len = end_x - start_x, end_y - start_y
        image[start_x:end_x, start_y:end_y] = image_data[:x_len, :y_len]
        rank_times[glb_idx] = time_data

        tasks_completed += 1
        worker_task_counts[worker] += 1

        # Send the next job to the worker
        if tasks_sent < NUM_GRIDS:
            task = tasks[tasks_sent]
            master_send_time -= perf_counter()
            comm.Send(task, dest=worker, tag=0)
            master_send_time += perf_counter()
            worker_queue[worker].append(task)
            tasks_sent += 1

        # Repost recieve to the worker
        if tasks_completed < NUM_GRIDS and len(worker_queue[worker]) > 0:
            recv_reqs[worker] = comm.Irecv(recv_buffers[worker], source=worker, tag=1)

    shutdown_task = np.array([-1, 0, 0, 0, 0], dtype=int)
    for worker in range(1, size_mpi):
        master_send_time -= perf_counter()
        comm.Send(shutdown_task, dest=worker, tag=0)
        master_send_time += perf_counter()

    print("\n--- SCHEDULING RESULTS ---")
    for w, count in worker_task_counts.items():
        print(f"Worker {w} completed {count} tasks.")

    time_summary = rank_times.sum(axis=0)
    print(f"""
    Master send time = {master_send_time:.4f}
    Master wait time = {master_wait_time:.4f}
    working idle time = {time_summary[0]:.4f}
    Computation time = {time_summary[1]:.4f}
          """)

    MPI.Finalize()


else:
    # -------- Working Ranks -----------
    # Prepare buffers
    NUM_BUFFERS = 3

    # Create three dimensions (each is an iwmage)
    image_buffer = np.zeros((NUM_BUFFERS, nx, ny), dtype=float)
    info_buffer = np.zeros((NUM_BUFFERS, 5), dtype=int)

    # Initialize requests
    send_reqs = [MPI.REQUEST_NULL] * NUM_BUFFERS
    recv_reqs = [MPI.REQUEST_NULL] * NUM_BUFFERS

    # Set the rank buffer index to shift between buffers
    buffer_idx = 0

    # Prepost the buffers
    for i in range(NUM_BUFFERS):
        recv_reqs[i] = comm.Irecv(info_buffer[i], source=0, tag=0)

    time_rank_idle = 0
    while True:
        # Wait for the info_buffer to be ready
        time_rank_idle -= perf_counter()
        recv_reqs[buffer_idx].Wait()
        time_rank_idle += perf_counter()

        # Make rank 1 artificially slow
        if rank == 1:
            sleep(0.1)

        # Load the necessary info for computation
        glb_idx, start_x, end_x, start_y, end_y = info_buffer[buffer_idx]

        # Catch the shutdown signal
        if glb_idx < 0:
            break

        # Ensure previous use of this image buffer is done
        time_rank_idle -= perf_counter()
        send_reqs[buffer_idx].Wait()
        time_rank_idle += perf_counter()

        # Post next recieve request, so it can be done while computing
        recv_reqs[buffer_idx] = comm.Irecv(info_buffer[buffer_idx], source=0, tag=0)

        # Setup pointer directly into the current slice of the image buffer
        current_image = image_buffer[buffer_idx]
        current_image.fill(0)

        # Rank computation
        time_rank_computation = -perf_counter()
        for x in range(start_x, end_x):
            cx = complex(xlim[0] + x * xconst, 0)

            for y in range(start_y, end_y):
                c = cx + complex(0, ylim[0] + y * yconst)
                z = 0

                for i in range(100):
                    z = z * z + c

                    if np.abs(z) > 2:
                        current_image[x - start_x, y - start_y] = i
                        break
        time_rank_computation += perf_counter()

        time_package = np.array(
            [time_rank_idle, time_rank_computation, start_x, end_x, start_y, end_y]
        )

        # Send the image back to rank 0
        data_bundle = np.append(current_image.flatten(), time_package)
        send_reqs[buffer_idx] = comm.Isend(data_bundle, dest=0, tag=1)

        # Move buffer to next slice
        buffer_idx = (buffer_idx + 1) % NUM_BUFFERS

    # Shutdown hanging requests
    MPI.Request.Waitall(send_reqs)

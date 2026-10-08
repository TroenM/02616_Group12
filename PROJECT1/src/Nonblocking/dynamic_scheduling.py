import sys
import numpy as np
from mpi4py import MPI
from time import perf_counter

_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with a chunk-size of 10
$ mpirun -n 4 python {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ mpirun -n 4 python {sys.argv[0]} 10 100x500
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

# Defaults
chunk_size = 10
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

if rank == 0:
    print(f"""\
    Calculating the Mandelbrot set with these arguments:

    {chunk_size = }
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


# TODO: Remove hard coded test values
NUM_GRIDS = 10
nx, ny = 5, 5

# Tag convertion:
#   0: Info data [glb_idx, start_x, end_x, start_y, end_y]
#   1: Sub image
#   2: Timing
if rank == 0:
    # -------- Master rank ----------
    image = np.zeros(size)
    rank_times = np.ones((NUM_GRIDS, 6)) * (-1)

    # TODO: use Chistians grid map to distribute (glb_idx, start_x, end_x, start_y, end_y)
    tasks = [
        np.array([i, 0, nx, 0, ny], dtype=int) for i in range(NUM_GRIDS)
    ]  # Dummy data

    # Stuff for keeping track of how far we are
    tasks_sent = 0
    tasks_completed = 0

    master_send_reqs = []

    # Queue used for remembering what we sent to who
    worker_queue = {w: [] for w in range(1, size_mpi)}

    # Dedicated contiguous receive buffer for each worker (0 is a dummy for rank 0)
    recv_buffers = [np.zeros((nx, ny), dtype=float) for _ in range(size_mpi)]
    recv_reqs = [MPI.REQUEST_NULL] * size_mpi

    # Send initial 2 grids to all workers
    for worker in range(1, size_mpi):
        for _ in range(2):
            if tasks_sent < NUM_GRIDS:
                task = tasks[tasks_sent]
                req = comm.Isend(task, dest=worker, tag=0)
                master_send_reqs.append(req)
                worker_queue[worker].append(task)
                tasks_sent += 1

        # Post the first non-blocking recieve for this worker
        if len(worker_queue[worker]) > 0:
            recv_reqs[worker] = comm.Irecv(recv_buffers[worker], source=worker, tag=1)

    # Dynamic scheduling loop
    while tasks_completed < NUM_GRIDS:
        # Wait for any worker to finish
        worker = MPI.Request.Waitany(recv_reqs)

        glb_idx, start_x, end_x, start_y, end_y = worker_queue[worker].pop(0)

        # Unpack the recieved image
        x_len, y_len = end_x - start_x, end_y - start_y
        image[start_x:end_x, start_y:end_y] = recv_buffers[worker][:x_len, :y_len]
        tasks_completed += 1

        comm.Recv(rank_times[glb_idx], source=worker, tag=2)

        # Send the next job to the worker
        if tasks_sent < NUM_GRIDS:
            task = tasks[tasks_sent]
            req = comm.Isend(task, dest=worker, tag=0)
            master_send_reqs.append(req)
            worker_queue[worker].append(task)
            tasks_sent += 1

        # Repost recieve to the worker
        if tasks_completed < NUM_GRIDS and len(worker_queue[worker]) > 0:
            recv_reqs[worker] = comm.Irecv(recv_buffers[worker], source=worker, tag=1)

    shutdown_task = np.array([-1, 0, 0, 0, 0], dtype=int)
    for worker in range(1, size_mpi):
        req = comm.Isend(shutdown_task, dest=worker, tag=0)
        master_send_reqs.append(req)

    # Make sure everyone recieves the shutdown task
    MPI.Request.Waitall(master_send_reqs)


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

        # Send the image back to rank 0
        send_reqs[buffer_idx] = comm.Isend(current_image, dest=0, tag=1)

        time_package = np.array(
            [time_rank_idle, time_rank_computation, start_x, end_x, start_y, end_y]
        )
        comm.Send(time_package, dest=0, tag=2)

        # Move buffer to next slice
        buffer_idx = (buffer_idx + 1) % NUM_BUFFERS

    # Shutdown hanging requests
    MPI.Request.Waitall(send_reqs)

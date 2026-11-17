
import numpy as np
from mpi4py import MPI


def split_axis(N: int, nb: int) -> list[tuple[int, int]]:
    q, r = divmod(N, nb) #get quotient and remainder
    sizes = [q + (1 if b < r else 0) for b in range(nb)] #divide axis into nb slices and distribute the remainder
    starts = np.concatenate(([0], np.cumsum(sizes))) #create coords for that axis
    return [(int(starts[b]), int(starts[b + 1])) for b in range(nb)] #create contiguous ranges [start, end)


class CartDecomposition:
    def __init__(self, comm, N: int, dims=(0, 0, 0)):

        self.N = N
        self.dims = MPI.Compute_dims(comm.Get_size(), list(dims))
        self.cart = comm.Create_cart(self.dims, periods=[False] * 3, reorder=True) #create 3x3 dimensions grid
        self.rank = self.cart.Get_rank()
        self.coords = self.cart.Get_coords(self.rank) #get the ranks position in the grid

        # (lower, upper) neighbour rank per axis
        # MPI.PROC_NULL at the domain boundary
        self.neighbours = [self.cart.Shift(axis, 1) for axis in range(3)]

        self.owned, self.local = self.ranges_of(self.coords) # get the owned = local owned and local = owned incl. ghost
        self.shape = tuple(e - s for s, e in self.local)

        self._make_datatypes()

    def ranges_of(self, coords):
        owned, local = [], []
        for a in range(3):
            s, e = split_axis(self.N, self.dims[a])[coords[a]]
            owned.append((s, e))
            local.append((
                s - 1 if coords[a] > 0 else s, 
                e + 1 if coords[a] < self.dims[a] - 1 else e
            ))
        return owned, local

    def allocate(self) -> np.ndarray:
        return np.zeros(self.shape)


    def _subarray(self, subsizes, starts):
        return MPI.DOUBLE.Create_subarray(list(self.shape), list(subsizes), list(starts)).Commit()

    def _plane(self, axis, idx):
        sub = list(self.shape); sub[axis] = 1
        starts = [0, 0, 0]; starts[axis] = idx
        return self._subarray(sub, starts)

    def _make_datatypes(self):

        self.face_types = []
        for axis in range(3):
            lo, hi = self.neighbours[axis]
            n = self.shape[axis]
            has_lo, has_hi = lo != MPI.PROC_NULL, hi != MPI.PROC_NULL
            self.face_types.append({
                "send_lo": self._plane(axis, 1) if has_lo else None,       # first owned plane
                "recv_lo": self._plane(axis, 0) if has_lo else None,       # lower ghost plane
                "send_hi": self._plane(axis, n - 2) if has_hi else None,   # last owned plane
                "recv_hi": self._plane(axis, n - 1) if has_hi else None,   # upper ghost plane
            })

        # Owned region inside the local array (used for gathering)
        self.owned_type = self._subarray(
            [e - s for s, e in self.owned],
            [os - ls for (os, _), (ls, _) in zip(self.owned, self.local)],
        )

    @staticmethod #funktionen bruger ikke et self argument, så static er tilføjet
    def _buf(u, dtype):
        # empty message when no neighbour on that side
        return [u, 1, dtype] if dtype is not None else [u, 0, MPI.DOUBLE]

    def free(self): #CL
        for types in self.face_types:
            for t in types.values():
                if t is not None:
                    t.Free()
        self.owned_type.Free()

    def halo_exchange(self, u: np.ndarray) -> None:

        #Fill ghos face of u from neighbors.
        # We use two Sendrecv pr axis
        for axis in range(3):
            lo, hi = self.neighbours[axis]
            t = self.face_types[axis]
            # shift up, last owned plane -> upper neighbour, lower ghost <- lower neighbour
            self.cart.Sendrecv(
                self._buf(u, t["send_hi"]), 
                dest=hi, 
                sendtag=axis, 
                recvbuf=self._buf(u, t["recv_lo"]), 
                source=lo, 
                recvtag=axis)
            # shift down. first owned plane -> lower neighbour, upper ghost <- upper neighbour
            self.cart.Sendrecv(
                self._buf(u, t["send_lo"]), 
                dest=lo, 
                sendtag=axis,
                recvbuf=self._buf(u, t["recv_hi"]), 
                source=hi, 
                recvtag=axis
                )

    def gather(self, u: np.ndarray, root: int = 0):
        """Stitch the owned parts of every rank into one N^3 array on `root`.

        CL on the below part:
        Each rank sends its owned region straight from u via the subarray type
        (no copy). Blocks can have different sizes, so root receives into one
        flat buffer with Gatherv and then places each piece in the global array.
        """
        if self.N > 200:
            print("You should only gather images of size <= 200. Aborting program")
            exit(0)

        all_owned = [self.ranges_of(self.cart.Get_coords(r))[0]
                     for r in range(self.cart.Get_size())]
        counts = [int(np.prod([e - s for s, e in owned])) for owned in all_owned]

        if self.rank == root:
            flat = np.empty(sum(counts))
            displs = np.concatenate(([0], np.cumsum(counts)[:-1])).tolist()
            recvbuf = [flat, counts, displs, MPI.DOUBLE]
        else:
            recvbuf = None

        self.cart.Gatherv([u, 1, self.owned_type], recvbuf, root=root)

        if self.rank != root:
            return None

        u_global = np.empty((self.N,) * 3)
        offset = 0
        for owned, count in zip(all_owned, counts):
            shape = tuple(e - s for s, e in owned)
            u_global[tuple(slice(s, e) for s, e in owned)] = \
                flat[offset:offset + count].reshape(shape)
            offset += count
        return u_global
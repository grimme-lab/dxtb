import torch

from .typing import Tensor

DTYPE = torch.int64


class AdjacencyList:
    """
    Sparse neighbour map in compressed sparse row format.

    A symmetric neighbour map given in dense format like

          | 1 | 2 | 3 | 4 | 5 | 6
       ---|---|---|---|---|---|---
        1 |   | x |   | x | x |
        2 | x |   | x |   | x | x
        3 |   | x |   | x |   | x
        4 | x |   | x |   | x | x
        5 | x | x |   | x |   |
        6 |   | x | x | x |   |

    is stored in two compressed array identifying the neighbouring atom `nlat`
    and its cell index `nltr`. Two index arrays `inl` for the offset
    and `nnl` for the number of entries map the atomic index to the row index.

       inl   =  0,       3,          7,      10,         14,      17, 20
       nnl   =  |  2 ->  |  3 ->     |  2 ->  |  3 ->     |  2 ->  |  |
       nlat  =     2, 4, 5, 1, 3, 5, 6, 2, 4, 6, 1, 3, 5, 6, 1, 2, 4, 2, 3, 4
       nltr  =     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

    An alternative representation would be to store just the offsets in `inl`
    and additional beyond the last element the total number of neighbors.
    However, the indexing is from inl(i) to inl(i+1)-1 could be confusing,
    therefore two arrays are used for clarity.
    """

    inl: torch.Tensor
    """Offset index in the neighbour map"""

    nnl: torch.Tensor
    """Number of neighbours for each atom"""

    nlat: torch.Tensor
    """Index of the neighbouring atom"""

    nltr: torch.Tensor
    """Cell index of the neighbouring atom"""

    def __init__(self, numbers: Tensor, positions: Tensor, cutoff: float) -> None:
        l = torch.numel(numbers)
        self.inl = torch.zeros(l, dtype=DTYPE)
        self.nnl = torch.zeros(l, dtype=DTYPE)

        tmp_nlat = []

        img = 0
        cutoff2 = cutoff * cutoff
        eps = torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype)

        for iat in range(l):
            self.inl[iat] = img

            for jat in range(iat + 1):
                vec = torch.sub(positions[iat, :], positions[jat, :])
                r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
                if r2 < eps or r2 > cutoff2:
                    continue

                tmp_nlat.append(jat)
                img += 1

            self.nnl[iat] = img - self.inl[iat]

        self.nlat = torch.tensor(tmp_nlat)

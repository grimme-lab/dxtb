from numpy import finfo
import torch

from .constants import INT16 as DTYPE
from .exlibs.tbmalt import Geometry


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

    trans: torch.Tensor
    """Generated lattice points (from :func:`~xtbml.cutoff.get_lattice_points`)"""

    inl: torch.Tensor
    """Offset index in the neighbour map"""

    nnl: torch.Tensor
    """Number of neighbours for each atom"""

    nlat: torch.Tensor
    """Index of the neighbouring atom"""

    nltr: torch.Tensor
    """Cell index of the neighbouring atom"""

    def __init__(self, mol: Geometry, trans: torch.Tensor, cutoff: float) -> None:
        self.trans = trans
        self.inl = torch.zeros(mol.get_length(), dtype=DTYPE)
        self.nnl = torch.zeros(mol.get_length(), dtype=DTYPE)

        tmp_nlat = torch.zeros(30 * mol.get_length(), dtype=DTYPE)
        tmp_nltr = torch.zeros(30 * mol.get_length(), dtype=DTYPE)

        img = 0
        cutoff2 = cutoff * cutoff

        for iat in range(mol.get_length()):
            self.inl[iat] = img

            for jat in range(iat + 1):
                for itr in range(trans.size(dim=1)):
                    vec = torch.sub(mol.positions[iat, :], mol.positions[jat, :])
                    vec = torch.sub(vec, trans[itr, :])
                    r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
                    if r2 < finfo(type(cutoff2)).eps or r2 > cutoff2:
                        continue

                    tmp_nlat[img] = jat
                    tmp_nltr[img] = itr
                    img += 1

            self.nnl[iat] = img - self.inl[iat]

        self.nlat = tmp_nlat[:img]
        self.nltr = tmp_nltr[:img]

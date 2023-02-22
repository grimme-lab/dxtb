"""
The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch

from .._types import Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..integral import mmd
from ..param import Param, get_elem_angular
from ..utils import batch, symmetrize, t2int


class Overlap(TensorLike):
    """
    Overlap from atomic orbitals.
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    unique: Tensor
    """Unique species of the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp

    def build(self, positions: Tensor) -> Tensor:
        """Overlap calculation of unique shells pairs.

        Returns
        -------
        Tensor
            Overlap matrix.
        """

        def get_overlap(bas: Basis, positions: Tensor, ihelp: IndexHelper) -> Tensor:
            """Overlap calculation for a single molecule.

            Parameters
            ----------
            numbers : Tensor
                Unique atomic numbers of whole batch.
            positions : Tensor
                Positions of single molecule.

            Returns
            -------
            Tensor
                Overlap matrix for single molecule.
            """

            umap, n_unique_pairs = bas.unique_shell_pairs(ihelp)
            alphas, coeffs = bas.create_cgtos()

            # spread stuff to orbitals for indexing
            alpha = batch.index(
                batch.index(batch.pack(alphas), ihelp.shells_to_ushell),
                ihelp.orbitals_to_shell,
            )
            coeff = batch.index(
                batch.index(batch.pack(coeffs), ihelp.shells_to_ushell),
                ihelp.orbitals_to_shell,
            )
            positions = batch.index(
                batch.index(positions, ihelp.shells_to_atom),
                ihelp.orbitals_to_shell,
            )
            ang = ihelp.spread_shell_to_orbital(ihelp.angular)

            # overlap calculation
            ovlp = torch.zeros(*umap.shape, dtype=self.dtype, device=self.device)
            for uval in range(n_unique_pairs):
                pairs = self.get_pairs(umap, uval)
                first_pair = pairs[0]

                angi, angj = ang[first_pair]
                norbi = 2 * t2int(angi) + 1
                norbj = 2 * t2int(angj) + 1

                # collect [0, 0] entry of each subblock
                upairs = self.get_subblock_start(umap, uval, norbi, norbj)

                # we only require one pair as all have the same basis function
                alpha_tuple = (
                    batch.deflate(alpha[first_pair][0]),
                    batch.deflate(alpha[first_pair][1]),
                )
                coeff_tuple = (
                    batch.deflate(coeff[first_pair][0]),
                    batch.deflate(coeff[first_pair][1]),
                )
                ang_tuple = (angi, angj)

                vec = positions[upairs][:, 0, :] - positions[upairs][:, 1, :]
                stmp = mmd.overlap(ang_tuple, alpha_tuple, coeff_tuple, -vec)

                # write overlap of unique pair to correct position in full overlap matrix
                for r, pair in enumerate(upairs):
                    ovlp[
                        pair[0] : pair[0] + norbi,
                        pair[1] : pair[1] + norbj,
                    ] = stmp[r]

            return ovlp.fill_diagonal_(1.0)

        if self.numbers.ndim > 1:
            o = []
            for _batch in range(self.numbers.shape[0]):
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `get_overlap`
                nums = batch.deflate(self.numbers[_batch])
                ihelp = IndexHelper.from_numbers(
                    nums, get_elem_angular(self.par.element)
                )

                bas = Basis(
                    torch.unique(nums),
                    self.par,
                    ihelp.unique_angular,
                    dtype=self.dtype,
                    device=self.device,
                )

                o.append(get_overlap(bas, positions[_batch], ihelp))

            overlap = batch.pack(o)
        else:
            bas = Basis(
                self.unique,
                self.par,
                self.ihelp.unique_angular,
                dtype=self.dtype,
                device=self.device,
            )
            overlap = get_overlap(bas, positions, self.ihelp)

        # force symmetry to avoid problems through numerical errors
        return symmetrize(overlap)

    def get_pairs(self, x: Tensor, i: int) -> Tensor:
        """Get indices of all unqiue shells pairs with index value `i`.

        Parameters
        ----------
        x : Tensor
            Matrix of unique shell pairs.
        i : int
            Value representing all unique shells in the matrix.

        Returns
        -------
        Tensor
            Indices of all unique shells pairs with index value `i` in the matrix.
        """

        return (x == i).nonzero(as_tuple=False)

    def get_subblock_start(
        self, umap: Tensor, i: int, norbi: int, norbj: int
    ) -> Tensor:
        """
        Filter out the top-left index of each subblock of unique shell pairs. This makes use of the fact that the pairs are sorted along
        the rows.

        Example: A "s" and "p" orbital would give the following 4x4 matrix
        of unique shell pairs:
        1 2 2 2
        3 4 4 4
        3 4 4 4
        3 4 4 4
        As the overlap routine gives back tensors of the shape `(norbi, norbj)`,
        i.e. 1x1, 1x3, 3x1 and 3x3 here, we require only the following four
        indices from the matrix of unique shell pairs: [0, 0] (1x1), [1, 0]
        (3x1), [0, 1] (1x3) and [1, 1] (3x3).


        Parameters
        ----------
        pairs : Tensor
            Indices of all unique shell pairs of one type (n, 2).
        norbi : int
            Number of orbitals per shell.
        norbj : int
            Number of orbitals per shell.

        Returns
        -------
        Tensor
            Top-left (i.e. [0, 0]) index of each subblock.
        """

        # no need to filter out a 1x1 block
        if norbi == 1 and norbj == 1:
            return self.get_pairs(umap, i)

        # sorting along rows allows only selecting every `norbj`th pair
        if norbi == 1:
            pairs = self.get_pairs(umap, i)
            return pairs[::norbj]

        if norbj == 1:
            pairs = self.get_pairs(umap.mT, i)

            # do the same for the transposed pairs, but switch columns
            return torch.index_select(pairs[::norbi], 1, torch.tensor([1, 0]))

        # more intricate because we can have variation in two dimensions
        pairs = self.get_pairs(umap, i)

        # remove every `norbj`th pair as before; only second dim is tricky
        pairs = pairs[::norbj]

        start = 0
        rest = pairs

        # init with dummy
        final = torch.tensor([[-1, -1]])

        while True:
            # get number of blocks in a row by counting the number of
            # same indices in the first dimension
            nb = (pairs[:, 0] == pairs[start, 0]).nonzero().flatten().size(0)

            # we need to skip the amount of rows in the block
            skip = nb * norbi

            # split for the blocks in each row because there can be different numbers of blocks in a row
            target, rest = torch.split(rest, [skip, rest.size(-2) - skip])

            # select only the top left index of each block
            final = torch.cat((final, target[:nb]), 0)

            start += skip
            if start >= pairs.size(-2):
                break

        # remove dummy
        return final[1:]

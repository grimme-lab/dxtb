"""
Overlap
=======

The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch

from .._types import Literal, Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..constants import defaults
from ..param import Param, get_elem_angular
from ..utils import batch, symmetrize, t2int
from .mmd import overlap_gto, overlap_gto_grad


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

    uplo: Literal["n", "u", "l"] = "l"
    """
    Whether the matrix of unique shell pairs should be create as a
    triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
    Defaults to `l` (lower triangular matrix).
    """

    cutoff: Tensor | float | int | None = 50.0
    """
    Real-space cutoff for integral calculation. Defaults to
    `constans.defaults.INTCUTOFF`.
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp
        self.cutoff = cutoff

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

    def build(self, positions: Tensor) -> Tensor:
        """
        Overlap calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates.

        Returns
        -------
        Tensor
            Overlap matrix.
        """
        if self.numbers.ndim > 1:
            o = []
            for _batch in range(self.numbers.shape[0]):
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `calc_overlap`
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

                o.append(self.calc_overlap(bas, positions[_batch], ihelp))

            overlap = batch.pack(o)
        else:
            bas = Basis(
                self.unique,
                self.par,
                self.ihelp.unique_angular,
                dtype=self.dtype,
                device=self.device,
            )
            overlap = self.calc_overlap(bas, positions, self.ihelp)

        # force symmetry to avoid problems through numerical errors
        if self.uplo == "n":
            return symmetrize(overlap)
        return overlap

    def calc_overlap(self, bas: Basis, positions: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Overlap calculation for a single molecule.

        Parameters
        ----------
        numbers : Tensor
            Unique atomic numbers of whole batch.
        positions : Tensor
            Positions of single molecule.
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Tensor
            Overlap matrix for single molecule.
        """
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

        # real-space integral cutoff; assumes orthogonalization of basis
        # functions as "self-overlap" is  explicitly removed with `dist > 0`
        if self.cutoff is None:
            mask = None
        else:
            # cdist does not return zero for distance between same vectors
            # https://github.com/pytorch/pytorch/issues/57690
            dist = torch.cdist(
                positions, positions, compute_mode="donot_use_mm_for_euclid_dist"
            )
            mask = (dist < self.cutoff) & (dist > 0.1)

        umap, n_unique_pairs = bas.unique_shell_pairs(ihelp, mask=mask, uplo=self.uplo)

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
            stmp = overlap_gto(ang_tuple, alpha_tuple, coeff_tuple, -vec)

            # write overlap of unique pair to correct position in full overlap matrix
            for r, pair in enumerate(upairs):
                ovlp[
                    pair[0] : pair[0] + norbi,
                    pair[1] : pair[1] + norbj,
                ] = stmp[r]

        # fill empty triangular matrix
        if self.uplo == "l":
            ovlp = torch.tril(ovlp, diagonal=-1) + torch.triu(ovlp.mT)
        elif self.uplo == "u":
            ovlp = torch.triu(ovlp, diagonal=1) + torch.tril(ovlp.mT)

        # fix diagonal as "self-overlap" was removed via mask earlier
        return ovlp.fill_diagonal_(1.0)

    def calc_overlap_grad(
        self, bas: Basis, positions: Tensor, ihelp: IndexHelper
    ) -> tuple[Tensor, Tensor]:
        """
        Overlap gradient dS/dr for a single molecule.

        Parameters
        ----------
        bas : Basis
            Basis set for calculation.
        positions : Tensor
            Positions of single molecule.
        ihelp : IndexHelper
            Index helper for orbital mapping.

        Returns
        -------
        tuple[Tensor, Tensor]
            Overlap and gradient of overlap for single molecule.
        """

        umap, n_unique_pairs = bas.unique_shell_pairs(ihelp, uplo="n")
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

        # overlap tensors
        ovlp = torch.zeros(*umap.shape, dtype=self.dtype, device=self.device)
        grad = torch.zeros((3, *umap.shape), dtype=self.dtype, device=self.device)

        # loop over unique pairs
        for uval in range(n_unique_pairs):
            pairs = self.get_pairs(umap, uval)
            first_pair = pairs[0]

            li, lj = ang[first_pair]
            norbi = 2 * t2int(li) + 1
            norbj = 2 * t2int(lj) + 1

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
            ang_tuple = (li, lj)

            vec = positions[upairs][:, 0, :] - positions[upairs][:, 1, :]

            stmp, dstmp = overlap_gto_grad(ang_tuple, alpha_tuple, coeff_tuple, -vec)
            # [upairs, norbi, norbj], [upairs, 3, norbi, norbj]

            # write overlap of unique pair to correct position in full overlap matrix
            for r, pair in enumerate(upairs):
                ovlp[
                    pair[0] : pair[0] + norbi,
                    pair[1] : pair[1] + norbj,
                ] = stmp[r]
                grad[
                    :,
                    pair[0] : pair[0] + norbi,
                    pair[1] : pair[1] + norbj,
                ] = dstmp[r]

        return symmetrize(ovlp), grad  # [norb, norb], [3, norb, norb]

    def get_pairs(self, x: Tensor, i: int) -> Tensor:
        """
        Get indices of all unqiue shells pairs with index value `i`.

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
        Filter out the top-left index of each subblock of unique shell pairs.
        This makes use of the fact that the pairs are sorted along the rows.

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

        # the remaining cases, i.e., if no s-orbitals are involved, are more
        # intricate because we can have variation in two dimensions...

        # If only a triangular matrix is considered, we need to take special
        # care of the diagonal because the blocks on the diagonal are cut off,
        # which leads to missing pairs for the `while` loop. Whether this is
        # the case for the unique index `i`, is checked by the trace of the
        # unique map: The trace will be zero if there are no blocks on the
        # diagonal. If there are blocks on the diagonal, we complete the
        # missing triangular matrix. This includes all unique indices `i` of
        # the unique map, and hence, introduces some redundancy for blocks that
        # are not on the diagonal.
        if self.uplo != "n":
            if (torch.where(umap == i, umap, umap.new_tensor(0))).trace() > 0.0:
                umap = torch.where(umap == -1, umap.mT, umap)

        pairs = self.get_pairs(umap, i)

        # remove every `norbj`th pair as before; only second dim is tricky
        pairs = pairs[::norbj]

        start = 0
        rest = pairs

        # init with dummy
        final = torch.tensor([[-1, -1]])

        while True:
            # get number of blocks in a row by counting the number of same
            # indices in the first dimension
            nb = (pairs[:, 0] == pairs[start, 0]).nonzero().flatten().size(0)

            # we need to skip the amount of rows in the block
            skip = nb * norbi

            # split for the blocks in each row because there can be different
            # numbers of blocks in a row
            target, rest = torch.split(rest, [skip, rest.size(-2) - skip])

            # select only the top left index of each block
            final = torch.cat((final, target[:nb]), 0)

            start += skip
            if start >= pairs.size(-2):
                break

        # remove dummy
        return final[1:]

    def get_gradient(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Gradient calculation for McMurchie-Davidson overlap.

        Parameters
        ----------
        positions : Tensor
            Positions of given molecule.

        Returns
        -------
        tuple[Tensor, Tensor]
            Overlap and gradient of overlap for single molecule.
        """

        if self.numbers.ndim > 1:
            _ovlps, _grads = [], []
            for _batch in range(self.numbers.shape[0]):
                # setup individual `IndexHelper` and `Basis`
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

                ds_i, dsdr_i = self.calc_overlap_grad(bas, positions[_batch], ihelp)
                _ovlps.append(ds_i)
                _grads.append(torch.einsum("ijk ->kji", dsdr_i))  # [norb, norb, 3]

            s = batch.pack(_ovlps)
            dsdr = batch.pack(_grads)

        else:
            bas = Basis(
                self.unique,
                self.par,
                self.ihelp.unique_angular,
                dtype=self.dtype,
                device=self.device,
            )

            # obtain overlap gradient
            s, dsdr = self.calc_overlap_grad(bas, positions, self.ihelp)
            dsdr: Tensor = torch.einsum("ijk ->kji", dsdr)  # [norb, norb, 3]

        return s, dsdr

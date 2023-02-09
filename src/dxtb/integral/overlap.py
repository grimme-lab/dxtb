"""
The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch

from .._types import Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..integral import mmd
from ..param import Param, get_elem_angular
from ..utils import batch, symmetrize, t2int, IntegralTransformError
from .mmd import sqrtpi3, nlm_cart, e_function, EFunction
from . import transform
import sys
import os


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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

                o.append(self.get_overlap(bas, positions[_batch], ihelp))

            overlap = batch.pack(o)
        else:
            bas = Basis(
                self.unique,
                self.par,
                self.ihelp.unique_angular,
                dtype=self.dtype,
                device=self.device,
            )
            overlap = self.get_overlap(bas, positions, self.ihelp)

        # force symmetry to avoid problems through numerical errors
        return symmetrize(overlap)

    def get_overlap(self, bas: Basis, positions: Tensor, ihelp: IndexHelper) -> Tensor:
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

        return ovlp  # .fill_diagonal_(1.0)

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

    def get_overlap_grad(
        self, bas: Basis, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """Overlap gradient dS/dr for a single molecule.

        Args:
            bas (Basis): Basis set for calculation.
            positions (Tensor): Positions of single molecule.
            ihelp (IndexHelper): Index helper for orbital mapping.

        Returns:
            Tensor: Gradient of overlap for single molecule.
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

        # overlap tensor
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

            vecs = positions[upairs][:, 0, :] - positions[upairs][:, 1, :]

            # TODO: vectorize w.r.t. multiple vectors
            tmp_ = []
            for vec in vecs:
                tmp_.append(
                    mmd.get_ovlp_grad(ang_tuple, alpha_tuple, coeff_tuple, +vec)
                )

            dstmp = torch.stack(tmp_, dim=-4)  # [upairs, 3, norbi, norbj]

            # write overlap of unique pair to correct position in full overlap matrix
            for r, pair in enumerate(upairs):
                grad[
                    :,
                    pair[0] : pair[0] + norbi,
                    pair[1] : pair[1] + norbj,
                ] = dstmp[r]

        return grad  # [3, norb, norb]

    def get_overlap_grad_atomwise(
        self, bas: Basis, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """Overlap gradient dS/dr for a single molecule.

        Args:
            bas (Basis): Basis set for calculation.
            positions (Tensor): Positions of single molecule.
            ihelp (IndexHelper): Index helper for orbital mapping.

        Returns:
            Tensor: Gradient of overlap for single molecule.
        """
        natms = positions.shape[-2]
        gradient = torch.zeros([natms, 3], dtype=self.dtype, device=self.device)

        angular_per_atom = [
            ihelp.angular[ihelp.shells_to_atom == i]
            for i in torch.unique_consecutive(ihelp.shells_to_atom)
        ]

        # spread coefficients to shells for indexing
        alphas, coeffs = bas.create_cgtos()

        _alpha = batch.index(batch.pack(alphas), ihelp.shells_to_ushell)
        _coeff = batch.index(batch.pack(coeffs), ihelp.shells_to_ushell)

        # TODO: for vectorised and batched cases, other solutions required
        #       e.g. create a AtomOrderedBasis in basis.type.py with def from_basis(bas: Basis):
        def get_atom_sorted(x: Tensor):
            # Spread to atom sorted representation
            x_per_atom = [
                x[ihelp.shells_to_atom == i]
                for i in torch.unique_consecutive(ihelp.shells_to_atom)
            ]
            x_atm = []
            for atm in range(natms):
                lmax = max(angular_per_atom[atm])
                if lmax > 0:
                    # x_atm.append(x_per_atom[atm])  # append dimension / view
                    x_atm.append([[x_per_atom[atm][l]] for l in range(lmax + 1)])
                else:
                    x_atm.append([x_per_atom[atm]])
                # alternatively wrap nsh = [tensor, tensor] --> [x_per_atom[atm][i] for i in x_per_atom[atm]] and [[x_per_atom[atm][l]] ...]
            # x = [[x_per_atom[atm][l] for l in range(max(angular_per_atom[atm]))] if max(angular_per_atom[atm]) > 0 else x_per_atom[atm] for atm in range(natms)]

            return x_atm

        # map coefficients to atom-wise representation
        coeff = get_atom_sorted(_coeff)
        alpha = get_atom_sorted(_alpha)

        # loop atomwise
        for A in range(natms):
            for B in range(A + 1):
                if A == B:
                    continue

                vec = positions[..., A, :] - positions[..., B, :]

                for li in torch.unique(angular_per_atom[A]):
                    for lj in torch.unique(angular_per_atom[B]):

                        ang_tuple = (li, lj)

                        # number of shells of angular momentum li(lj) for atom A (B)
                        nshi = len(alpha[A][li])
                        nshj = len(alpha[B][lj])

                        # number of orbitals
                        norbi = 2 * t2int(li) + 1
                        norbj = 2 * t2int(lj) + 1

                        # TODO: directly loop over orbitals
                        for ish in range(nshi):
                            for jsh in range(nshj):
                                # obtain respective coefficients
                                alpha_tuple = (alpha[A][li][ish], alpha[B][lj][jsh])
                                coeff_tuple = (coeff[A][li][ish], coeff[B][lj][jsh])

                                stmp = mmd.get_ovlp_grad(
                                    ang_tuple, alpha_tuple, coeff_tuple, vec
                                )

                                # correct assignment of overlap components
                                for i in range(norbi):
                                    for j in range(norbj):
                                        gradient[A, :] += stmp[:, i, j]
                                        gradient[B, :] -= stmp[:, i, j]

        return gradient

    def get_gradient(self, positions: Tensor, atomwise: bool = False) -> Tensor:

        print("self.numbers.ndim > 1", self.numbers.ndim > 1)

        bas = Basis(
            self.unique,
            self.par,
            self.ihelp.unique_angular,
            dtype=self.dtype,
            device=self.device,
        )

        # obtain overlap gradient
        if atomwise:  # [natm, 3]
            dsdr = self.get_overlap_grad_atomwise(bas, positions, self.ihelp)
        else:  # [natm, natm, 3]
            dsdr = self.get_overlap_grad(bas, positions, self.ihelp)  # [3, norb, norb]

            # reduce to atom-resolved representation (since xyz-directions
            # are linear independent, apply to each dimension individually)
            # dsdr = torch.stack([self.ihelp.reduce_orbital_to_atom(x, dim=(-2, -1)) for x in dsdr], dim=-1)

        return dsdr

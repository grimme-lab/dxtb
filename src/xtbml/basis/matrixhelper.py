"""
Matrix helper utility to help dealing with matricies such as the Hamiltonian.
Allow to extract blocks specifying atomic contributions on an orbital-resolved
Hamiltonian.

Example
-------
>>> from ..basis import IndexHelper
>>> # anuglar momenta and respective orbitals for different elements
>>> angular, valence_shells = get_elem_param_shells(par.element, valence=True)
>>> # index helper
>>> ihelp = IndexHelper.from_numbers(sample.numbers, angular)
>>> # columns
>>> columns = MatrixHelper.get_orbital_columns(h0, ihelp)>>>
>>> # scalars
>>> scalars = MatrixHelper.get_orbital_sum(h0, ihelp)
>>> # NOTE: this induces a difference between similar atoms, e.g. the H in H20
>>> # block for atom i, j
>>> i, j = 0, 1
>>> block_ij = MatrixHelper.get_atomblock(h0, i, j, ihelp)
>>> # diagonal blocks
>>> diag_blocks = MatrixHelper.get_diagonal_blocks(h0, ihelp)
"""

from __future__ import annotations

import torch

from ..basis import IndexHelper
from ..param import Element
from ..param.gfn1 import GFN1_XTB as par
from ..typing import Tensor


# TODO: could be added to IndexHelper
def get_elem_param_shells(
    par_element: dict[str, Element], valence: bool = False
) -> tuple[dict, dict]:
    """
    Obtain angular momenta of the shells of all atoms.
    This returns the required input for the `IndexHelper`.
    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.
    Returns
    -------
    dict
        Angular momenta of all elements.
    """

    d = {}
    aqm2lsh = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

    if valence:
        v = {}

    for i, item in enumerate(par_element.values()):
        # convert shells: [ "1s", "2s" ] -> [ 0, 0 ]
        l = []
        for shell in getattr(item, "shells"):
            if shell[1] not in aqm2lsh:
                raise ValueError(f"Unknown shell type '{shell[1]}'.")

            l.append(aqm2lsh[shell[1]])

        d[i + 1] = l

        if valence:
            # https://stackoverflow.com/questions/62300404/how-can-i-zero-out-duplicate-values-in-each-row-of-a-pytorch-tensor

            r = torch.tensor(l, dtype=torch.long)
            tmp = torch.ones(r.shape, dtype=torch.bool)
            if r.size(0) < 2:
                v[i + 1] = tmp
                continue

            # sorting the rows so that duplicate values appear together
            # e.g. [1, 2, 3, 3, 3, 4, 4]
            y, idxs = torch.sort(r)

            # subtracting, so duplicate values will become 0
            # e.g. [1, 2, 3, 0, 0, 4, 0]
            tmp[1:] = (y[1:] - y[:-1]) != 0

            # retrieving the original indices of elements
            _, idxs = torch.sort(idxs)

            # re-organizing the rows following original order
            # e.g. [1, 2, 3, 4, 0, 0, 0]
            v[i + 1] = torch.gather(tmp, 0, idxs).tolist()

    if valence:
        return d, v


# TODO: could be added to IndexHelper
def get_orbitals_per_atom(ihelp: IndexHelper) -> Tensor:
    """Number of orbitals per atom

    Parameters
    ----------
    ihelp : IndexHelper
        IndexHelper for given basis

    Returns
    -------
    Tensor
        List-like tensor containing number of orbitals per atom
    """
    orbitals_per_atom = torch.scatter_reduce(
        ihelp.orbitals_per_shell,
        dim=-1,
        index=ihelp.shells_to_atom,
        reduce="sum",
    )
    return orbitals_per_atom


class MatrixHelper:
    """
    Matrix helper for dealing with e.g. Hamiltonian
    """

    def get_orbital_columns(x: Tensor, ihelp: IndexHelper, dim=1) -> list[Tensor]:
        """Retrieve atomwise orbital partitions by splitting hamiltonian
        row-wise or column-wise.

        Parameters
        ----------
        x : Tensor
            Tensor to be partionised
        ihelp : IndexHelper
            Index mapping for the basis set
        dim : int, optional
            Dimension along split is conducted, by default 0 (column-wise)
            leading to an output shape of [n_shells, n_orbitals] for each atom

        Returns
        -------
        list[Tensor]
            List of atomic partitions for orbitals
        """
        orbitals_per_atom = get_orbitals_per_atom(ihelp)
        return torch.split(x, orbitals_per_atom.tolist(), dim=dim)

    def get_orbital_sum(x: Tensor, ihelp: IndexHelper) -> Tensor:
        """Retrieve atomwise partitions by summing over hamiltonian row-wise.

        Parameters
        ----------
        x : Tensor
            Tensor to be partionised
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        list[Tensor]
            List of atomic partitions for orbitals
        """
        x = torch.sum(x, dim=1, keepdim=False)
        return MatrixHelper.get_orbital_columns(x, ihelp, dim=0)

    def select_block(x: Tensor, idx0: Tensor, idx1: Tensor) -> Tensor:
        """Generate block from tensor given by indices specified in two 1D index-tensors.

        Args:
            x (Tensor): 2D tensor from which block is extracted
            idx0 (Tensor): Row index vector along 1st dimension
            idx1 (Tensor): Column index vector along 2nd dimension

        Returns:
            Tensor: Block tensor of shape [len(idx0), len(idx1)]
        """
        slice0 = x.index_select(dim=0, index=idx0)
        return slice0.index_select(dim=1, index=idx1)

    def convert_slice_indices(slice_idx: Tensor, n: int) -> Tensor:
        """Convert indices to ranges for slicing.

        Args:
            slice_idx (Tensor): Slice indices in 1D tensor
            n (int): Specifiying the n-th slice

        Returns:
            Tensor: Indices for the n-th slice
        """
        if n == 0:
            return torch.LongTensor(range(slice_idx[n]))
        else:
            return torch.LongTensor(range(slice_idx[n - 1], slice_idx[n]))

    def get_atomblock(
        x: Tensor, atom_idx0: int, atom_idx1: int, ihelp: IndexHelper
    ) -> Tensor:
        """Get block for atom 0 and atom 1 contribution. Output shape is
        based on the total number of orbitals per atom.

        Parameters
        ----------
        x : Tensor
            Matrix to be sliced, e.g. hamiltonian
        atom_idx0 : int
            Atom index of first atom
        atom_idx1 : int
            Atom index of second atom
        ihelp : IndexHelper
            Index helper for given basis

        Returns
        -------
        Tensor
            Block containing contribution of atom 0 and atom 1.
            Shape: [n_orbitals0, n_orbitals1]
        """
        orbitals_per_atom = get_orbitals_per_atom(ihelp)
        # slicing based on number of orbitals
        slices = torch.cumsum(orbitals_per_atom, dim=0)
        # index vectors
        v0 = MatrixHelper.convert_slice_indices(slices, atom_idx0)
        v1 = MatrixHelper.convert_slice_indices(slices, atom_idx1)

        return MatrixHelper.select_block(x, v0, v1)

    def get_diagonal_blocks(x: Tensor, ihelp: IndexHelper) -> list[Tensor]:
        """Obtain all diagonal blocks, i.e. the atom self-contribution.

        Parameters
        ----------
        x : Tensor
            Matrix to be sliced
        ihelp : IndexHelper
            Index helper for given basis, specifiying atoms and orbitals

        Returns
        -------
        list[Tensor]
            List of diagonal blocks
        """

        n_atoms = len(ihelp.shells_per_atom)
        return [MatrixHelper.get_atomblock(x, i, i, ihelp) for i in range(n_atoms)]

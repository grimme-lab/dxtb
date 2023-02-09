"""
Index helper utility to create index maps between atomic, shell-resolved, and
orbital resolved representations of quantities.

Example
-------
>>> import torch
>>> from xtbml.basis import IndexHelper
>>> numbers = torch.tensor([6, 1, 1, 1, 1])
>>> angular = {1: [0], 6: [0, 1]}
>>> ihelp = IndexHelper.from_numbers(numbers, angular)
>>> torch.sum(ihelp.angular >= 0)
torch.tensor(6)
"""
from __future__ import annotations

import torch

from .._types import Any, NoReturn, Tensor
from ..utils import batch, wrap_gather, wrap_scatter_reduce

__all__ = ["IndexHelper"]


def _fill(index: Tensor, repeat: Tensor) -> Tensor:
    """
    Fill an index map using index offsets and number of repeats
    """
    index_map = index.new_zeros(int(torch.sum(repeat).item()))
    for idx, offset, count in zip(torch.arange(index.shape[-1]), index, repeat):
        index_map[offset : offset + count] = idx
    return index_map


def _expand(index: Tensor, repeat: Tensor) -> Tensor:
    """
    Expand an index map using index offsets and number of repeats
    """

    return index.new_tensor(
        [
            idx
            for offset, count in zip(index, repeat)
            for idx in torch.arange(offset.item(), (offset + count).item(), 1)
        ]
    )


class IndexHelper:
    """
    Index helper for basis set
    """

    unique_angular: Tensor
    """Angular momenta of all unique shells"""

    angular: Tensor
    """Angular momenta for all shells"""

    atom_to_unique: Tensor
    """Mapping of atoms to unique species"""

    ushells_to_unique: Tensor
    """Mapping of unique shells to unique species"""

    shells_to_ushell: Tensor
    """Mapping of shells to unique unique"""

    shells_per_atom: Tensor
    """Number of shells for each atom"""

    orbitals_per_shell: Tensor
    """Number of orbitals for each shell"""

    shell_index: Tensor
    """Offset index for starting the next shell block"""

    orbital_index: Tensor
    """Offset index for starting the next orbital block"""

    shells_to_atom: Tensor
    """Mapping of shells to atoms"""

    orbitals_to_shell: Tensor
    """Mapping of orbitals to shells"""

    def __init__(
        self,
        unique_angular: Tensor,
        angular: Tensor,
        atom_to_unique: Tensor,
        ushells_to_unique: Tensor,
        shells_to_ushell: Tensor,
        shells_per_atom: Tensor,
        shell_index: Tensor,
        shells_to_atom: Tensor,
        orbitals_per_shell: Tensor,
        orbital_index: Tensor,
        orbitals_to_shell: Tensor,
        device: torch.device | None = None,
    ):
        self.unique_angular = unique_angular
        self.angular = angular
        self.atom_to_unique = atom_to_unique
        self.ushells_to_unique = ushells_to_unique
        self.shells_to_ushell = shells_to_ushell
        self.shells_per_atom = shells_per_atom
        self.shell_index = shell_index
        self.shells_to_atom = shells_to_atom
        self.orbitals_per_shell = orbitals_per_shell
        self.orbital_index = orbital_index
        self.orbitals_to_shell = orbitals_to_shell

        self.__device = device
        self.__dtype = shells_per_atom.dtype

        if any(
            tensor.dtype != self.dtype
            for tensor in (
                self.unique_angular,
                self.angular,
                self.atom_to_unique,
                self.ushells_to_unique,
                self.shells_to_ushell,
                self.shells_per_atom,
                self.shell_index,
                self.shells_to_atom,
                self.orbitals_per_shell,
                self.orbital_index,
                self.orbitals_to_shell,
            )
        ):
            raise ValueError("All tensors must have same dtype")

        if any(
            tensor.device != self.device
            for tensor in (
                self.unique_angular,
                self.angular,
                self.atom_to_unique,
                self.ushells_to_unique,
                self.shells_to_ushell,
                self.shells_per_atom,
                self.shell_index,
                self.shells_to_atom,
                self.orbitals_per_shell,
                self.orbital_index,
                self.orbitals_to_shell,
            )
        ):
            raise ValueError("All tensors must be on the same device")

    @classmethod
    def from_numbers(
        cls, numbers: Tensor, angular: dict[int, list[int]]
    ) -> IndexHelper:
        """
        Construct an index helper instance from atomic numbers and their angular momenta.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for the system
        angular : dict[int, Tensor]
            Map between atomic numbers and angular momenta of all shells

        Returns
        -------
        IndexHelper
            Instance of index helper for given basis set
        """

        device = numbers.device
        batched = numbers.ndim > 1
        pad_val = -999

        unique, atom_to_unique = torch.unique(numbers, return_inverse=True)

        unique_angular = torch.tensor(
            [l for number in unique for l in angular.get(number.item(), [-1])],
            device=device,
        )

        # note that padding (i.e., when number = 0) is assigned one shell
        ushells_per_unique = torch.tensor(
            [len(angular.get(number.item(), [-1])) for number in unique],
            device=device,
        )

        ushell_index = torch.cumsum(ushells_per_unique, dim=-1) - ushells_per_unique
        ushells_to_unique = _fill(ushell_index, ushells_per_unique)

        if batched:
            # remove the single shell assigned to the padding value in order to
            # avoid an additional count in the expansion as this will cause
            # errors in certain situations
            # (see https://github.com/grimme-lab/xtbML/issues/67)
            if (unique == 0.0).any():
                ushells_per_unique[0] = 0

            shells_to_ushell = batch.pack(
                [
                    _expand(
                        ushell_index[atom_to_unique[_batch, :]],
                        ushells_per_unique[atom_to_unique[_batch, :]],
                    )
                    for _batch in range(numbers.shape[0])
                ],
                value=-1,
            )
        else:
            shells_to_ushell = _expand(
                ushell_index[atom_to_unique],
                ushells_per_unique[atom_to_unique],
            )

        shells_per_atom = ushells_per_unique[atom_to_unique]
        shell_index = torch.cumsum(shells_per_atom, -1) - shells_per_atom
        shell_index[shells_per_atom == 0] = pad_val

        if batched:
            shells_to_atom = batch.pack(
                [
                    _fill(shell_index[_batch, :], shells_per_atom[_batch, :])
                    for _batch in range(numbers.shape[0])
                ],
                value=pad_val,
            )
        else:
            shells_to_atom = _fill(shell_index, shells_per_atom)

        lsh = torch.where(
            shells_to_ushell >= 0,
            unique_angular[shells_to_ushell],
            pad_val,
        )

        orbitals_per_shell = torch.where(
            lsh >= 0, 2 * lsh + 1, torch.tensor(0, device=device)
        )

        orbital_index = torch.cumsum(orbitals_per_shell, -1) - orbitals_per_shell
        orbital_index[orbitals_per_shell == 0] = pad_val

        if batched:
            orbitals_to_shell = batch.pack(
                [
                    _fill(orbital_index[_batch, :], orbitals_per_shell[_batch, :])
                    for _batch in range(numbers.shape[0])
                ],
                value=pad_val,
            )
        else:
            orbitals_to_shell = _fill(orbital_index, orbitals_per_shell)

        return cls(
            unique_angular=unique_angular,
            angular=lsh,
            atom_to_unique=atom_to_unique,
            ushells_to_unique=ushells_to_unique,
            shells_to_ushell=shells_to_ushell,
            shells_per_atom=shells_per_atom,
            shell_index=shell_index,
            shells_to_atom=shells_to_atom,
            orbitals_per_shell=orbitals_per_shell,
            orbital_index=orbital_index,
            orbitals_to_shell=orbitals_to_shell,
            device=device,
        )

    def reduce_orbital_to_shell(
        self, x: Tensor, dim: int | tuple[int, int] = -1, reduce: str = "sum"
    ) -> Tensor:
        """
        Reduce orbital-resolved tensor to shell-resolved tensor

        Parameters
        ----------
        x : Tensor
            Orbital-resolved tensor
        dim : int | (int, int)
            Dimension to reduce over, defaults to -1
        reduce : str
            Reduction method, defaults to "sum"

        Returns
        -------
        Tensor
            Shell-resolved tensor
        """

        return wrap_scatter_reduce(x, dim, self.orbitals_to_shell, reduce)

    def reduce_shell_to_atom(
        self, x: Tensor, dim: int | tuple[int, int] = -1, reduce: str = "sum"
    ) -> Tensor:
        """
        Reduce shell-resolved tensor to atom-resolved tensor

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor
        dim : int | (int, int)
            Dimension to reduce over, defaults to -1
        reduce : str
            Reduction method, defaults to "sum"

        Returns
        -------
        Tensor
            Atom-resolved tensor
        """

        return wrap_scatter_reduce(x, dim, self.shells_to_atom, reduce)

    def reduce_orbital_to_atom(
        self, x: Tensor, dim: int | tuple[int, int] = -1, reduce: str = "sum"
    ) -> Tensor:
        """
        Reduce orbital-resolved tensor to atom-resolved tensor

        Parameters
        ----------
        x : Tensor
            Orbital-resolved tensor
        dim : int | (int, int)
            Dimension to reduce over, defaults to -1
        reduce : str
            Reduction method, defaults to "sum"

        Returns
        -------
        Tensor
            Atom-resolved tensor
        """

        return self.reduce_shell_to_atom(
            self.reduce_orbital_to_shell(x, dim=dim, reduce=reduce),
            dim=dim,
            reduce=reduce,
        )

    def spread_atom_to_shell(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread atom-resolved tensor to shell-resolved tensor

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Shell-resolved tensor
        """

        return wrap_gather(x, dim, self.shells_to_atom)

    def spread_shell_to_orbital(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread shell-resolved tensor to orbital-resolved tensor

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Orbital-resolved tensor
        """

        return wrap_gather(x, dim, self.orbitals_to_shell)

    def spread_atom_to_orbital(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread atom-resolved tensor to orbital-resolved tensor

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Orbital-resolved tensor
        """

        return self.spread_shell_to_orbital(
            self.spread_atom_to_shell(x, dim=dim), dim=dim
        )

    def spread_uspecies_to_atom(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread unique species tensor to atom-resolved tensor

        Parameters
        ----------
        x : Tensor
            Unique specie tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Atom-resolved tensor
        """

        return wrap_gather(x, dim, self.atom_to_unique)

    def spread_uspecies_to_shell(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread unique species tensor to shell-resolved tensor

        Parameters
        ----------
        x : Tensor
            Unique specie tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Shell-resolved tensor
        """

        return self.spread_atom_to_shell(
            self.spread_uspecies_to_atom(x, dim=dim), dim=dim
        )

    def spread_uspecies_to_orbital(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread unique species tensor to orbital-resolved tensor

        Parameters
        ----------
        x : Tensor
            Unique specie tensor
        dim : int
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Orbital-resolved tensor
        """

        return self.spread_atom_to_orbital(
            self.spread_uspecies_to_atom(x, dim=dim), dim=dim
        )

    def spread_ushell_to_shell(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread unique shell tensor to shell-resolved tensor

        Parameters
        ----------
        x : Tensor
            Unique shell tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Shell-resolved tensor
        """

        return wrap_gather(x, dim, self.shells_to_ushell)

    def spread_ushell_to_orbital(
        self, x: Tensor, dim: int | tuple[int, int] = -1
    ) -> Tensor:
        """
        Spread unique shell tensor to orbital-resolved tensor

        Parameters
        ----------
        x : Tensor
            Unique shell tensor
        dim : int | (int, int)
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Orbital-resolved tensor
        """

        return self.spread_shell_to_orbital(
            self.spread_ushell_to_shell(x, dim=dim), dim=dim
        )

    @property
    def device(self) -> torch.device | None:
        """The device on which the `IndexHelper` object resides."""
        return self.__device

    @device.setter
    def device(self, *args: Any) -> NoReturn:
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by IndexHelper object."""
        return self.__dtype

    def to(self, device: torch.device) -> IndexHelper:
        """
        Returns a copy of the `IndexHelper` instance on the specified device.

        This method creates and returns a new copy of the `IndexHelper` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        IndexHelper
            A copy of the `IndexHelper` instance placed on the specified device.

        Notes
        -----
        If the `IndexHelper` instance is already on the desired device `self`
        will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(
            self.unique_angular.to(device=device),
            self.angular.to(device=device),
            self.atom_to_unique.to(device=device),
            self.ushells_to_unique.to(device=device),
            self.shells_to_ushell.to(device=device),
            self.shells_per_atom.to(device=device),
            self.shell_index.to(device=device),
            self.shells_to_atom.to(device=device),
            self.orbitals_per_shell.to(device=device),
            self.orbital_index.to(device=device),
            self.orbitals_to_shell.to(device=device),
            device=self.device,
        )

    def type(self, dtype: torch.dtype) -> IndexHelper:
        """
        Returns a copy of the `IndexHelper` instance with specified floating point type.
        This method creates and returns a new copy of the `IndexHelper` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Type of the floating point numbers used by the `IndexHelper` instance.

        Returns
        -------
        IndexHelper
            A copy of the `IndexHelper` instance with the specified dtype.

        Notes
        -----
        If the `IndexHelper` instance has already the desired dtype `self` will
        be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__(
            self.unique_angular.type(dtype),
            self.angular.type(dtype),
            self.atom_to_unique.type(dtype),
            self.ushells_to_unique.type(dtype),
            self.shells_to_ushell.type(dtype),
            self.shells_per_atom.type(dtype),
            self.shell_index.type(dtype),
            self.shells_to_atom.type(dtype),
            self.orbitals_per_shell.type(dtype),
            self.orbital_index.type(dtype),
            self.orbitals_to_shell.type(dtype),
        )

    def get_shell_indices(self, atom_idx: int) -> Tensor:
        """Get shell indices belong to given atom.

        Parameters
        ----------
        atom_idx : int
            Index of given atom.

        Returns
        -------
        Tensor
            Index list of shells belonging to given atom.
        """
        return (self.shells_to_atom == atom_idx).nonzero(as_tuple=True)[0]

    def get_orbital_indices(self, shell_idx: int) -> Tensor:
        """Get orbital indices belong to given shell.

        Parameters
        ----------
        shell_idx : int
            Index of given shell.

        Returns
        -------
        Tensor
            Index list of orbitals belonging to given shell.
        """
        return (self.orbitals_to_shell == shell_idx).nonzero(as_tuple=True)[0]

    def orbital_atom_mapping(self, atom_idx: int) -> Tensor:
        """Mapping of atom index to orbital index,
            i.e. return indices of orbitals belonging
            to given atom. The orbital order is given
            by the IndexHelper ihelp.orbitals_to_shell
            attribute.

        Args:
            atom_idx (int): Index of target atom

        Returns:
            Tensor: 1d-Tensor containing the indices of the orbitals
        """
        return torch.tensor(
            [
                oidx
                for sidx in self.get_shell_indices(atom_idx)
                for oidx in self.get_orbital_indices(sidx).tolist()
            ]
        )

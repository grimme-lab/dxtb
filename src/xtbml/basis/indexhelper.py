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

from ..exlibs.tbmalt import batch
from ..typing import Tensor


def _fill(index: Tensor, repeat: Tensor) -> Tensor:
    """
    Fill an index map using index offsets and number of repeats
    """
    index_map = index.new_zeros(torch.sum(repeat))
    for idx, offset, count in zip(torch.arange(index.shape[-1]), index, repeat):
        index_map[offset : offset + count] = idx
    return index_map


class IndexHelper:
    """
    Index helper for basis set
    """

    angular: Tensor
    """Angular momenta for all shells"""

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
        angular: Tensor,
        shells_per_atom: Tensor,
        shell_index: Tensor,
        shells_to_atom: Tensor,
        orbitals_per_shell: Tensor,
        orbital_index: Tensor,
        orbitals_to_shell: Tensor,
    ):
        self.angular = angular
        self.shells_per_atom = shells_per_atom
        self.shell_index = shell_index
        self.shells_to_atom = shells_to_atom
        self.orbitals_per_shell = orbitals_per_shell
        self.orbital_index = orbital_index
        self.orbitals_to_shell = orbitals_to_shell

        self.__device = shells_per_atom.device
        self.__dtype = shells_per_atom.dtype

        if any(
            [
                tensor.dtype != self.dtype
                for tensor in (
                    # self.angular,
                    self.shells_per_atom,
                    self.shell_index,
                    self.shells_to_atom,
                    self.orbitals_per_shell,
                    self.orbital_index,
                    self.orbitals_to_shell,
                )
            ]
        ):
            raise ValueError("All tensors must have same dtype")

        if any(
            [
                tensor.device != self.device
                for tensor in (
                    self.angular,
                    self.shells_per_atom,
                    self.shell_index,
                    self.shells_to_atom,
                    self.orbitals_per_shell,
                    self.orbital_index,
                    self.orbitals_to_shell,
                )
            ]
        ):
            raise ValueError("All tensors must be on the same device")

    @classmethod
    def from_numbers(
        cls,
        numbers: Tensor,
        angular: dict[int, list[int | float]],
        dtype: torch.dtype = torch.int64,
    ) -> "IndexHelper":
        """
        Construct an index helper instance from atomic numbers and their angular momenta.

        Args:
            numbers (Tensor)
                Atomic numbers for the system
            angular (dict[int, Tensor])
                Map between atomic numbers and angular momenta of all shells

        Returns:
            IndexHelper
                Instance of index helper for given basis set
        """
        pad_val = -999

        batched = numbers.ndim > 1

        _angular = batch.pack(
            [
                torch.tensor(angular.get(number.item(), []), dtype=dtype)
                for number in numbers.flatten()
            ],
            value=pad_val,
        )
        _angular = _angular.reshape((*numbers.shape, _angular.shape[-1]))

        nsh_at = torch.sum(_angular >= 0, -1)
        ish_at = torch.cumsum(nsh_at, -1) - nsh_at
        ish_at[nsh_at == 0] = pad_val

        if batched:
            sh2at = batch.pack(
                [
                    _fill(ish_at[_batch, :], nsh_at[_batch, :])
                    for _batch in range(numbers.shape[0])
                ],
                value=0,
            )
        else:
            sh2at = _fill(ish_at, nsh_at)

        if batched:
            lsh = batch.pack(
                [
                    _angular[_batch, _angular[_batch, :] >= 0]
                    for _batch in range(numbers.shape[0])
                ],
                value=pad_val,
            )
        else:
            lsh = _angular[_angular > pad_val]

        nao_sh = torch.where(lsh >= 0, 2 * lsh + 1, torch.tensor(0, dtype=dtype))

        # NOTE:
        # If we care for the actual values in `angular`, we must allow other
        # dtypes apart from `torch.int64` (example: shell-resolved Hubbard
        # parameters in ES2). For these cases, however, orbital indices become
        # meaningless and we can simply convert to `torch.int64` to avoid an
        # error in the `fill` function.
        nao_sh = nao_sh.type(numbers.dtype)

        iao_sh = torch.cumsum(nao_sh, -1) - nao_sh
        iao_sh[nao_sh == 0] = pad_val

        if batched:
            ao2sh = batch.pack(
                [
                    _fill(iao_sh[_batch, :], nao_sh[_batch, :])
                    for _batch in range(numbers.shape[0])
                ],
                value=0,
            )
        else:
            ao2sh = _fill(iao_sh, nao_sh)

        return cls(
            angular=lsh,
            shells_per_atom=nsh_at,
            shell_index=ish_at,
            shells_to_atom=sh2at,
            orbitals_per_shell=nao_sh,
            orbital_index=iao_sh,
            orbitals_to_shell=ao2sh,
        )

    def reduce_orbital_to_shell(
        self, x: Tensor, dim: int = -1, reduce: str = "sum"
    ) -> Tensor:
        """
        Reduce orbital-resolved tensor to shell-resolved tensor

        Parameters
        ----------
        x : Tensor
            Orbital-resolved tensor
        dim : int
            Dimension to reduce over, defaults to -1
        reduce : str
            Reduction method, defaults to "sum"

        Returns
        -------
        Tensor
            Shell-resolved tensor
        """

        return torch.scatter_reduce(x, dim, self.orbitals_to_shell, reduce=reduce)

    def reduce_shell_to_atom(
        self, x: Tensor, dim: int = -1, reduce: str = "sum"
    ) -> Tensor:
        """
        Reduce shell-resolved tensor to atom-resolved tensor

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor
        dim : int
            Dimension to reduce over, defaults to -1
        reduce : str
            Reduction method, defaults to "sum"

        Returns
        -------
        Tensor
            Atom-resolved tensor
        """

        return torch.scatter_reduce(x, dim, self.shells_to_atom, reduce=reduce)

    def reduce_orbital_to_atom(
        self, x: Tensor, dim: int = -1, reduce: str = "sum"
    ) -> Tensor:
        """
        Reduce orbital-resolved tensor to atom-resolved tensor

        Parameters
        ----------
        x : Tensor
            Orbital-resolved tensor
        dim : int
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

    def spread_atom_to_shell(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Spread atom-resolved tensor to shell-resolved tensor

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor
        dim : int
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Shell-resolved tensor
        """

        return torch.gather(x, dim, self.shells_to_atom)

    def spread_shell_to_orbital(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Spread shell-resolved tensor to orbital-resolved tensor

        Parameters
        ----------
        x : Tensor
            Shell-resolved tensor
        dim : int
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Orbital-resolved tensor
        """

        return torch.gather(x, dim, self.orbitals_to_shell)

    def spread_atom_to_orbital(self, x: Tensor, dim: int = -1) -> Tensor:
        """
        Spread atom-resolved tensor to orbital-resolved tensor

        Parameters
        ----------
        x : Tensor
            Atom-resolved tensor
        dim : int
            Dimension to spread over, defaults to -1

        Returns
        -------
        Tensor
            Orbital-resolved tensor
        """

        return self.spread_shell_to_orbital(
            self.spread_atom_to_shell(x, dim=dim), dim=dim
        )

    @property
    def device(self) -> torch.device:
        """The device on which the `IndexHelper` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by IndexHelper object."""
        return self.__dtype

    def to(self, device: torch.device) -> "IndexHelper":
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
        If the `IndexHelper` instance is already on the desired device `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(
            self.angular.to(device=device),
            self.shells_per_atom.to(device=device),
            self.shell_index.to(device=device),
            self.shells_to_atom.to(device=device),
            self.orbitals_per_shell.to(device=device),
            self.orbital_index.to(device=device),
            self.orbitals_to_shell.to(device=device),
        )

    def type(self, dtype: torch.dtype) -> "IndexHelper":
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
        If the `IndexHelper` instance has already the desired dtype `self` will be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__(
            self.angular.type(dtype),
            self.shells_per_atom.type(dtype),
            self.shell_index.type(dtype),
            self.shells_to_atom.type(dtype),
            self.orbitals_per_shell.type(dtype),
            self.orbital_index.type(dtype),
            self.orbitals_to_shell.type(dtype),
        )

"""
Provides base class for interactions in the extended tight-binding Hamiltonian.
The `Interaction` class is not purely abstract as its methods return zero.
"""
from __future__ import annotations

import torch

from .._types import Any, Tensor, TensorLike
from ..basis import IndexHelper


class Interaction(TensorLike):
    """
    Base class for defining interactions with the charge density.
    """

    label: str
    """Label for the interaction."""

    __slots__ = ["label"]

    class Cache:
        """
        Restart data for individual interactions, extended by subclasses as needed.
        """

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__(device, dtype)
        self.label = self.__class__.__name__

    # pylint: disable=unused-argument
    def get_cache(
        self, *, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Cache:
        """
        Create restart data for individual interactions.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning an empty `Cache`.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Cartesian coordinates.
        ihelp: IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Interaction.Cache
            Restart data for the interaction.
        """

        return self.Cache()

    def get_potential(
        self, charges: Tensor, cache: Interaction.Cache, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        cache : Interaction.Cache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """

        qsh = ihelp.reduce_orbital_to_shell(charges)
        vsh = self.get_shell_potential(qsh, cache)

        qat = ihelp.reduce_shell_to_atom(qsh)
        vat = self.get_atom_potential(qat, cache)

        vsh += ihelp.spread_atom_to_shell(vat)
        return ihelp.spread_shell_to_orbital(vsh)

    def get_shell_potential(self, charges: Tensor, *_) -> Tensor:
        """
        Compute the potential from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_atom_potential(self, charges: Tensor, *_) -> Tensor:
        """
        Compute the potential from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_energy(
        self, charges: Tensor, cache: Interaction.Cache, ihelp: IndexHelper
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        cache : Interaction.Cache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom resolved energy vector.

        Note
        ----
        The subclasses of `Interaction` should implement the `get_atom_energy`
        and `get_shell_energy` methods.
        """
        qsh = ihelp.reduce_orbital_to_shell(charges)
        esh = self.get_shell_energy(qsh, cache)

        qat = ihelp.reduce_shell_to_atom(qsh)
        eat = self.get_atom_energy(qat, cache)

        return eat + ihelp.reduce_shell_to_atom(esh)

    def get_shell_energy(self, charges: Tensor, *_: Any) -> Tensor:
        """
        Compute the energy from the charges, all quantities are shell-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.

        Returns
        -------
        Tensor
            Energy vector for each shell partial charge.
        """

        return torch.zeros_like(charges)

    def get_atom_energy(self, charges: Tensor, *_: Any) -> Tensor:
        """
        Compute the energy from the charges, all quantities are atom-resolved.

        This method should be implemented by the subclass. Here, it serves
        only to create an empty `Interaction` by returning zeros.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.

        Returns
        -------
        Tensor
            Energy vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_gradient(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
        cache: Interaction.Cache,
        ihelp: IndexHelper,
    ) -> Tensor:
        """
        Compute the nuclear gradient using orbital-resolved charges.

        Note
        ----
        This method calls both `get_atom_gradient` and `get_shell_gradient` and
        adds up both gradients. Hence, one of the contributions must be zero.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Cartesian coordinates.
        charges : Tensor
            Orbital-resolved partial charges.
        cache : Interaction.Cache
            Restart data for the interaction.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """

        qsh = ihelp.reduce_orbital_to_shell(charges)
        gsh = self.get_shell_gradient(numbers, positions, qsh, cache, ihelp)

        qat = ihelp.reduce_shell_to_atom(qsh)
        gat = self.get_atom_gradient(numbers, positions, qat, cache)

        return gsh + gat

    def get_shell_gradient(self, _: Any, positions: Tensor, *__: Any) -> Tensor:
        """
        Return zero gradient.

        This method should be implemented by the subclass.
        However, returning zeros here serves three purposes:
         - the interaction can (theoretically) be empty
         - the gradient of the interaction is indeed zero and thus requires no
           gradient implementation (one can, however, implement a method that returns zeros to make this more obvious)
         - the interaction always uses atom-resolved charges and shell-resolved
           charges are never required

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates (for shape of gradient).

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """

        return torch.zeros_like(positions)

    def get_atom_gradient(self, _: Any, positions: Tensor, *__: Any) -> Tensor:
        """
        Return zero gradient.

        This method should be implemented by the subclass.
        However, returning zeros here serves three purposes:
         - the interaction can (theoretically) be empty
         - the gradient of the interaction is indeed zero and thus requires no
           gradient implementation (one can, however, implement a method that
           returns zeros to make this more obvious)
         - the interaction always uses shell-resolved charges and atom-resolved
           charges are never required

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates (for shape of gradient).

        Returns
        -------
        Tensor
            Nuclear gradient for each atom.
        """

        return torch.zeros_like(positions)

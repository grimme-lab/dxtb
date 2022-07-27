"""
Provides abstract base class for interactions in the extended tight-binding Hamiltonian
"""

import torch

from ..typing import Tensor
from ..basis import IndexHelper


class Interaction:
    """
    Base class for defining interactions with the charge density.
    """

    label: str
    """Label for the interaction."""

    class Cache:
        """
        Restart data for individual interactions, extended by subclasses as needed.
        """

        pass

    def __init__(self):
        self.label = self.__class__.__name__

    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> "Interaction.Cache":
        """
        Create restart data for individual interactions.

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
        self, charges: Tensor, ihelp: IndexHelper, cache: "Interaction.Cache"
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """
        qsh = ihelp.reduce_orbital_to_shell(charges)
        vsh = self.get_shell_potential(qsh, ihelp, cache)

        qat = ihelp.reduce_shell_to_atom(qsh)
        vat = self.get_atom_potential(qat, ihelp, cache)

        vsh += ihelp.spread_atom_to_shell(vat)
        return ihelp.spread_shell_to_orbital(vsh)

    def get_shell_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Interaction.Cache"
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are shell-resolved.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_atom_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Interaction.Cache"
    ) -> Tensor:
        """
        Compute the potential from the charges, all quantities are atom-resolved.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_energy(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Interaction.Cache"
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom resolved energy vector.
        """

        qsh = ihelp.reduce_orbital_to_shell(charges)
        esh = self.get_shell_energy(qsh, ihelp, cache)

        qat = ihelp.reduce_shell_to_atom(qsh)
        eat = self.get_atom_energy(qat, ihelp, cache)

        return eat + ihelp.reduce_shell_to_atom(esh)

    def get_shell_energy(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Interaction.Cache"
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are shell-resolved.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each shell partial charge.
        """

        return torch.zeros_like(charges)

    def get_atom_energy(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Interaction.Cache"
    ) -> Tensor:
        """
        Compute the energy from the charges, all quantities are atom-resolved.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Energy vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

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

    def get_potential(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the potential from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each orbital partial charge.
        """
        qsh = ihelp.reduce_orbital_to_shell(charges)
        vsh = self.get_shell_potential(qsh, ihelp)

        qat = ihelp.reduce_shell_to_atom(qsh)
        vat = self.get_atom_potential(qat, ihelp)

        vsh += ihelp.spread_atom_to_shell(vat)
        return ihelp.spread_shell_to_orbital(vsh)

    def get_shell_potential(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the potential from the charges, all quantities are shell-resolved.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_atom_potential(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the potential from the charges, all quantities are atom-resolved.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Potential vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

    def get_energy(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the energy from the charges, all quantities are orbital-resolved.

        Parameters
        ----------
        charges : Tensor
            Orbital-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Atom resolved energy vector.
        """

        qsh = ihelp.reduce_orbital_to_shell(charges)
        esh = self.get_shell_energy(qsh, ihelp)

        qat = ihelp.reduce_shell_to_atom(qsh)
        eat = self.get_atom_energy(qat, ihelp)

        return eat + ihelp.reduce_shell_to_atom(esh)

    def get_shell_energy(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the energy from the charges, all quantities are shell-resolved.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Energy vector for each shell partial charge.
        """

        return torch.zeros_like(charges)

    def get_atom_energy(self, charges: Tensor, ihelp: IndexHelper) -> Tensor:
        """
        Compute the energy from the charges, all quantities are atom-resolved.

        Parameters
        ----------
        charges : Tensor
            Atom-resolved partial charges.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Energy vector for each atom partial charge.
        """

        return torch.zeros_like(charges)

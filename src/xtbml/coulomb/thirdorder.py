"""
On-site third-order electrostatic energy (ES3)
==============================================

This module implements the third-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import xtbml.coulomb.thirdorder as es3
>>> from xtbml.param import GFN1_XTB, get_element_param
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [0.00000000000000, -0.00000000000000, 0.00000000000000],
...     [1.61768389755830, 1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [1.61768389755830, -1.61768389755830, 1.61768389755830],
...     [-1.61768389755830, 1.61768389755830, 1.61768389755830],
... ])
>>> qat = torch.tensor([
...     -8.41282505804719e-2,
...     2.10320626451180e-2,
...     2.10320626451178e-2,
...     2.10320626451179e-2,
...     2.10320626451179e-2,
... ])
>>> hubbard_derivs = get_element_param(GFN1_XTB.element, "gam3")
>>> es = es3.ES3(hubbard_derivs)
>>> e = es.get_energy(numbers, qat)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0155669)
"""

import torch

from ..typing import Tensor
from ..interaction import Interaction
from ..basis import IndexHelper


class ES3(Interaction):
    """On-site third-order electrostatic energy."""

    hubbard_derivs: Tensor
    "Hubbard derivatives of all atoms."

    class Cache(Interaction.Cache):
        """Restart data for the interaction."""

        pass

    def __init__(self, hubbard_derivs: Tensor) -> None:
        Interaction.__init__(self)
        self.hubbard_derivs = hubbard_derivs

    def get_cache(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Interaction.Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers.
        positions : Tensor
            Cartesian coordinates.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Interaction.Cache
            Restart data for the interaction.
        """

        return self.Cache()

    def get_atom_energy(
        self,
        charges: Tensor,
        ihelp: IndexHelper,
        cache: Interaction.Cache,
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:
        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
        S. Spicher and S. Grimme, *WIREs Computational Molecular Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493 <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction energies.
        """

        return (
            ihelp.spread_uspecies_to_atom(self.hubbard_derivs)
            * torch.pow(charges, 3.0)
            / 3.0
        )

    def get_atom_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: Interaction.Cache
    ) -> Tensor:
        """Calculate the third-order electrostatic potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        ihelp : IndexHelper
            Index mapping for the basis set.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction potential.
        """

        return ihelp.spread_uspecies_to_atom(self.hubbard_derivs) * torch.pow(
            charges, 2.0
        )

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


class ES3:
    """On-site third-order electrostatic energy."""

    hubbard_derivs: Tensor
    "Hubbard derivatives of all elements."

    def __init__(self, hubbard_derivs: Tensor) -> None:
        self.hubbard_derivs = hubbard_derivs

    def get_energy(
        self,
        numbers: Tensor,
        qat: Tensor,
    ) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:
        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
        S. Spicher and S. Grimme, *WIREs Computational Molecular Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493 <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of the atoms.
        qat : Tensor
            Atomic charges of all atoms.


        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction energies.
        """

        return self.hubbard_derivs[numbers] * torch.pow(qat, 3.0) / 3.0

    def get_potential(self, numbers: Tensor, qat: Tensor) -> Tensor:
        """Calculate the third-order electrostatic potential.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of the atoms.
        qat : Tensor
            Atomic charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction potential.
        """

        return self.hubbard_derivs[numbers] * torch.pow(qat, 2.0)

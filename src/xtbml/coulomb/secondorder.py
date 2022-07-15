"""
Isotropic second-order electrostatic energy (ES2)
=================================================

This module implements the second-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import xtbml.coulomb.secondorder as es2
>>> from xtbml.coulomb.average import harmonic_average as average
>>> from xtbml.param import GFN1_XTB, get_element_param
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [0.00000000000000, -0.00000000000000, 0.00000000000000],
...     [1.61768389755830, 1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [1.61768389755830, -1.61768389755830, 1.61768389755830],
...     [-1.61768389755830, 1.61768389755830, 1.61768389755830],
... ])
>>> q = torch.tensor([
...     -8.41282505804719e-2,
...     2.10320626451180e-2,
...     2.10320626451178e-2,
...     2.10320626451179e-2,
...     2.10320626451179e-2,
... ])
>>> # get parametrization
>>> gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
>>> hubbard = get_element_param(GFN1_XTB.element, "gam")
>>> # calculate energy
>>> es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)
>>> cache = es.get_cache(numbers, positions)
>>> e = es.get_energy(cache, qat)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0005078)
"""


from __future__ import annotations
import torch

from .average import AveragingFunction, harmonic_average
from ..basis.indexhelper import IndexHelper
from ..exlibs.tbmalt import batch
from ..typing import Tensor
from ..interaction import Interaction


class ES2(Interaction):
    """Isotropic second-order electrostatic energy (ES2)"""

    hubbard: Tensor
    """Hubbard parameters of all elements."""

    lhubbard: dict[int, list[float]] | None = None
    """Shell-resolved scaling factors for Hubbard parameters (default: None, i.e no shell resolution)."""

    average: AveragingFunction = harmonic_average
    """Function to use for averaging the Hubbard parameters (default: harmonic_average)."""

    gexp: Tensor = torch.tensor(2.0)
    """Exponent of the second-order Coulomb interaction (default: 2.0)."""

    ihelp: IndexHelper | None = None
    """Index helper for shell-resolved Hubbard parameters."""

    shell_resolved: bool
    """Electrostatics is shell-resolved"""

    class Cache:
        """Cache for Coulomb matrix."""

        def __init__(self, mat):
            self.mat = mat

    def __init__(
        self,
        hubbard: Tensor,
        lhubbard: dict[int, list[float]] | None = None,
        average: AveragingFunction = harmonic_average,
        gexp: Tensor = torch.tensor(2.0),
    ) -> None:
        Interaction.__init__(self)
        self.hubbard = hubbard
        self.lhubbard = lhubbard
        self.average = average
        self.gexp = gexp

        self.shell_resolved = lhubbard is not None

    def get_cache(
        self,
        numbers: Tensor,
        positions: Tensor,
        ihelp: IndexHelper,
    ):

        return self.Cache(
            self.get_shell_coulomb_matrix(numbers, positions, ihelp)
            if self.shell_resolved
            else self.get_atom_coulomb_matrix(numbers, positions, ihelp)
        )

    def get_atom_coulomb_matrix(self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper):
        """
        Calculate the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """

        h = self.hubbard[numbers]

        # masks
        real = h != 0
        mask = real.unsqueeze(-2) * real.unsqueeze(-1)
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        # all distances to the power of "gexp" (R^2_AB from Eq.26)
        dist_gexp = torch.where(
            mask,
            torch.pow(torch.cdist(positions, positions, p=2), self.gexp),
            torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype),
        )

        # Eq.30: averaging function for hardnesses (Hubbard parameter)
        avg = self.average(h)

        # Eq.26: Coulomb matrix
        return 1.0 / torch.pow(dist_gexp + torch.pow(avg, -self.gexp), 1.0 / self.gexp)

    def get_shell_coulomb_matrix(self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper):
        """
        Calculate the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """

        unique = torch.unique(numbers)

        lh = positions.new_tensor([
            u
            for specie in unique
            for u in self.lhubbard.get(specie.item(), [0.0])
        ])
        h = lh * self.hubbard[unique][ihelp.ushells_to_unique]

        # masks
        real = numbers != 0
        mask = real.unsqueeze(-2) * real.unsqueeze(-1)
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        # all distances to the power of "gexp" (R^2_AB from Eq.26)
        dist_gexp = ihelp.spread_atom_to_shell(
            torch.where(
                mask,
                torch.pow(torch.cdist(positions, positions, p=2), self.gexp),
                torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype),
            ),
            (-1, -2),
        )

        # Eq.30: averaging function for hardnesses (Hubbard parameter)
        avg = self.average(h[ihelp.shells_to_ushell])

        # Eq.26: Coulomb matrix
        return 1.0 / torch.pow(dist_gexp + torch.pow(avg, -self.gexp), 1.0 / self.gexp)

    def get_atom_energy(self, charges: Tensor, ihelp: IndexHelper, cache: "Cache") -> Tensor:
        return 0.5 * charges * self.get_atom_potential(charges, ihelp, cache)

    def get_shell_energy(self, charges: Tensor, ihelp: IndexHelper, cache: "Cache") -> Tensor:
        return 0.5 * charges * self.get_shell_potential(charges, ihelp, cache)

    def get_atom_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Cache"
    ) -> Tensor:
        return (
            torch.zeros_like(charges)
            if self.shell_resolved
            else torch.einsum("...ik,...k->...i", cache.mat, charges)
        )

    def get_shell_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: "Cache"
    ) -> Tensor:
        return (
            torch.einsum("...ik,...k->...i", cache.mat, charges)
            if self.shell_resolved
            else torch.zeros_like(charges)
        )

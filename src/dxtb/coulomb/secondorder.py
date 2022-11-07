"""
Isotropic second-order electrostatic energy (ES2)
=================================================

This module implements the second-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import xtbml.coulomb.secondorder as es2
>>> from dxtb.coulomb.average import harmonic_average as average
>>> from dxtb.param import GFN1_XTB, get_element_param
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
>>> es = es2.ES2(positions, hubbard=hubbard, average=average, gexp=gexp)
>>> cache = es.get_cache(numbers, positions)
>>> e = es.get_energy(cache, qat)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0005078)
"""

import torch

from ..basis import IndexHelper
from ..constants import xtb
from ..interaction import Interaction
from ..param import Param, get_elem_param
from ..typing import Tensor
from ..utils import real_pairs
from .average import AveragingFunction, averaging_function, harmonic_average


class ES2(Interaction):
    """
    Isotropic second-order electrostatic energy (ES2).
    """

    hubbard: Tensor
    """Hubbard parameters of all elements."""

    lhubbard: Tensor | None = None
    """
    Shell-resolved scaling factors for Hubbard parameters (default: None, i.e.,
    no shell resolution).
    """

    average: AveragingFunction = harmonic_average
    """
    Function to use for averaging the Hubbard parameters (default:
    harmonic_average).
    """

    gexp: Tensor = torch.tensor(xtb.DEFAULT_ES2_GEXP)
    """Exponent of the second-order Coulomb interaction (default: 2.0)."""

    ihelp: IndexHelper | None = None
    """Index helper for shell-resolved Hubbard parameters."""

    shell_resolved: bool
    """Electrostatics is shell-resolved"""

    class Cache(Interaction.Cache):
        """
        Cache for Coulomb matrix in ES2.
        """

        mat: Tensor
        """Coulomb matrix"""

        def __init__(self, mat):
            self.mat = mat

    def __init__(
        self,
        positions: Tensor,
        hubbard: Tensor,
        lhubbard: Tensor | None = None,
        average: AveragingFunction = harmonic_average,
        gexp: Tensor = torch.tensor(xtb.DEFAULT_ES2_GEXP),
    ) -> None:
        super().__init__(positions.device, positions.dtype)

        self.hubbard = maybe_move(hubbard, self.device, self.dtype)
        self.lhubbard = (
            lhubbard
            if lhubbard is None
            else maybe_move(lhubbard, self.device, self.dtype)
        )
        self.gexp = maybe_move(gexp, self.device, self.dtype)

        self.average = average

        self.shell_resolved = lhubbard is not None

    def get_cache(
        self,
        numbers: Tensor,
        positions: Tensor,
        ihelp: IndexHelper,
    ) -> "ES2.Cache":
        """
        Obtain the cache object containing the Coulomb matrix.

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
        ES2.Cache
            Cache object for second order electrostatics.

        Note
        ----
        The cache of an interaction requires `positions` as they do not change
        during the self-consistent charge iterations.
        """

        return self.Cache(
            self.get_shell_coulomb_matrix(numbers, positions, ihelp)
            if self.shell_resolved
            else self.get_atom_coulomb_matrix(numbers, positions, ihelp)
        )

    def get_atom_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ):
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

        h = ihelp.spread_uspecies_to_atom(self.hubbard)

        # mask
        mask = real_pairs(numbers, diagonal=True)

        # all distances to the power of "gexp" (R^2_AB from Eq.26)
        dist_gexp = torch.where(
            mask,
            torch.pow(
                cdist(positions, mask),
                self.gexp,
            ),
            positions.new_tensor(torch.finfo(positions.dtype).eps),
        )

        # Eq.30: averaging function for hardnesses (Hubbard parameter)
        avg = self.average(h)

        # Eq.26: Coulomb matrix
        return 1.0 / torch.pow(dist_gexp + torch.pow(avg, -self.gexp), 1.0 / self.gexp)

    def get_shell_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ):
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

        if self.lhubbard is None:
            raise ValueError("No 'lhubbard' parameters set.")

        lh = ihelp.spread_ushell_to_shell(self.lhubbard)
        h = lh * ihelp.spread_uspecies_to_shell(self.hubbard)

        # masks
        real = numbers != 0
        mask = real.unsqueeze(-2) * real.unsqueeze(-1)
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        # all distances to the power of "gexp" (R^2_AB from Eq.26)
        dist_gexp = ihelp.spread_atom_to_shell(
            torch.where(
                mask,
                torch.pow(cdist(positions, mask), self.gexp),
                positions.new_tensor(torch.finfo(positions.dtype).eps),
            ),
            (-1, -2),
        )

        # Eq.30: averaging function for hardnesses (Hubbard parameter)
        avg = self.average(h)

        # Eq.26: Coulomb matrix
        return 1.0 / torch.pow(dist_gexp + torch.pow(avg, -self.gexp), 1.0 / self.gexp)

    def get_atom_energy(
        self, charges: Tensor, ihelp: IndexHelper, cache: Cache
    ) -> Tensor:
        return 0.5 * charges * self.get_atom_potential(charges, ihelp, cache)

    def get_shell_energy(
        self, charges: Tensor, ihelp: IndexHelper, cache: Cache
    ) -> Tensor:
        return 0.5 * charges * self.get_shell_potential(charges, ihelp, cache)

    def get_atom_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: Cache
    ) -> Tensor:
        return (
            torch.zeros_like(charges)
            if self.shell_resolved
            else torch.einsum("...ik,...k->...i", cache.mat, charges)
        )

    def get_shell_potential(
        self, charges: Tensor, ihelp: IndexHelper, cache: "ES2.Cache"
    ) -> Tensor:
        return (
            torch.einsum("...ik,...k->...i", cache.mat, charges)
            if self.shell_resolved
            else torch.zeros_like(charges)
        )


def new_es2(
    numbers: Tensor, positions: Tensor, par: Param, shell_resolved: bool = True
) -> ES2 | None:
    """
    Create new instance of ES2.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    par : Param
        Representation of an extended tight-binding model.
    shell_resolved: bool
        Electrostatics is shell-resolved.

    Returns
    -------
    ES2 | None
        Instance of the ES2 class or `None` if no ES2 is used.
    """

    if hasattr(par, "charge") is False or par.charge is None:
        return None

    unique = torch.unique(numbers)
    hubbard = get_elem_param(unique, par.element, "gam")
    lhubbard = (
        get_elem_param(unique, par.element, "lgam") if shell_resolved is True else None
    )
    average = averaging_function[par.charge.effective.average]
    gexp = (
        par.charge.effective.gexp
        if torch.is_tensor(par.charge.effective.gexp)
        else torch.tensor(par.charge.effective.gexp)
    )

    return ES2(positions, hubbard, lhubbard, average, gexp)

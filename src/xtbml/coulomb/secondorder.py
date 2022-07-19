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


class ES2:
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

    def __init__(
        self,
        hubbard: Tensor,
        lhubbard: dict[int, list[float]] | None = None,
        average: AveragingFunction = harmonic_average,
        gexp: Tensor = torch.tensor(2.0),
    ) -> None:
        self.hubbard = hubbard
        self.lhubbard = lhubbard
        self.average = average
        self.gexp = gexp

    def get_cache(
        self,
        numbers: Tensor,
        positions: Tensor,
    ):
        class Cache:
            """Cache for Coulomb matrix."""

            def __init__(self, mat):
                self.mat = mat

        return Cache(self.get_coulomb_matrix(numbers, positions))

    def get_coulomb_matrix(self, numbers: Tensor, positions: Tensor):
        """Calculate the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """

        h = self.hubbard[numbers]

        if self.lhubbard is not None:
            # NOTE: Maybe use own function here instead of "misusing" the IndexHelper?
            self.ihelp = IndexHelper.from_numbers(
                numbers, self.lhubbard, dtype=positions.dtype
            )
            shell_idxs = self.ihelp.shells_to_atom.type(torch.long)

            h = batch.index(h, shell_idxs) * self.ihelp.angular
            positions = batch.index(positions, shell_idxs)

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

    def get_energy(self, cache, q: Tensor):
        """
        Calculate the second-order Coulomb interaction energy.

        Implements Eq.25 of the following paper:
        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
        S. Spicher and S. Grimme, *WIREs Computational Molecular Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493 <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        cache : Cache
            Reusable cache for the Coulomb matrix.
        q : Tensor
            Atomic or shell-resolved charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise second-order Coulomb interaction energies.
        """

        # Eq.25: single and batched matrix-vector multiplication
        mv = 0.5 * torch.einsum("...ik, ...k -> ...i", cache.mat, q)

        if self.lhubbard is not None and self.ihelp is not None:
            return torch.scatter_reduce(
                mv * q, -1, self.ihelp.shells_to_atom, reduce="sum"
            )

        return mv * q

    def get_potential(self, cache, q: Tensor) -> Tensor:
        """Calculate the second-order Coulomb interaction potential.
        
        Parameters
        ----------
        cache : Cache
            Reusable cache for the Coulomb matrix.
        q : Tensor
            Atomic or shell-resolved charges of all atoms.

        Returns
        -------
        Tensor
            Atomwise second-order Coulomb interaction potential.
        """
        
        return torch.einsum("...ik, ...k -> ...i", cache.mat, q)
        
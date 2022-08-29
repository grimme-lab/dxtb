"""
Halogen bond correction
=======================

This module implements the halogen bond correction. The Halogen class is
constructed similar to the Repulsion class.

Example
-------
>>> import torch
>>> from xtbml.basis import IndexHelper
>>> from xtbml.classical import new_halogen
>>> from xtbml.param import GFN1_XTB, get_elem_param
>>> numbers = torch.tensor([35, 35, 7, 1, 1, 1])
>>> positions = torch.tensor([
...     [+0.00000000000000, +0.00000000000000, +3.11495251300000],
...     [+0.00000000000000, +0.00000000000000, -1.25671880600000],
...     [+0.00000000000000, +0.00000000000000, -6.30201130100000],
...     [+0.00000000000000, +1.78712709700000, -6.97470840000000],
...     [-1.54769692500000, -0.89356260400000, -6.97470840000000],
...     [+1.54769692500000, -0.89356260400000, -6.97470840000000],
... ])
>>> xb = new_halogen(numbers, positions, GFN1_XTB)
>>> ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
>>> cache = xb.get_cache(numbers, ihelp)
>>> energy = xb.get_energy(positions, cache)
>>> print(energy.sum(-1))
tensor(0.0025)
"""

from __future__ import annotations
import torch

from ..basis import IndexHelper
from ..data import atomic_rad
from ..exlibs.tbmalt import batch
from ..param import Param, get_elem_param
from ..utils import maybe_move
from ..typing import Tensor, TensorLike


default_cutoff: float = 20.0
"""Default real space cutoff for halogen bonding interactions."""


class Halogen(TensorLike):
    """
    Representation of the halogen bond correction.

    Note
    ----
    The positions are only passed to the constructor for `dtype` and `device`,
    they are no class property, as this setup facilitates geometry optimization.
    """

    numbers: Tensor
    """Atomic numbers of all atoms."""

    halogens: list[int]
    """Atomic numbers of halogen atoms considered in correction."""

    bases: list[int]
    """Atomic numbers of base atoms considered in correction."""

    damp: Tensor
    """Damping factor in Lennard-Jones like potential."""

    rscale: Tensor
    """Scaling factor for atomic radii."""

    bond_strength: Tensor
    """Halogen bond strengths for unique species."""

    cutoff: Tensor = torch.tensor(default_cutoff)
    """Real space cutoff for halogen bonding interactions (default: 20.0)."""

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        damp: Tensor,
        rscale: Tensor,
        bond_strength: Tensor,
        cutoff: Tensor = torch.tensor(default_cutoff),
    ) -> None:
        super().__init__(positions.device, positions.dtype)

        self.numbers = numbers
        self.damp = maybe_move(damp, self.device, self.dtype)
        self.rscale = maybe_move(rscale, self.device, self.dtype)
        self.bond_strength = maybe_move(bond_strength, self.device, self.dtype)
        self.cutoff = maybe_move(cutoff, self.device, self.dtype)

        # element numbers of halogens and bases
        self.halogens = [17, 35, 53, 85]
        self.base = [7, 8, 15, 16]

    class Cache:
        """Cache for the halogen bond parameters."""

        xbond: Tensor
        """Halogen bond strengths."""

        __slots__ = ["xbond"]

        def __init__(self, xbond: Tensor):
            self.xbond = xbond

    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> "Halogen.Cache":
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Repulsion.Cache
            Cache for halogen bond correction.
        """

        xbond = ihelp.spread_uspecies_to_atom(self.bond_strength)
        return self.Cache(xbond)

    def get_energy(self, positions: Tensor, cache: "Halogen.Cache") -> Tensor:
        """
        Handle batchwise and single calculation of halogen bonding energy.

        Returns
        -------
        Tensor
            Atomwise energy contributions from halogen bonds.
        """

        if self.numbers.ndim > 1:
            return batch.pack(
                [
                    self._xbond_energy(
                        self.numbers[_batch],
                        positions[_batch],
                        cache.xbond[_batch],
                    )
                    for _batch in range(self.numbers.shape[0])
                ]
            )
        else:
            return self._xbond_energy(
                self.numbers,
                positions,
                cache.xbond,
            )

    def _xbond_list(self, numbers: Tensor, positions: Tensor) -> Tensor:
        """
        Calculate triples for halogen bonding interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.
        positions : Tensor
            Cartesian coordinates of all atoms.

        Returns
        -------
        Tensor
            Triples for halogen bonding interactions.

        Note
        ----
        We cannot use `self.numbers` here, because it is not batched.
        """

        adj = []

        # find all halogen-base pairs
        for i, i_at in enumerate(numbers):
            for j, j_at in enumerate(numbers):
                if i_at in self.halogens and j_at in self.base:
                    if torch.norm(positions[i, :] - positions[j, :]) > self.cutoff:
                        continue

                    adj.append([i, j, 0])

        # convert to tensor
        adj = torch.tensor(adj, dtype=torch.long, device=self.device)

        # find nearest neighbor of halogen
        for i in range(adj.size(-2)):
            iat = adj[i][0]

            dist = positions.new_tensor(torch.finfo(self.dtype).max)
            for k, kat in enumerate(numbers):
                # skip padding
                if kat == 0:
                    continue

                r1 = torch.norm(positions[iat, :] - positions[k, :])
                if 0.0 < r1 < dist:
                    adj[i, 2] = k
                    dist = r1

        return adj

    def _xbond_energy(
        self,
        numbers: Tensor,
        positions: Tensor,
        xbond: Tensor,
    ) -> Tensor:
        """
        Calculate atomwise energy contribution for each triple of halogen bonding interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms.
        positions : Tensor
            Cartesian coordinates of all atoms.
        bond_strength : Tensor
            Halogen bond strengths.

        Returns
        -------
        Tensor
            Atomwise energy contributions from halogen bonding interactions.

        Note
        ----
        We cannot use `self.numbers` here, because it is not batched.
        """

        halogen_mask = torch.zeros_like(numbers).type(torch.bool)
        for halogen in self.halogens:
            halogen_mask += numbers == halogen

        # return if no halogens are present
        if halogen_mask.nonzero().size(-2) == 0:
            return torch.zeros(numbers.shape, dtype=positions.dtype)

        base_mask = torch.zeros_like(numbers).type(torch.bool)
        for base in self.base:
            base_mask += numbers == base

        # return if no bases are present
        if base_mask.nonzero().size(-2) == 0:
            return torch.zeros(numbers.shape, dtype=positions.dtype)

        # triples for halogen bonding interactions
        adj = self._xbond_list(numbers, positions)

        # parameters
        rads = atomic_rad[numbers].to(self.device).type(self.dtype) * self.rscale

        # init tensor for atomwise energies
        energies = positions.new_zeros(numbers.size(-1))

        for i in range(adj.size(-2)):
            xat = adj[i][0]  # index of halogen atom
            jat = adj[i][1]  # index of base atom
            kat = adj[i][2]  # index of nearest neighbor of halogen atom

            r0xj = rads[xat] + rads[jat]
            dxj = positions[jat, :] - positions[xat, :]
            dxk = positions[kat, :] - positions[xat, :]
            dkj = positions[kat, :] - positions[jat, :]

            d2xj = torch.sum(dxj * dxj)  # distance hal-acc
            d2xk = torch.sum(dxk * dxk)  # distance hal-neighbor
            d2kj = torch.sum(dkj * dkj)  # distance acc-neighbor

            rxj = torch.sqrt(d2xj)
            xy = torch.sqrt(d2xk * d2xj)

            # Lennard-Jones like potential
            lj6 = torch.pow(r0xj / rxj, 6.0)
            lj12 = torch.pow(lj6, 2.0)
            lj = (lj12 - self.damp * lj6) / (1.0 + lj12)

            # cosine of angle (base-halogen-neighbor) via rule of cosines
            cosa = (d2xk + d2xj - d2kj) / xy

            # angle-dependent damping function
            fdamp = torch.pow(0.5 - 0.25 * cosa, 6.0)

            energies[xat] += lj * fdamp * xbond[xat]

        return energies


def new_halogen(
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    cutoff: Tensor = torch.tensor(default_cutoff),
    grad_par: bool = False,
) -> Halogen | None:
    """
    Create new instance of Halogen class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    par : Param
        Representation of an extended tight-binding model.
    cutoff : Tensor
        Real space cutoff for halogen bonding interactions (default: 20.0).

    Returns
    -------
    Halogen | None
        Instance of the Halogen class or `None` if no halogen bond correction is used.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if par.halogen is None:
        return None

    damp = torch.tensor(par.halogen.classical.damping, requires_grad=grad_par)
    rscale = torch.tensor(par.halogen.classical.rscale, requires_grad=grad_par)

    unique = torch.unique(numbers)
    bond_strength = get_elem_param(
        unique, par.element, "xbond", pad_val=0, requires_grad=grad_par
    )

    return Halogen(numbers, positions, damp, rscale, bond_strength, cutoff=cutoff)

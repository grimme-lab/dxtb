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

from .._types import Tensor, TensorLike
from ..basis import IndexHelper
from ..constants import xtb
from ..data import atomic_rad
from ..param import Param, get_elem_param
from ..utils import batch
from .base import Classical

__all__ = ["Halogen", "LABEL_HALOGEN", "new_halogen"]


LABEL_HALOGEN = "Halogen"
"""Label for the 'Halogen' component, coinciding with the class name."""


class Halogen(Classical):
    """
    Representation of the halogen bond correction.
    """

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

    cutoff: Tensor
    """Real space cutoff for halogen bonding interactions (default: 20.0)."""

    __slots__ = ["damp", "rscale", "bond_strength", "cutoff"]

    def __init__(
        self,
        damp: Tensor,
        rscale: Tensor,
        bond_strength: Tensor,
        cutoff: Tensor = torch.tensor(xtb.DEFAULT_XB_CUTOFF),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.damp = damp.to(self.device).type(self.dtype)
        self.rscale = rscale.to(self.device).type(self.dtype)
        self.bond_strength = bond_strength.to(self.device).type(self.dtype)
        self.cutoff = cutoff.to(self.device).type(self.dtype)

        # element numbers of halogens and bases
        self.halogens = [17, 35, 53, 85]
        self.base = [7, 8, 15, 16]

    class Cache(TensorLike):
        """Cache for the halogen bond parameters."""

        xbond: Tensor
        """Halogen bond strengths."""

        __slots__ = ("numbers", "xbond")

        def __init__(
            self,
            numbers: Tensor,
            xbond: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else xbond.device,
                dtype=dtype if dtype is None else xbond.dtype,
            )
            self.numbers = numbers
            self.xbond = xbond

    def get_cache(self, numbers: Tensor, ihelp: IndexHelper) -> Halogen.Cache:
        """
        Store variables for energy calculation.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        Repulsion.Cache
            Cache for halogen bond correction.

        Note
        ----
        The cache of a classical contribution does not require `positions` as
        it only becomes useful if `numbers` remain unchanged and `positions`
        vary, i.e., during geometry optimization.
        """

        xbond = ihelp.spread_uspecies_to_atom(self.bond_strength)
        return self.Cache(numbers, xbond)

    def get_energy(self, positions: Tensor, cache: Halogen.Cache) -> Tensor:
        """
        Handle batchwise and single calculation of halogen bonding energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        cache : Halogen.Cache
            Cache for the halogen bond parameters.

        Returns
        -------
        Tensor
             Atomwise energy contributions from halogen bonds.
        """

        if cache.numbers.ndim > 1:
            return batch.pack(
                [
                    self._xbond_energy(
                        cache.numbers[_batch],
                        positions[_batch],
                        cache.xbond[_batch],
                    )
                    for _batch in range(cache.numbers.shape[0])
                ]
            )
        else:
            return self._xbond_energy(
                cache.numbers,
                positions,
                cache.xbond,
            )

    def _xbond_list(self, numbers: Tensor, positions: Tensor) -> Tensor | None:
        """
        Calculate triples for halogen bonding interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).

        Returns
        -------
        Tensor | None
            Triples for halogen bonding interactions or `None` if no triples
            where found.

        Note
        ----
        We cannot use `self.numbers` here, because it is not batched.
        """

        adjlist = []

        # find all halogen-base pairs
        for i, i_at in enumerate(numbers):
            if i_at not in self.halogens:
                continue

            for j, j_at in enumerate(numbers):
                if j_at not in self.base:
                    continue

                if torch.norm(positions[i, :] - positions[j, :]) > self.cutoff:
                    continue

                adjlist.append([i, j, 0])

        if len(adjlist) == 0:
            return None

        # convert to tensor
        adj = torch.tensor(adjlist, dtype=torch.long, device=self.device)

        # find nearest neighbor of halogen
        for i in range(adj.size(-2)):
            iat = adj[i][0]

            dist = torch.tensor(
                torch.finfo(self.dtype).max,
                dtype=positions.dtype,
                device=positions.device,
            )
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
        Calculate atomwise energy contribution for each triple of halogen
        bonding interactions.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
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
            return torch.zeros(
                numbers.shape, dtype=positions.dtype, device=positions.device
            )

        base_mask = torch.zeros_like(numbers).type(torch.bool)
        for base in self.base:
            base_mask += numbers == base

        # return if no bases are present
        if base_mask.nonzero().size(-2) == 0:
            return torch.zeros(
                numbers.shape, dtype=positions.dtype, device=positions.device
            )

        # triples for halogen bonding interactions
        adj = self._xbond_list(numbers, positions)
        if adj is None:
            return torch.zeros(
                numbers.shape, dtype=positions.dtype, device=positions.device
            )

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
    par: Param,
    cutoff: Tensor = torch.tensor(xtb.DEFAULT_XB_CUTOFF),
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Halogen | None:
    """
    Create new instance of Halogen class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
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

    if hasattr(par, "halogen") is False or par.halogen is None:
        return None

    damp = torch.tensor(par.halogen.classical.damping)
    rscale = torch.tensor(par.halogen.classical.rscale)

    unique = torch.unique(numbers)
    bond_strength = get_elem_param(unique, par.element, "xbond", pad_val=0)

    return Halogen(
        damp, rscale, bond_strength, cutoff=cutoff, device=device, dtype=dtype
    )

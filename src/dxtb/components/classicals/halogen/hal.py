"""
Halogen Bond Correction: Class
==============================

This module implements the halogen bond correction class. The `Halogen` class is
constructed similar to the `Repulsion` class.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack
from tad_mctc.data.radii import ATOMIC as ATOMIC_RADII
from tad_mctc.typing import Tensor, TensorLike

from dxtb.basis import IndexHelper
from dxtb.constants import xtb

from ..base import Classical

__all__ = ["Halogen", "LABEL_HALOGEN"]


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
            return pack(
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
        rads = ATOMIC_RADII.to(**self.dd)[numbers] * self.rscale

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
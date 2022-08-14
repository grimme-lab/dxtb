"""Halogen bond correction."""

from __future__ import annotations
import torch


from ..basis import IndexHelper
from ..data import atomic_rad
from ..exlibs.tbmalt import batch
from ..param import Param, get_elem_param
from ..typing import Tensor, TensorLike


default_cutoff = 20.0


class Halogen(TensorLike):
    """Representation of a halogen bond correction."""

    numbers: Tensor
    """Atomic numbers of all atoms."""

    halogens: list[int]
    """Atomic numbers of halogen atoms considered in correction."""

    acceptors: list[int]
    """Atomic numbers of acceptor atoms considered in correction."""

    damp: Tensor
    """Damping factor in Lennard-Jones like potential."""

    rscale: Tensor
    """Scaling factor for atomic radii."""

    bond_strength: Tensor
    """Halogen bond strengths for unique species."""

    cutoff: float = default_cutoff
    """Real space cutoff for halogen bonding interactions (default: 20.0)."""

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        damp: Tensor,
        rscale: Tensor,
        bond_strength: Tensor,
        cutoff: float = default_cutoff,
    ) -> None:
        super().__init__(positions.device, positions.dtype)

        self.numbers = numbers
        self.damp = damp.to(self.device).type(self.dtype)
        self.rscale = rscale.to(self.device).type(self.dtype)
        self.bond_strength = bond_strength.to(self.device).type(self.dtype)
        self.cutoff = cutoff

        # element numbers of halogens and acceptors
        self.halogens = [17, 35, 53, 85]
        self.acceptors = [7, 8, 15, 16]

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

        c = positions.new_tensor(self.cutoff)

        # find all halogen-acceptor pairs
        for i, i_at in enumerate(numbers):
            for j, j_at in enumerate(numbers):
                if i_at in self.halogens and j_at in self.acceptors:
                    if torch.norm(positions[i, :] - positions[j, :]) > c:
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
                if r1 < dist and r1 > 0.0:
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

        acceptor_mask = torch.zeros_like(numbers).type(torch.bool)
        for acceptor in self.acceptors:
            acceptor_mask += numbers == acceptor

        # return if no acceptors are present
        if acceptor_mask.nonzero().size(-2) == 0:
            return torch.zeros(numbers.shape, dtype=positions.dtype)

        # triples for halogen bonding interactions
        adj = self._xbond_list(numbers, positions)

        # parameters
        rads = atomic_rad[numbers].to(self.device).type(self.dtype) * self.rscale

        # init tensor for atomwise energies
        energies = positions.new_zeros(numbers.size(-1))

        for i in range(adj.size(-2)):
            xat = adj[i][0]  # index of halogen atom
            jat = adj[i][1]  # index of acceptor atom
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

            # cosine of angle (acceptor-halogen-neighbor) via rule of cosines
            cosa = (d2xk + d2xj - d2kj) / xy

            # angle-dependent damping function
            fdamp = torch.pow(0.5 - 0.25 * cosa, 6.0)

            energies[xat] += lj * fdamp * xbond[xat]

        return energies


def new_halogen(
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    cutoff: float = default_cutoff,
    grad: bool = False,
) -> Halogen | None:
    """Create new instance of Halogen class.

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

    damp = torch.tensor(par.halogen.classical.damping)
    rscale = torch.tensor(par.halogen.classical.rscale)

    unique = torch.unique(numbers)
    bond_strength = get_elem_param(unique, par.element, "xbond", pad_val=0)

    return Halogen(numbers, positions, damp, rscale, bond_strength, cutoff=cutoff)

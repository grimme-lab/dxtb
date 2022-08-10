"""Halogen bond correction."""

from __future__ import annotations
import torch



from ..data import atomic_rad
from ..param import Element, Param
from ..typing import Tensor


def get_xbond(par_element: dict[str, Element]) -> Tensor:
    """Obtain halogen bond strengths.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.

    Returns
    -------
    Tensor
        Halogen bond strengths of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
    """

    # dummy for indexing with atomic numbers
    z = [0.0]

    for item in par_element.values():
        z.append(item.xbond)

    return torch.tensor(z)


def get_xbond_list(numbers: Tensor, positions: Tensor, cutoff: Tensor) -> Tensor:
    """Calculate triples for halogen bonding interactions.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    cutoff : Tensor
        Real space cutoff for halogen bonding interactions.

    Returns
    -------
    Tensor
        Triples for halogen bonding interactions.
    """
    acceptors = [7, 8, 15, 16]
    halogens = [17, 35, 53, 85]
    adj = []

    # find all halogen-acceptor pairs
    for i, i_at in enumerate(numbers):
        for j, j_at in enumerate(numbers):
            if i_at in halogens and j_at in acceptors:
                if torch.norm(positions[i, :] - positions[j, :]) > cutoff:
                    continue

                adj.append([i, j, 0])

    # convert to tensor
    adj = torch.tensor(adj)

    # find nearest neighbor of halogen
    for i in range(adj.size(-2)):
        iat = adj[i][0]

        dist = torch.finfo(positions.dtype).max
        for k, kat in enumerate(numbers):
            # skip padding
            if kat == 0:
                continue

            r1 = torch.norm(positions[iat, :] - positions[k, :])
            if r1 < dist and r1 > 0.0:
                adj[i, 2] = k
                dist = r1

    return adj


def get_xbond_energy(
    numbers: Tensor,
    positions: Tensor,
    adj: Tensor,
    damp: Tensor,
    rscale: Tensor,
    bond_strength: Tensor,
) -> Tensor:
    """Calculate atomwise energy contribution for each triple of halogen bonding interactions.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    adj : Tensor
        Triples for halogen bonding interactions.
    damp : Tensor
        Damping factor in Lennard-Jones like potential.
    rscale : Tensor
        Scaling factor for atomic radii.
    bond_strength : Tensor
        Halogen bond strengths.

    Returns
    -------
    Tensor
        Atomwise energy contributions from halogen bonding interactions.
    """

    # parameters
    rads = atomic_rad[numbers] * rscale
    xbond = bond_strength[numbers]

    # init tensor for atomwise energies
    energies = torch.zeros(numbers.size(-1), dtype=positions.dtype)

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
        lj = (lj12 - damp * lj6) / (1.0 + lj12)

        # cosine of angle (acceptor-halogen-neighbor) via rule of cosines
        cosa = (d2xk + d2xj - d2kj) / xy

        # angle-dependent damping function
        fdamp = torch.pow(0.5 - 0.25 * cosa, 6.0)

        energies[xat] += lj * fdamp * xbond[xat]

    return energies


def get_energy(
    numbers: Tensor,
    positions: Tensor,
    damp: Tensor,
    rscale: Tensor,
    bond_strength: Tensor,
    cutoff: Tensor = torch.tensor(20.0),
) -> Tensor:
    """Calculate atomwise energy contribution of halogen bonds for GFN1-xTB.

    Args:
    -----
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    damp : Tensor
        Damping factor in Lennard-Jones like potential.
    rscale : Tensor
        Scaling factor for atomic radii.
    bond_strength : Tensor
        Halogen bond strengths.
    cutoff : Tensor
        Real space cutoff for halogen bonding interactions (default: 20.0).

    Returns:
    --------
    Tensor
        Atomwise energy contributions from halogen bonds.
    """

    acceptor_mask = (numbers == 7) | (numbers == 8) | (numbers == 15) | (numbers == 16)
    if acceptor_mask.nonzero().size(-2) == 0:
        return torch.zeros(numbers.shape, dtype=positions.dtype)

    halogen_mask = (numbers == 17) | (numbers == 35) | (numbers == 53) | (numbers == 85)
    if halogen_mask.nonzero().size(-2) == 0:
        return torch.zeros(numbers.shape, dtype=positions.dtype)

    # triples for halogen bonding interactions
    adj = get_xbond_list(numbers, positions, cutoff)

    return get_xbond_energy(numbers, positions, adj, damp, rscale, bond_strength)


class Halogen:

    numbers: Tensor
    """ Atomic numbers of all atoms."""

    positions: Tensor
    """Cartesian coordinates of all atoms."""

    damp: Tensor
    """Damping factor in Lennard-Jones like potential."""

    rscale: Tensor
    """Scaling factor for atomic radii."""

    bond_strength: Tensor
    """Halogen bond strengths."""

    cutoff: Tensor = torch.tensor(20.0)
    """Real space cutoff for halogen bonding interactions (default: 20.0)."""

    def __init__(
        self,
        numbers: Tensor,
        positions: Tensor,
        damp: Tensor,
        rscale: Tensor,
        bond_strength: Tensor,
        cutoff: Tensor = torch.tensor(20.0),
    ) -> None:
        self.numbers = numbers
        self.positions = positions
        self.damp = damp
        self.rscale = rscale
        self.bond_strength = bond_strength
        self.cutoff = cutoff

    def get_energy(self) -> Tensor:
        """
        Handle batchwise and single calculation of halogen bonding energy.

        Returns
        -------
        Tensor
            Atomwise energy contributions from halogen bonds.
        """

        if len(self.numbers.shape) > 1:
            energies = torch.stack(
                [
                    get_energy(
                        self.numbers[i],
                        self.positions[i],
                        self.damp,
                        self.rscale,
                        self.bond_strength,
                        self.cutoff,
                    )
                    for i in range(self.numbers.shape[0])
                ],
                dim=0,
            )

        else:
            energies = get_energy(
                self.numbers,
                self.positions,
                self.damp,
                self.rscale,
                self.bond_strength,
                self.cutoff,
            )

        return energies


def new_halogen(numbers: Tensor, positions: Tensor, par: Param) -> Halogen:
    """Create new instance of Halogen class.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Halogen
        Instance of the Halogen class.

    Raises
    ------
    ValueError
        If parametrization does not contain a halogen bond correction.
    """

    if par.halogen is None:
        raise ValueError("No halogen bond correction parameters provided.")

    damp = torch.tensor(par.halogen.classical.damping)
    rscale = torch.tensor(par.halogen.classical.rscale)
    bond_strength = get_xbond(par.element)

    return Halogen(numbers, positions, damp, rscale, bond_strength)

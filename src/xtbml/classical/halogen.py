"""Halogen bond correction."""

from __future__ import annotations
import torch


from ..data.atomicrad import atomic_rad
from ..param import Element
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


def get_xbond_list(
    numbers: Tensor, positions: Tensor, dim: int, cutoff: Tensor
) -> Tensor:
    """Calculate triples for halogen bonding interactions.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    positions : Tensor
        Cartesian coordinates of all atoms.
    dim : int
        Maximum number of halogen bonding interactions.
    cutoff : Tensor
        Real space cutoff for halogen bonding interactions.

    Returns
    -------
    Tensor
        Triples for halogen bonding interactions.
    """
    acceptors = [7, 8, 15, 16]
    halogens = [17, 35, 53, 85]

    # init adjacency list
    adj = torch.zeros((dim, 3), dtype=torch.long)

    nxb = 0
    resize = 0
    for i, i_at in enumerate(numbers):
        for j, j_at in enumerate(numbers):
            if i_at in halogens and j_at in acceptors:
                if torch.norm(positions[i, :] - positions[j, :]) > cutoff:
                    resize += 1
                    continue

                adj[nxb, 0] = i
                adj[nxb, 1] = j

                nxb += 1

    # resize adjacency list
    adj = adj[: (dim - resize), :]

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
    num_accs = acceptor_mask.nonzero().size(-2)
    if num_accs == 0:
        return torch.zeros(numbers.shape, dtype=positions.dtype)

    halogen_mask = (numbers == 17) | (numbers == 35) | (numbers == 53) | (numbers == 85)
    num_hals = halogen_mask.nonzero().size(-2)
    if num_hals == 0:
        return torch.zeros(numbers.shape, dtype=positions.dtype)

    # triples for halogen bonding interactions
    adj = get_xbond_list(numbers, positions, num_hals * num_accs, cutoff)

    return get_xbond_energy(numbers, positions, adj, damp, rscale, bond_strength)


def halogen_bond_correction(
    numbers: Tensor,
    positions: Tensor,
    damp: Tensor,
    rscale: Tensor,
    bond_strength: Tensor,
    cutoff: Tensor = torch.tensor(20.0),
) -> Tensor:
    """Handle batchwise and single calculation of halogen bonding energy.

    Parameters
    ----------
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
    cutoff : Tensor, optional
        Real space cutoff for halogen bonding interactions (default: 20.0).

    Returns
    -------
    Tensor
        Atomwise energy contributions from halogen bonds.
    """

    if len(numbers.shape) > 1:
        energies = torch.stack(
            [
                get_energy(
                    numbers[i], positions[i], damp, rscale, bond_strength, cutoff
                )
                for i in range(numbers.shape[0])
            ],
            dim=0,
        )

    else:
        energies = get_energy(numbers, positions, damp, rscale, bond_strength, cutoff)

    return energies

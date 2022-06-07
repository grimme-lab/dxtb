"""Integrator for Born radii based on the Onufriev-Bashford-Case model."""

import torch
from typing import Optional
from ..typing import Tensor
from .data import vdw_rad_d3


def get_born_radii(
    numbers: Tensor,
    positions: Tensor,
    rvdw: Optional[Tensor] = None,
    cutoff: Tensor = torch.tensor(66.0),
    born_scale: float = 1.0,
    born_offset: float = 0.0,
    descreening: float = 0.8,
    obc: Tensor = torch.tensor([1.0, 0.8, 4.85]),
) -> Tensor:
    """
    Calculate Born radii for a set of atoms using the Onufriev-Bashford-Case model published in
    - A. Onufriev, D. Bashford, D. A. Case, *Proteins: Struct., Funct., Bioinf.*, **2004**, 55, 383–394. DOI: `10.1002/prot.20033 < https://doi.org/10.1002/prot.20033>`__

    Args:
        numbers: Atomic numbers of the atoms
        positions: Cartesian coordinates of the atoms
        rvdw: Covalent radii of the atoms (default: VdW radii)
        cutoff: Real-space cutoff for Born radii integration (default: 66.0 Bohr)
        born_scale: Scaling factor for Born radii (default: 1.0)
        born_offset: Offset for Born radii (default: 0.0)
        descreening: Dielectric descreening parameter (default: 0.8)
        obc: Onufriev-Bashford-Case parameters (default: [1.0, 0.8, 4.85])

    Returns:
        Born radii for the atoms
    """

    if cutoff is None:
        cutoff = torch.tensor(25.0, dtype=positions.dtype)
    if rvdw is None:
        rvdw = vdw_rad_d3[numbers].type(positions.dtype)
    if numbers.shape != rvdw.shape:
        raise ValueError(
            "Shape of covalent radii is not consistent with atomic numbers"
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")

    srvdw = rvdw - born_offset

    psi = compute_psi(numbers, positions, rvdw, descreening)

    r1 = torch.divide(torch.tensor(1.0), rvdw)
    s1 = torch.divide(torch.tensor(1.0), srvdw)

    psis2 = psi * 0.5 * srvdw

    # Eq.6: tanh(alpha*psi - beta*psi^2 + gamma*psi^3)
    arg2 = psis2 * (obc[2] * psis2 - obc[1])
    arg = psis2 * (obc[0] + arg2)
    th = torch.tanh(arg)

    brads = torch.divide(torch.tensor(1.0), (s1 - r1 * th)) * born_scale

    return brads


def compute_psi(
    numbers: Tensor,
    positions: Tensor,
    rvdw: Optional[Tensor] = None,
    descreening: float = 0.8,
) -> Tensor:
    if rvdw is None:
        rvdw = vdw_rad_d3[numbers].type(positions.dtype)

    # TODO: implement individual descreening
    rho = rvdw * descreening

    psi = torch.zeros_like(rvdw)
    for i in range(len(numbers)):
        for j in range(i):
            if i == j:
                continue
            vec = positions[i, :] - positions[j, :]
            rij = torch.norm(vec, dim=-1)

            rho_i = rho[i]
            rho_j = rho[j]
            rvdw_i = rvdw[i]
            rvdw_j = rvdw[j]

            r1 = 1.0 / rij

            # ij contribution
            ap_ij = rij + rho_j
            am_ij = rij - rho_j
            ab_ij = ap_ij * am_ij
            rhab_ij = rho_j / ab_ij
            lnab_ij = 0.5 * torch.log(am_ij / ap_ij) * r1
            g_ij = rhab_ij + lnab_ij

            # ji contribution
            ap_ji = rij + rho_i
            am_ji = rij - rho_i
            ab_ji = ap_ji * am_ji
            rhab_ji = rho_i / ab_ji
            lnab_ji = 0.5 * torch.log(am_ji / ap_ji) * r1
            g_ji = rhab_ji + lnab_ji

            # overlap?
            ijov = (rij < (rvdw_i + rho_j)).item()
            jiov = (rij < (rvdw_j + rho_i)).item()

            if not ijov and not jiov:  # nonoverlaping spheres
                # ij contribution
                psi[i] = psi[i] + g_ij

                # ji contribution
                if torch.abs(rho_i - rho_j) < 1e-6:  # equal reduced radii
                    psi[j] = psi[j] + g_ji

                else:  # unequal reduced radii
                    psi[j] = psi[j] + g_ji

            elif not ijov and jiov:
                # ij contribution
                psi[i] = psi[i] + g_ij

                # ji contribution
                if rij + rho_i > rvdw_j:
                    r12 = 0.5 * r1
                    rh1 = 1.0 / rvdw_j
                    rhr1 = 1.0 / ap_ji
                    aprh1_ji = ap_ji * rh1
                    lnab_ji = torch.log(aprh1_ji)

                    g_ji = (
                        rh1
                        - rhr1
                        + r12 * (0.5 * am_ji * (rhr1 - rh1 * aprh1_ji) - lnab_ji)
                    )

                    psi[j] = psi[j] + g_ji

            elif ijov and not jiov:
                # ij contribution
                if (rij + rho_j) > rvdw_i:
                    r12 = 0.5 * r1
                    rh1 = 1.0 / rvdw_i
                    rhr1 = 1.0 / ap_ij
                    aprh1_ij = ap_ij * rh1
                    lnab_ij = torch.log(aprh1_ij)

                    g_ij = (
                        rh1
                        - rhr1
                        + r12 * (0.5 * am_ij * (rhr1 - rh1 * aprh1_ij) - lnab_ij)
                    )

                    psi[i] = psi[i] + g_ij

                # ji contribution
                psi[j] = psi[j] + g_ji

            elif ijov and jiov:  # overlapping spheres

                # ij contribution
                if rij + rho_j > rvdw_i:
                    r12 = 0.5 * r1
                    rh1 = 1.0 / rvdw_i
                    rhr1 = 1.0 / ap_ij
                    aprh1_ij = ap_ij * rh1
                    lnab_ij = torch.log(aprh1_ij)

                    g_ij = (
                        rh1
                        - rhr1
                        + r12 * (0.5 * am_ij * (rhr1 - rh1 * aprh1_ij) - lnab_ij)
                    )

                    psi[i] = psi[i] + g_ij

                # ji contribution
                if rij + rho_i > rvdw_j:
                    r12 = 0.5 * r1
                    rh1 = 1.0 / rvdw_j
                    rhr1 = 1.0 / ap_ji
                    aprh1_ji = ap_ji * rh1
                    lnab_ji = torch.log(aprh1_ji)

                    g_ji = (
                        rh1
                        - rhr1
                        + r12 * (0.5 * am_ji * (rhr1 - rh1 * aprh1_ji) - lnab_ji)
                    )

                    psi[j] = psi[j] + g_ji

    return psi

"""
Born Radii
==========

Integrator for Born radii based on the Onufriev-Bashford-Case model.

Example
-------

>>> import torch
>>> from xtbml.solvation import born
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor(
...     [
...         0.00000000000000,
...         -0.00000000000000,
...         0.00000000000000,
...         1.61768389755830,
...         1.61768389755830,
...         -1.61768389755830,
...         -1.61768389755830,
...         -1.61768389755830,
...         -1.61768389755830,
...         1.61768389755830,
...         -1.61768389755830,
...         1.61768389755830,
...         -1.61768389755830,
...         1.61768389755830,
...         1.61768389755830,
...     ],
... ).reshape((-1, 3))
>>> rads = born.get_born_radii(numbers, positions)
>>> print(rads)
tensor([3.6647, 2.4621, 2.4621, 2.4621, 2.4621])
"""

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
    - A. Onufriev, D. Bashford, D. A. Case, *Proteins: Struct., Funct., Bioinf.*, **2004**, 55, 383–394. DOI: `10.1002/prot.20033 <https://doi.org/10.1002/prot.20033>`__

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

    if rvdw is None:
        rvdw = vdw_rad_d3[numbers].type(positions.dtype)
    if numbers.shape != rvdw.shape:
        raise ValueError(
            "Shape of covalent radii is not consistent with atomic numbers"
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError("Shape of positions is not consistent with atomic numbers")

    # mask for padding
    mask = numbers != 0
    zero = torch.tensor(0.0, dtype=positions.dtype)

    # get dielectric descreening integral I for Eq.6 (psi = I * scaled_rho)
    # NOTE: compute_psi actually only computes I not psi
    psi = compute_psi(numbers, positions, rvdw, cutoff, descreening)

    # some temporary variables
    srvdw = rvdw - born_offset
    r1 = torch.where(mask, 1.0 / rvdw, zero)
    s1 = torch.where(mask, 1.0 / srvdw, zero)
    psis2 = 0.5 * psi * srvdw

    # Eq.6 part 1: tanh(alpha*psi - beta*psi^2 + gamma*psi^3)
    tmp2 = psis2 * (obc[2] * psis2 - obc[1])
    tmp = psis2 * (obc[0] + tmp2)
    th = torch.tanh(tmp)

    # Eq.6 part 2: R^-1 = scaled_rho^-1 - rho^-1 * tanh(arg)
    return torch.where(mask, born_scale / (s1 - r1 * th), zero)


def compute_psi(
    numbers: Tensor,
    positions: Tensor,
    rvdw: Tensor,
    cutoff: Tensor = torch.tensor(66.0),
    descreening: float = 0.8,
) -> Tensor:

    # TODO: implement individual descreening
    rho = rvdw * descreening

    # mask for padding
    real = numbers != 0
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    eps = torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype)
    zero = torch.tensor(0.0, dtype=positions.dtype)

    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2),
        eps,
    )
    r1 = 1.0 / distances

    # mask determining overlapping atoms
    mask_ovlp = distances < (rvdw.unsqueeze(-2) + rho.unsqueeze(-1))

    # mask for contributions from overlapping atoms
    # (emulates "(rij + rho_j) > rvdw_i" and "(rij + rho_i) > rvdw_j")
    mask_ovlp_contr = (distances + rho.unsqueeze(-1)) > (
        rvdw.unsqueeze(-2) * rvdw.new_ones(rvdw.shape).unsqueeze(-1)
    )

    # temporary variables
    d_pl = distances + rho.unsqueeze(-1)
    d_mi = distances - rho.unsqueeze(-1)
    d_pr = torch.where(mask, d_pl * d_mi, zero)
    d_qu = torch.where(mask, d_mi / d_pl, zero)

    # contributions from non-overlapping atoms
    rho_dpr = torch.where(mask, rho.unsqueeze(-1) / d_pr, zero)
    ln_dqu = torch.where(mask, 0.5 * torch.log(d_qu) * r1, zero)
    non_ovlp = rho_dpr + ln_dqu

    # contributions from overlapping atoms
    rvdw1 = torch.where(mask, 1.0 / rvdw.unsqueeze(-2), zero)
    d_pl1 = 1.0 / d_pl
    dpl_rvdw1 = d_pl * rvdw1
    ln_dpl_rvdw1 = torch.where(mask, torch.log(dpl_rvdw1), zero)
    ovlp = (
        rvdw1
        - d_pl1
        + 0.5 * r1 * (0.5 * d_mi * (d_pl1 - rvdw1 * dpl_rvdw1) - ln_dpl_rvdw1)
    )

    psi = torch.where(
        mask * (distances <= cutoff),
        torch.where(mask_ovlp * mask_ovlp_contr, ovlp, non_ovlp),
        zero,
    )

    return torch.sum(psi, dim=-2)


def compute_psi_loops(
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
            vec = positions[i, :] - positions[j, :]
            rij = torch.norm(vec, dim=-1)
            r1 = 1.0 / rij
            r12 = 0.5 * r1

            rho_i = rho[i]
            rho_j = rho[j]
            rvdw_i = rvdw[i]
            rvdw_j = rvdw[j]

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

            #
            rh1_ij = 1.0 / rvdw_i
            rhr1_ij = 1.0 / ap_ij
            aprh1_ij = ap_ij * rh1_ij
            lnab_ij = torch.log(aprh1_ij)

            rh1_ji = 1.0 / rvdw_j
            rhr1_ji = 1.0 / ap_ji
            aprh1_ji = ap_ji * rh1_ji
            lnab_ji = torch.log(aprh1_ji)

            # overlap?
            ijov = (rij < (rvdw_i + rho_j)).item()
            jiov = (rij < (rvdw_j + rho_i)).item()

            if not ijov and not jiov:  # nonoverlaping spheres
                # ij contribution
                psi[i] = psi[i] + g_ij
                psi[j] = psi[j] + g_ji

                # ji contribution
                if torch.abs(rho_i - rho_j) < 1e-6:  # equal reduced radii
                    pass
                else:  # unequal reduced radii
                    pass

            elif not ijov and jiov:
                # ij contribution
                psi[i] = psi[i] + g_ij

                # ji contribution
                if (rij + rho_i) > rvdw_j:
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
                    rh1_ij = 1.0 / rvdw_i
                    rhr1_ij = 1.0 / ap_ij
                    aprh1_ij = ap_ij * rh1_ij
                    lnab_ij = torch.log(aprh1_ij)

                    g_ij = (
                        rh1_ij
                        - rhr1_ij
                        + r12 * (0.5 * am_ij * (rhr1_ij - rh1_ij * aprh1_ij) - lnab_ij)
                    )

                    psi[i] = psi[i] + g_ij

                # ji contribution
                if rij + rho_i > rvdw_j:
                    r12 = 0.5 * r1
                    rh1_ji = 1.0 / rvdw_j
                    rhr1_ji = 1.0 / ap_ji
                    aprh1_ji = ap_ji * rh1_ji
                    lnab_ji = torch.log(aprh1_ji)

                    g_ji = (
                        rh1_ji
                        - rhr1_ji
                        + r12 * (0.5 * am_ji * (rhr1_ji - rh1_ji * aprh1_ji) - lnab_ji)
                    )

                    psi[j] = psi[j] + g_ji

    return psi

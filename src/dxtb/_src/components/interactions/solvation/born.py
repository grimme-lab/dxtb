# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Born Radii
==========

Integrator for Born radii based on the Onufriev-Bashford-Case model.

Example
-------

.. code-block:: python

    import torch
    from dxtb.solvation import born

    # Define atomic numbers and positions of the atoms
    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, -0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])

    # Calculate the Born radii for the given atomic configuration
    rads = born.get_born_radii(numbers, positions)

    # Print the calculated Born radii
    print(rads)  # Output: tensor([3.6647, 2.4621, 2.4621, 2.4621, 2.4621])
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.batch import real_atoms, real_pairs
from tad_mctc.data.radii import VDW_D3

from dxtb._src.typing import DD, Tensor

__all__ = ["get_born_radii"]


def get_born_radii(
    numbers: Tensor,
    positions: Tensor,
    rvdw: Tensor | None = None,
    cutoff: Tensor = torch.tensor(66.0),
    born_scale: float = 1.0,
    born_offset: float = 0.0,
    descreening: float | Tensor = 0.8,
    obc: Tensor = torch.tensor([1.0, 0.8, 4.85]),
) -> Tensor:
    """
    Calculate Born radii for a set of atoms using the Onufriev-Bashford-Case
    model published in

    - A. Onufriev, D. Bashford and D. A. Case, *Proteins: Struct., Funct.,
      Bioinf.*, **2004**, 55, 383–394. DOI: `10.1002/prot.20033
      <https://doi.org/10.1002/prot.20033>`__

    Parameters:
    -----------
    numbers: Tensor, dtype long
        Atomic numbers of the atoms.
    positions: Tensor, dtype float
        Cartesian coordinates of the atoms.
    rvdw: Tensor, dtype float, optional
        Covalent radii of the atoms (default: D3 vdW radii).
    cutoff: Tensor, dtype float, optional
        Real-space cutoff for Born radii integration (default: 66.0 Bohr).
    born_scale: float, optional
        Scaling factor for Born radii (default: 1.0).
    born_offset: float, optional
        Offset for Born radii (default: 0.0).
    descreening: float | Tensor, optional
        Dielectric descreening parameter (default: 0.8).
    obc: Tensor, dtype float, optional
        Onufriev-Bashford-Case parameters (default: [1.0, 0.8, 4.85]).

    Returns:
    --------
    Tensor:
        Born radii for the atoms.

    Raises:
    -------
    ValueError:
        The number of atoms is not equal to the number of positions or the
        number of radii.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    if rvdw is None:
        rvdw = VDW_D3.to(**dd)[numbers]
    if numbers.shape != rvdw.shape:
        raise ValueError(
            f"Shape of covalent radii ({rvdw.shape}) is not consistent with "
            f"atomic numbers ({numbers.shape})."
        )
    if numbers.shape != positions.shape[:-1]:
        raise ValueError(
            f"Shape of positions ({positions.shape[:-1]}) is not consistent "
            f"with atomic numbers ({numbers.shape})."
        )
    if isinstance(descreening, Tensor):
        if numbers.shape != descreening.shape:
            raise ValueError(
                f"Shape of descreening values ({descreening.shape}) is not "
                f"consistent with atomic numbers ({numbers.shape})."
            )

    zero = torch.tensor(0.0, **dd)

    # mask for padding
    mask = real_atoms(numbers)

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
    descreening: float | Tensor = 0.8,
) -> Tensor:
    """Compute dielectric descreening integral I.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of the atoms.
    rvdw : Tensor
        Covalent radii of the atoms.
    cutoff : Tensor, optional
        Real-space cutoff for Born radii integration (default: 66.0 Bohr).
    descreening : float | Tensor, optional
        Dielectric descreening parameter (default: 0.8).

    Returns
    -------
    Tensor
        Dielectric descreening integral I.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    eps = torch.tensor(torch.finfo(positions.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)

    # mask for padding
    mask = real_pairs(numbers, mask_diagonal=True)

    rho = rvdw * descreening

    distances = torch.where(mask, storch.cdist(positions, p=2), eps)
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
    d_pr = torch.where(mask, d_pl * d_mi, eps)  # eps avoids zero division in grad
    d_qu = torch.where(mask, d_mi / d_pl, zero)

    # contributions from non-overlapping atoms
    rho_dpr = torch.where(mask, rho.unsqueeze(-1) / d_pr, zero)
    ln_dqu = torch.where(
        mask * (d_qu > 0), 0.5 * torch.log(torch.where(d_qu > 0, d_qu, eps)) * r1, zero
    )
    non_ovlp = rho_dpr + ln_dqu

    # contributions from overlapping atoms
    rvdw1 = torch.where(mask, 1.0 / rvdw.unsqueeze(-2), zero)
    d_pl1 = 1.0 / d_pl
    dpl_rvdw1 = d_pl * rvdw1
    ln_dpl_rvdw1 = torch.where(
        mask * (dpl_rvdw1 > 0),
        torch.log(torch.where(dpl_rvdw1 > 0, dpl_rvdw1, eps)),
        zero,
    )
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

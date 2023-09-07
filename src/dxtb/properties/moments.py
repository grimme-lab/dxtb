"""
Multipole Moments
=================

Analytical calculation of multipole moments. Currently, dipole moment and
traceless quadrupole moment are implemented. However, this module is serves
more as a short-cut for the calculation in ``Calculator`` as it should hide
some implementation details.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..constants import get_zvalence

__all__ = ["dipole", "quadrupole"]


def dipole(
    charge: Tensor, positions: Tensor, density: Tensor, integral: Tensor
) -> Tensor:
    """
    Analytical calculation of electric dipole moment with electric dipole
    contribution from nuclei (sum_i(r_ik * q_i)) and electrons.

    Parameters
    ----------
    charge : Tensor
        Atom-resolved charges.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    density : Tensor
        Density matrix.
    integral : Tensor
        Dipole integral.

    Returns
    -------
    Tensor
        Electric dipole moment.

    Note
    ----
    This version follows the `tblite` implementation, which employs `r-rj` as
    moment operator and requires the SCC charges for the nuclear dipole
    contribution.
    """
    # TODO: Shape checks

    e_dipole = -torch.einsum("...xij,...ij->...x", integral, density)
    n_dipole = torch.einsum("...ix,...i->...x", positions, charge)
    return n_dipole + e_dipole


def dipole_xtb(
    numbers: Tensor, positions: Tensor, density: Tensor, integral: Tensor
) -> Tensor:
    """
    Analytical calculation of electric dipole moment with electric dipole
    contribution from nuclei (sum_i(r_ik * q_i)) and electrons.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers in the system.
    charge : Tensor
        Atom-resolved charges.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    density : Tensor
        Density matrix.
    integral : Tensor
        Dipole integral.

    Returns
    -------
    Tensor
        Electric dipole moment.

    Note
    ----
    This version follows the `xtb` implementation, where the `r0` moment
    operator is used and the nuclear contribution uses the valence charges.
    """
    # TODO: Shape checks

    # electric component from dipole integral and density matrix
    e_dipole = -torch.einsum("...xij,...ij->...x", integral, density)

    # moment operator "r0" combines with valence charges (xtb implementation)
    n_dipole = torch.einsum(
        "...ix,...i->...x",
        positions,
        get_zvalence(numbers, device=positions.device, dtype=positions.dtype),
    )

    dip = n_dipole + e_dipole
    return dip


def quadrupole(qat: Tensor, dpat: Tensor, qpat: Tensor, positions: Tensor) -> Tensor:
    """
    Analytical calculation of traceless electric quadrupole moment.

    Parameters
    ----------
    qat : Tensor
        Atom-resolved monopolar charges.
    dpat : Tensor
        Atom-resolved dipolar charges.
    qpat : Tensor
        Atom-resolved quadrupolar charges.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).

    Returns
    -------
    Tensor
        Traceless electric quadrupole moment.
    """
    # TODO: Shape checks

    if qpat.shape[-1] == 9:
        # (..., nat, 9) -> (..., nat, 3, 3)
        qpat = qpat.view(*qpat.shape[:-1], 3, 3)

        # trace: (..., nat, 3, 3) -> (..., nat)
        tr = 0.5 * torch.einsum("...ii->...", qpat)

        qpat = torch.stack(
            [
                1.5 * qpat[..., 0, 0] - tr,  # xx
                3 * qpat[..., 1, 0],  # yx
                1.5 * qpat[..., 1, 1] - tr,  # yy
                3 * qpat[..., 2, 0],  # zx
                3 * qpat[..., 2, 1],  # zy
                1.5 * qpat[..., 2, 2] - tr,  # zz
            ],
            dim=-1,
        )

    # This incorporates the electric quadrupole contribution from the
    # nuclei: Q_ij = ∑_k Z_k r_ki r_kj
    vec = torch.einsum("...ij,...i->...ij", positions, qat)

    # temporary
    pv2d = positions * (vec + 2 * dpat)

    # Compute the atomic contributions to molecular quadrupole moment
    cart = torch.empty(
        (*positions.shape[:-1], 6), device=positions.device, dtype=positions.dtype
    )
    cart[..., 0] = pv2d[..., 0]
    cart[..., 1] = (
        positions[..., 0] * (vec[..., 1] + dpat[..., 1])
        + dpat[..., 0] * positions[..., 1]
    )
    cart[..., 2] = pv2d[..., 1]
    cart[..., 3] = (
        positions[..., 0] * (vec[..., 2] + dpat[..., 2])
        + dpat[..., 0] * positions[..., 2]
    )
    cart[..., 4] = (
        positions[..., 1] * (vec[..., 2] + dpat[..., 2])
        + dpat[..., 1] * positions[..., 2]
    )
    cart[..., 5] = pv2d[..., 2]

    # Compute the trace and make the tensor traceless
    tr = 0.5 * (cart[..., 0] + cart[..., 2] + cart[..., 5])
    cart[..., 0] = 1.5 * cart[..., 0] - tr
    cart[..., 1] *= 3.0
    cart[..., 2] = 1.5 * cart[..., 2] - tr
    cart[..., 3] *= 3.0
    cart[..., 4] *= 3.0
    cart[..., 5] = 1.5 * cart[..., 5] - tr

    # sum up contributions
    return qpat.sum(dim=-2) + cart.sum(dim=-2)

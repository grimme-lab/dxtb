"""
Spectra
=======

This module contains the calculation of spectra (IR, Raman).
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..constants import units
from ..utils.grad import _jac

__all__ = ["ir", "raman"]


def ir(
    dipole: Tensor, positions: Tensor, freqs: Tensor, modes: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Calculate IR intensities from nuclear gradient of dipole moment.

    Parameters
    ----------
    dipole : Tensor
        Dipole moment.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    freqs : Tensor
        Vibrational frequencies.
    modes : Tensor
        Normal modes of frequencies.

    Returns
    -------
    tuple[Tensor, Tensor]
        Frequencies and IR intensities.

    Raises
    ------
    RuntimeError
        Position tensor needs `requires_grad=True`.
    """
    # TODO: Frequencies are essentially pass-through args (only here for unit)

    if positions.requires_grad is False:
        raise RuntimeError("Position tensor needs `requires_grad=True`.")

    # derivative of dipole moment w.r.t. positions
    dmu_dr = _jac(dipole, positions)  # (ndim, nat * ndim)
    dmu_dq = torch.matmul(dmu_dr, modes)  # (ndim, nfreqs)
    ir_ints = torch.einsum("...df,...df->...f", dmu_dq, dmu_dq)  # (nfreqs,)

    # print("modes\n", modes)
    # print("dmu_dr\n", dmu_dr)
    # print("")

    print("\nir_ints", ir_ints)
    print(units.AU2KMMOL)
    # print("")
    # print(ir_ints * units.AU2KMMOL)
    # print(ir_ints * 974.8801118351438)
    # print("")

    # TODO: Improve unit handling (maybe extra function?)
    return freqs * units.AU2RCM, ir_ints * 1378999.7790799031


def raman(alpha: Tensor, freqs: Tensor, modes: Tensor) -> tuple[Tensor, Tensor]:
    """
    Calculate the static intensities of Raman spectra.
    Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

    Parameters
    ----------
    alpha : Tensor
        Polarizability tensore (3, 3).
    freqs : Tensor
        Vibrational frequencies.
    modes : Tensor
        Normal modes of frequencies.

    Returns
    -------
    tuple[Tensor, Tensor]
        Frequencies and Raman intensities.
    """
    # TODO: Functionality not tested!
    # https://github.com/fishjojo/pyscfad/blob/main/examples/cc/20-raman.py#L49
    #
    # TODO: Shape checks
    # TODO: Frequencies are essentially pass-through args (only here for unit)
    alphaq = torch.matmul(alpha, modes)  # (ndim, ndim, nmodes)

    # Eq.3 with alpha' = a
    a = torch.einsum("...iij->...j")

    # Eq.4 with (gamma')^2 = g = 0.5 * (g1 + g2 + g3 + g4)
    g1 = (alphaq[0, 0] - alphaq[1, 1]) ** 2
    g2 = (alphaq[0, 0] - alphaq[2, 2]) ** 2
    g3 = (alphaq[2, 2] - alphaq[1, 1]) ** 2
    g4 = alphaq[0, 1] ** 2 + alphaq[1, 2] ** 2 + alphaq[2, 0] ** 2
    g = 0.5 * (g1 + g2 + g3 + g4)

    # Eq.1 (the 1/3 from Eq.3 is squared and reduces the 45)
    raman_ints = 5 * torch.pow(a, 2.0) + 7 * g

    # TODO: Improve unit handling (maybe extra function?)
    return freqs, raman_ints

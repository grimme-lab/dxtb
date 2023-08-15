"""
Vibrational Analysis
====================

This module contains the calculation of vibrational frequencies and the
corresponding normal modes from the mass-weighted Hessian.
"""
from __future__ import annotations

import torch

from .._types import Tensor
from ..constants import get_atomic_masses
from ..utils.geometry import is_linear_molecule
from ..utils.symeig import eighb


def frequencies(
    numbers: Tensor, positions: Tensor, hessian: Tensor, project: bool = True
) -> tuple[Tensor, Tensor]:
    """
    Vibrational frequencies and normal modes from mass-weighted Hessian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system
    hessian : Tensor
        Hessian matrix.
    project : bool, optional
        Whether the 6 (5) lowest frequencies should be removed (project out
        rotational and translational degrees of freedom).

    Returns
    -------
    tuple[Tensor, Tensor]
        Frequencies and normal modes.
    """
    # TODO: Shape checks

    eps = torch.tensor(
        torch.finfo(hessian.dtype).eps,
        device=hessian.device,
        dtype=hessian.dtype,
    )

    hess = (hessian + hessian.transpose(-2, -1).conj()) * 0.5
    masses = get_atomic_masses(
        numbers, atomic_units=True, device=hessian.device, dtype=hessian.dtype
    )
    mass_mat = torch.diag_embed(masses.repeat_interleave(3))

    # instead of calculating and diagonalizing the mass-weighted Hessian,
    # we solve the equivalent general eigenvalue problem Hv=Mve
    evals, evecs = eighb(a=hess, b=mass_mat)

    # Vibrational frequencies ω = √λ with some additional logic for
    # handling possible negative eigenvalues. Note that we are not dividing
    # by 2π (ω = √λ / 2π) in order to immediately get frequencies in
    # Hartree: E = hbar * ω with hbar = 1 in atomic units. Dividing by 2π
    # effectively converts from angular frequency (ω) to the frequency in
    # cycles per second (ν, Hz), which would requires the following
    # conversion to cm^-1: 1e-2 / units.CODATA.c / units.AU2SECOND.
    e = torch.sqrt(torch.clamp(evals, min=eps))
    freqs = e * torch.sign(evals)

    if project is True:
        lin = is_linear_molecule(positions)
        return project_freqs(freqs, modes=evecs, is_linear=lin)

    return freqs, evecs


def project_freqs(
    freqs: Tensor, modes: Tensor, is_linear: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Project out rotational and translational degrees of freedom, i.e., the 5 or
    6 lowest eigenvalues (frequencies).

    Parameters
    ----------
    freqs : Tensor
        Vibrational frequencies
    modes : Tensor
        Normal modes (nat * 3, nfreqs)
    is_linear : bool, optional
        Whether the molecule is linear. Defaults to `False`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Projected frequencies and normal modes
    """
    skip = 5 if is_linear is True else 6
    return freqs[skip:], modes[:, skip:]

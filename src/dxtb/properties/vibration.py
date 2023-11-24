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
        Atomic numbers for all atoms in the system.
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
    mass_mat = torch.diag_embed(masses.repeat_interleave(3, dim=-1))

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
        lin = is_linear_molecule(numbers, positions)
        return project_freqs(freqs, modes=evecs, is_linear=lin)

    return freqs, evecs


def project_freqs(
    freqs: Tensor, modes: Tensor, is_linear: Tensor
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
    is_linear : Tensor
        Whether the molecule is linear.

    Returns
    -------
    tuple[Tensor, Tensor]
        Projected frequencies and normal modes
    """
    skip = torch.where(is_linear, 5, 6)

    # Non-batched version
    if freqs.ndim == 1:
        return freqs[skip:], modes[:, skip:]

    # Batched version
    from ..utils.batch import deflate, pack

    projected_freqs = []
    projected_modes = []
    for i in range(freqs.size(0)):
        skip = 5 if is_linear[i] else 6

        # deflating removes padding that can mess up the shapes
        # Example: LiH and H2O
        # (o: projected out, x: actual value, p: padding)
        #
        # Input tensors (batched):
        # LiH [o, o, o, o, o, x, p, p, p]
        # H2O [o, o, o, o, o, o, x, x, x]
        #
        # After projection we obtain:
        # LiH [x, p, p, p]  (5->end)
        # H2O [x, x, x]     (6->end)
        #
        # The final packing yields an extra dimension:
        # LiH [x, p, p, p]
        # H2O [x, x, x, p] -> increased size by one!
        projected_freqs.append(deflate(freqs[i, skip:]))
        projected_modes.append(deflate(modes[i, :, skip:]))

    return pack(projected_freqs), pack(projected_modes)

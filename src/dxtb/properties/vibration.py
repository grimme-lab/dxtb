"""
Vibrational Analysis
====================

This module contains the calculation of vibrational frequencies and the
corresponding normal modes from the mass-weighted Hessian.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.data.mass import ATOMIC as ATOMIC_MASSES

from .._types import Tensor
from ..constants import get_atomic_masses
from ..utils.geometry import is_linear_molecule, mass_center
from ..utils.math import eigh, einsum, qr
from ..utils.symeig import eighb

LINDEP_THRESHOLD = 1e-7


def _get_translational_modes(mass: Tensor):
    """Translational modes"""
    massp = storch.sqrt(mass)
    Tx = einsum("...m,x->...mx", massp, torch.tensor([1, 0, 0]))
    Ty = einsum("...m,y->...my", massp, torch.tensor([0, 1, 0]))
    Tz = einsum("...m,z->...mz", massp, torch.tensor([0, 0, 1]))
    return Tx.ravel(), Ty.ravel(), Tz.ravel()


def _get_rotational_modes(mass: Tensor, mpos: Tensor):
    # Computation of inertia tensor and transformation to its traceless
    # form (convenient for calculating the principal axes of rotation).
    im = einsum("...m,...mx,...my->...xy", mass, mpos, mpos)
    im = torch.eye(3) * im.trace() - im

    # Eigendecomposition yields the principal moments of inertia (w)
    # and the principal axes of rotation (paxes) of a molecule.
    w, paxes = torch.linalg.eigh(im)

    # make z-axis rotation vector with smallest moment of inertia
    w = torch.flip(w, [0])
    paxes = torch.flip(paxes, [1])
    ex, ey, ez = paxes.mT

    # rotational mode
    coords_rot_frame = mpos @ paxes  # einsum("...ij,...jk->...ik", mpos, paxes)
    cx, cy, cz = coords_rot_frame.mT

    massp = storch.sqrt(mass)
    _massp = massp[..., :, None]
    _cx = cx[..., :, None]
    _cy = cy[..., :, None]
    _cz = cz[..., :, None]

    Rx = _massp * (_cy * ez - _cz * ey)
    Ry = _massp * (_cz * ex - _cx * ez)
    Rz = _massp * (_cx * ey - _cy * ex)
    return Rx.ravel(), Ry.ravel(), Rz.ravel()


def frequencies(
    numbers: Tensor,
    positions: Tensor,
    hessian: Tensor,
    project_translational: bool = True,
    project_rotational: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Vibrational frequencies and normal modes from mass-weighted Hessian.

    http://gaussian.com/vib/
    https://github.com/psi4/psi4/blob/master/psi4/driver/qcdb/vib.py
    https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py

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
        Frequencies (n_modes) and normal modes (3*n_at, n_modes).
    """
    # TODO: Shape checks

    mass = ATOMIC_MASSES.to(device=hessian.device, dtype=hessian.dtype)[numbers]
    invsqrtmass = storch.reciprocal(storch.sqrt(mass))

    # (nb, nat, 3, nat, 3) * (nb, nat) * (nb, nat) -> (nb, nat, 3, nat, 3)
    mhess = einsum("...pxqy,...p,...q->...pxqy", hessian, invsqrtmass, invsqrtmass)

    # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
    hess = mhess.reshape(*[*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]])

    # symmetrize
    h = (hess + hess.transpose(-2, -1)) * 0.5

    # TODO: More sophisticated checks for atom
    # TODO: Test batch
    TRspace = []
    if project_translational is True and numbers.shape[-1] > 1:
        TRspace.extend(_get_translational_modes(mass))

    if project_rotational is True and numbers.shape[-1] > 1:
        mpos = positions - mass_center(mass, positions)

        if (is_linear_molecule(numbers, positions) == True).all():
            TRspace.extend(_get_rotational_modes(mass, mpos)[:-1])
        else:
            TRspace.extend(_get_rotational_modes(mass, mpos))

    if len(TRspace) > 0:
        TRspace = torch.vstack(TRspace)

        # create orthogonal basis (q) for the modes
        q, _ = qr(TRspace.T)

        # The special projection matrix P=I-QQ^T is crucial for isolating
        # specific subspaces of interest in a high-dimensional space by
        # projecting vectors onto the subspace orthogonal to the column space
        # of QQ. In the context of vibrational analysis, the projection matrix
        # P is used to remove translational and rotational modes from
        # consideration, focusing the analysis on the true vibrational modes of
        # a molecular system.
        qqT = q @ q.mT  # einsum("...ij,...kj->...ik", q, q)
        P = torch.eye(*[3 * numbers.shape[-1]]) - qqT
        w, v = eigh(P)
        bvec = v[..., :, w > LINDEP_THRESHOLD]

        if bvec.shape[-1] == 0:
            raise RuntimeError(
                "The projection matrix for transformation to internal "
                f"coordinates appears to be empty (shape: {bvec.shape}) "
                "This is either caused by linear dependencies in the "
                "projection matrix or faulty geometry detection."
            )

        # transform Hessian in mass-weighted cartesian coordinates (MWC) to new
        # Hessian in internal coordinates (INT): h_INT = bvec.T @ h_MVC @ bvec
        h = bvec.mT @ h @ bvec  # einsum("...ji,...jk,...kl->...il", b, h, b)

        # eigendecomposition of Hessian yields force constants (not
        # frequencies!) and normal modes of vibration
        force_const_au, _mode = eigh(h)
        mode = bvec @ _mode
    else:
        force_const_au, mode = eigh(h)

    # Vibrational frequencies ω = √λ with some additional logic for
    # handling possible negative eigenvalues. Note that we are not dividing
    # by 2π (ω = √λ / 2π) in order to immediately get frequencies in
    # Hartree: E = hbar * ω with hbar = 1 in atomic units. Dividing by 2π
    # effectively converts from angular frequency (ω) to the frequency in
    # cycles per second (ν, Hz), which would requires the following
    # conversion to cm^-1: 1e-2 / units.CODATA.c / units.AU2SECOND.
    sgn = torch.sign(force_const_au)
    freqs_au = torch.sqrt(torch.abs(force_const_au)) * sgn

    return freqs_au, mode


def frequencies_old(
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

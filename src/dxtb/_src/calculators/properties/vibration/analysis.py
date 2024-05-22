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
Vibrational Analysis: Frequencies
=================================

This module contains the calculation of vibrational frequencies and the
corresponding normal modes from the mass-weighted Hessian.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.data.mass import ATOMIC as ATOMIC_MASSES
from tad_mctc.math import einsum
from tad_mctc.molecule.geometry import is_linear
from tad_mctc.molecule.property import inertia_moment, positions_rel_com

from dxtb._src.typing import Any, Literal, NoReturn, Tensor
from dxtb._src.utils.math import qr

from .result import BaseResult

__all__ = ["VibResult", "vib_analysis"]


LINDEP_THRESHOLD = 1e-7


class VibResult(BaseResult):
    """
    Data from the vibrational analysis.

    - Vibrational frequencies.
    - Normal modes.
    """

    __slots__ = ["_modes", "_modes_unit"]

    def __init__(
        self,
        freqs: Tensor,
        modes: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the vibrational analysis result.

        Parameters
        ----------
        freqs : Tensor
            Vibrational frequencies in atomic units.
        modes : Tensor
            Normal modes (unitless).
        device : torch.device | None, optional
            Device of the tensors. If ``None``, the device of `freqs` is used.
            Defaults to ``None``.
        dtype : torch.dtype | None, optional
            Data type of the tensors. If ``None``, the data type of `freqs` is
            used. Defaults to ``None``.
        """
        super().__init__(
            freqs=freqs,
            device=device if device is not None else freqs.device,
            dtype=dtype if dtype is not None else freqs.dtype,
        )

        self._modes = modes
        self._modes_unit = None

    # intensities

    @property
    def modes(self) -> Tensor:
        return self._modes

    @modes.setter
    def modes(self, *_: Any) -> NoReturn:
        raise RuntimeError("Setting normal modes is not supported.")

    @property
    def modes_unit(self) -> None:
        return self._modes_unit

    @modes_unit.setter
    def modes_unit(self, *_: Any) -> None:
        raise RuntimeError("Normal modes are unitless.")

    # conversion

    def to_unit(self, value: Literal["freqs", "modes"], unit: str) -> Tensor:
        """
        Convert a value from one unit to another based on the converter
        dictionary.
        """
        if value == "freqs":
            return self._convert(self.freqs, unit, self.converter_freqs)

        # if value == "modes":
        #   return self._convert(self.modes, unit, self.converter_modes)

        raise ValueError(f"Unsupported value for conversion: {value}")

    def use_common_units(self) -> None:
        """
        Convert the frequencies and intensities to common units, that is,
        `cm^-1` for frequencies.
        """
        self.freqs_unit = "cm^-1"


def _get_translational_modes(mass: Tensor):
    """Translational modes"""
    massp = storch.sqrt(mass)
    Tx = einsum("...m,x->...mx", massp, torch.tensor([1, 0, 0]))
    Ty = einsum("...m,y->...my", massp, torch.tensor([0, 1, 0]))
    Tz = einsum("...m,z->...mz", massp, torch.tensor([0, 0, 1]))
    return Tx.ravel(), Ty.ravel(), Tz.ravel()


def _get_rotational_modes(mass: Tensor, positions: Tensor):
    mpos = positions_rel_com(mass, positions)
    im = inertia_moment(mass, mpos, pos_already_com=True)

    # Eigendecomposition yields the principal moments of inertia (w)
    # and the principal axes of rotation (paxes) of a molecule.
    w, paxes = storch.eighb(im)

    # make z-axis rotation vector with smallest moment of inertia
    w = torch.flip(w, [-1])
    paxes = torch.flip(paxes, [-1])
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


def vib_analysis(
    numbers: Tensor,
    positions: Tensor,
    hessian: Tensor,
    project_translational: bool = True,
    project_rotational: bool = True,
) -> VibResult:
    """
    Vibrational analysis yielding frequencies and normal modes from
    mass-weighted Hessian.

    - http://gaussian.com/vib/
    - https://github.com/psi4/psi4/blob/master/psi4/driver/qcdb/vib.py
    - https://github.com/pyscf/pyscf/blob/master/pyscf/hessian/thermo.py

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms in the system ``(..., nat, 3)``.
    hessian : Tensor
        Hessian matrix of shape ``(..., nat, 3, nat, 3)``.
    project_rotational : bool, optional
        Whether to project out rotational degrees of freedom.
    project_translational : bool, optional
        Whether to project out translational degrees of freedom.

    Returns
    -------
    tuple[Tensor, Tensor]
        Frequencies of shape ``(..., nfreqs)`` and normal modes of shape
        ``(..., 3*nat, nfreqs)``.
    """
    nat = numbers.shape[-1]
    if hessian.shape == (*numbers.shape[:-1], nat, 3, nat, 3):
        # reshape (nb, nat, 3, nat, 3) to (nb, nat*3, nat*3)
        hessian = hessian.reshape(*[*numbers.shape[:-1], *2 * [3 * nat]])

        # Mass-weighting without reshaping:
        # (nb, nat, 3, nat, 3) * (nb, nat) * (nb, nat) -> (nb, nat, 3, nat, 3)
        # mhess = einsum("...pxqy,...p,...q->...pxqy", hessian, ism, ism)

    elif hessian.shape == (*numbers.shape[:-1], nat * 3, nat * 3):
        raise ValueError(
            f"The Hessian matrix must have either shape (..., nat, 3, nat, 3) "
            "or (..., nat*3, nat*3) for a system with `nat` atoms. The shape "
            f"of the Hessian matrix is {hessian.shape}, while the shape of the "
            f"atomic numbers is {numbers.shape}."
        )

    mass = ATOMIC_MASSES.to(device=hessian.device, dtype=hessian.dtype)[numbers]

    # 1/sqrt(m) of shape (..., nat) -> (..., nat*3)
    invsqrtmass = torch.repeat_interleave(
        storch.reciprocal(storch.sqrt(mass)), 3, dim=-1
    )

    # mass-weighted Hessian
    mhess = einsum("...p,...pq,...q->...pq", invsqrtmass, hessian, invsqrtmass)

    # symmetrize
    h = (mhess + mhess.transpose(-2, -1)) * 0.5

    # TODO: More sophisticated checks for atom
    # TODO: Test batch
    TRspace = []
    if project_translational is True and numbers.shape[-1] > 1:
        TRspace.extend(_get_translational_modes(mass))

    if project_rotational is True and numbers.shape[-1] > 1:
        if (is_linear(numbers, positions) == True).all():
            TRspace.extend(_get_rotational_modes(mass, positions)[:-1])
        else:
            TRspace.extend(_get_rotational_modes(mass, positions))

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
        w, v = storch.eighb(P)
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
        force_const_au, _mode = storch.eighb(h)
        mode = bvec @ _mode
    else:
        force_const_au, mode = storch.eighb(h)

        # Instead of calculating and diagonalizing the mass-weighted Hessian,
        # one could also solve the equivalent general eigenvalue problem Hv=Mve
        # This also directly un-mass-weights the normal modes.
        #
        # mass_mat = torch.diag_embed(masses.repeat_interleave(3, dim=-1))
        # evals, evecs = eighb(a=h, b=mass_mat)

    # Vibrational frequencies ω = √λ with some additional logic for
    # handling possible negative eigenvalues. Note that we are not dividing
    # by 2π (ω = √λ / 2π) in order to immediately get frequencies in
    # Hartree: E = hbar * ω with hbar = 1 in atomic units. Dividing by 2π
    # effectively converts from angular frequency (ω) to the frequency in
    # cycles per second (ν, Hz), which would requires the following
    # conversion to cm^-1: 1e-2 / units.CODATA.c / units.AU2SECOND.
    sgn = torch.sign(force_const_au)
    freqs_au = torch.sqrt(torch.abs(force_const_au)) * sgn

    # un-mass-weight the normal modes
    mode_au = einsum("...i,...ij->...ij", invsqrtmass, mode)

    return VibResult(freqs_au, mode_au)


# TODO: Remove this function after checking batched version
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
    from tad_mctc.batch import deflate, pack

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

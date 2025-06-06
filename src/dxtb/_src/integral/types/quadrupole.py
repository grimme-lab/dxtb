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
Integral Types: Quadrupole
==========================

Quadrupole integral.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Literal, Tensor

from ..base import BaseIntegral

__all__ = ["QuadrupoleIntegral"]


class QuadrupoleIntegral(BaseIntegral):
    """
    Quadrupole integral from atomic orbitals.
    """

    def reduce_9_to_6(self, uplo: Literal["u", "U", "l", "L"] = "l") -> Tensor:
        """
        Remove symmetry-equivalent elements from the quadrupole integral,
        i.e., only the lower/upper triangular matrix is kept.

        Lower triangular matrix:
        xx xy xz       0 1 2      0
        yx yy yz  <=>  3 4 5  ->  3 4
        zx zy zz       6 7 8      6 7 8

        Upper triangular matrix:
        xx xy xz       0 1 2      0 1 2
        yx yy yz  <=>  3 4 5  ->    4 5
        zx zy zz       6 7 8          8

        Parameters
        ----------
        uplo : Literal["u", "U", "l", "L"], optional
            Upper or lower triangular matrix, by default "l".

        Returns
        -------
        Tensor
            Reduced quadrupole integral of shape ``(..., 6, n, n)``.
        """
        self.matrix = _reduce_9_to_6(self.matrix, uplo=uplo)
        return self.matrix

    def traceless(self) -> Tensor:
        """
        Make a quadrupole (integral) tensor traceless.

        Parameters
        ----------
        qpint : Tensor
            Quadrupole moment tensor of shape ``(..., 6, n, n)``.

        Returns
        -------
        Tensor
            Traceless Quadrupole moment tensor of shape
            ``(..., 6, n, n)``.

        Raises
        ------
        RuntimeError
            Supplied quadrupole integral is no ``6xNxN`` tensor.
        """

        if self.matrix.ndim not in (3, 4) or self.matrix.shape[-3] != 6:
            raise RuntimeError(
                "Quadrupole integral must be a tensor tensor of shape "
                f"'(6, norb, norb)' but is {self.matrix.shape}."
            )

        tr = 0.5 * (
            self.matrix[..., 0, :, :]
            + self.matrix[..., 2, :, :]
            + self.matrix[..., 5, :, :]
        )

        self.matrix = torch.stack(
            [
                1.5 * self.matrix[..., 0, :, :] - tr,  # xx
                1.5 * self.matrix[..., 1, :, :],  ###### yx
                1.5 * self.matrix[..., 2, :, :] - tr,  # yy
                1.5 * self.matrix[..., 3, :, :],  ###### zx
                1.5 * self.matrix[..., 4, :, :],  ###### zy
                1.5 * self.matrix[..., 5, :, :] - tr,  # zz
            ],
            dim=-3,
        )
        return self.matrix

    def shift_r0r0_rjrj(
        self,
        r0: Tensor,
        overlap: Tensor,
        pos: Tensor,
        uplo: Literal["u", "U", "l", "L"] = "l",
    ) -> Tensor:
        r"""
        Shift the centering of the quadrupole integral (moment operator) from
        the origin (:math:`r0 = r - (0, 0, 0)`) to atoms (ket index,
        :math:`rj = r - r_j`).

        The output tensor will always be of shape ``(..., 6, n, n)``, i.e., the
        quadrupole integral is always reduced to the lower/upper triangular
        matrix if not already done.

        We start with the quadrupole integral generated by the ``r0`` moment
        operator:

        .. math::

            Q_{xx}^{r0} = \langle i | (r_x - r0)^2 | j \rangle = \langle i | r_x^2 | j \rangle

        Now, we shift the integral to ``r_j`` yielding the quadrupole integral
        center on the respective atoms:

        .. math::

            \begin{align}
                Q_{xx} &= \langle i | (r_x - r_{xj})^2 | j \rangle \\
                &= \langle i | r_x^2 | j \rangle - 2 \langle i | r_{xj} r_x | j \rangle + \langle i | r_{xj}^2 | j \rangle \\
                &= Q_{xx}^{r0} - 2 r_{xj} \langle i | r_x | j \rangle + r_{xj}^2 \langle i | j \rangle \\
                &= Q_{xx}^{r0} - 2 r_{xj} D_{x}^{r0} + r_{xj}^2 S_{ij}
            \end{align}

        Next, we create the shift contribution for all off-diagonal elements of
        the quadrupole integral.

        .. math::

            \begin{align}
                Q_{ab} &= \langle i | (r_a - r_{aj})(r_b - r_{bj}) | j \rangle \\
                &= \langle i | r_a r_b | j \rangle - \langle i | r_a r_{bj} | j \rangle - \langle i | r_{aj} r_b | j \rangle + \langle i | r_{aj} r_{bj} | j \rangle \\
                &= Q_{ab}^{r0} - r_{bj} \langle i | r_a | j \rangle - r_{aj} \langle i | r_b | j \rangle + r_{aj} r_{bj} \langle i | j \rangle \\
                &= Q_{ab}^{r0} - r_{bj} D_a^{r0} - r_{aj} D_b^{r0} + r_{aj} r_{bj} S_{ij}
            \end{align}

        Parameters
        ----------
        r0 : Tensor
            Origin-centered dipole integral.
        overlap : Tensor
            Monopole integral (overlap).
        pos : Tensor
            Orbital-resolved atomic positions.
        uplo : Literal["u", "U", "l", "L"], optional
            If the symmetry-equivalent elements have not been removed yet,
            this method will always do so by calling :meth:`.reduce_9_to_6`.
            With this parameter, you can specify whether the matrix is upper
            or lower triangular, by default "l".

        Returns
        -------
        Tensor
            Second-index (ket) atom-centered quadrupole integral of shape
            ``(..., 6, n, n)``.

        Raises
        ------
        RuntimeError
            Shape mismatch between ``positions`` and ``overlap``.
            The positions must be orbital-resolved.
        RuntimeError
            Quadrupole integral is no ``9xNxN`` or ``6xNxN`` tensor.
        """
        if pos.shape[-2] != overlap.shape[-1]:
            raise RuntimeError(
                "Shape mismatch between positions and overlap integral. "
                "The position tensor must be spread to orbital-resolution."
                "Use the `IndexHelper` to spread the positions: "
                "ihelp.spread_atom_to_orbital(positions, dim=-2, extra=True)"
            )

        if self.matrix.shape[-3] != 6:
            if self.matrix.shape[-3] != 9:
                raise RuntimeError(
                    "Quadrupole integral must be a tensor of shape "
                    "'(6, norb, norb)' or '(9, norb, norb)' but is "
                    f"{self.matrix.shape}."
                )
            self.reduce_9_to_6(uplo=uplo)

        # cartesian components for convenience
        x = pos[..., 0]
        y = pos[..., 1]
        z = pos[..., 2]
        dpx = r0[..., 0, :, :]
        dpy = r0[..., 1, :, :]
        dpz = r0[..., 2, :, :]

        # construct shift contribution from dipole and monopole (overlap) moments
        shift_xx = _shift_diagonal(x, dpx, overlap)
        shift_yy = _shift_diagonal(y, dpy, overlap)
        shift_zz = _shift_diagonal(z, dpz, overlap)
        shift_yx = _shift_offdiag(y, x, dpy, dpx, overlap)
        shift_zx = _shift_offdiag(z, x, dpz, dpx, overlap)
        shift_zy = _shift_offdiag(z, y, dpz, dpy, overlap)

        self.matrix = torch.stack(
            [
                self.matrix[..., 0, :, :] + shift_xx,  # xx
                self.matrix[..., 1, :, :] + shift_yx,  # yx
                self.matrix[..., 2, :, :] + shift_yy,  # yy
                self.matrix[..., 3, :, :] + shift_zx,  # zx
                self.matrix[..., 4, :, :] + shift_zy,  # zy
                self.matrix[..., 5, :, :] + shift_zz,  # zz
            ],
            dim=-3,
        )
        return self.matrix


def _reduce_9_to_6(
    qpint: Tensor, uplo: Literal["u", "U", "l", "L"] = "l"
) -> Tensor:
    """
    Remove symmetry-equivalent elements from the quadrupole integral,
    i.e., only the lower/upper triangular matrix is kept.

    Lower triangular matrix:
    xx xy xz       0 1 2      0
    yx yy yz  <=>  3 4 5  ->  3 4
    zx zy zz       6 7 8      6 7 8

    Upper triangular matrix:
    xx xy xz       0 1 2      0 1 2
    yx yy yz  <=>  3 4 5  ->    4 5
    zx zy zz       6 7 8          8

    Parameters
    ----------
    qpint : Tensor
        Quadrupole integral of shape ``(..., 9, n, n)``.
    uplo : Literal["u", "U", "l", "L"], optional
        Upper or lower triangular matrix, by default "l".

    Returns
    -------
    Tensor
        Reduced quadrupole integral of shape ``(..., 6, n, n)``.

    Raises
    ------
    RuntimeError
        Quadrupole integral is no ``9xNxN`` tensor.
    ValueError
        Invalid value for optional `uplo` parameter.
    """
    if qpint.ndim in (3, 4) and qpint.shape[-3] != 9:
        raise RuntimeError(
            "Quadrupole integral must be a tensor tensor of shape "
            f"'(..., 9, norb, norb)' but is '{qpint.shape}'."
        )

    if uplo in ("u", "U"):
        return torch.stack(
            [
                qpint[..., 0, :, :],  # xx
                qpint[..., 1, :, :],  # xy
                qpint[..., 2, :, :],  # xz
                qpint[..., 4, :, :],  # yy
                qpint[..., 5, :, :],  # yz
                qpint[..., 8, :, :],  # zz
            ],
            dim=-3,
        )

    if uplo in ("l", "L"):
        return torch.stack(
            [
                qpint[..., 0, :, :],  # xx
                qpint[..., 3, :, :],  # yx
                qpint[..., 4, :, :],  # yy
                qpint[..., 6, :, :],  # zx
                qpint[..., 7, :, :],  # zy
                qpint[..., 8, :, :],  # zz
            ],
            dim=-3,
        )

    raise ValueError(f"Invalid value for `uplo`: {uplo}")


def _shift_diagonal(c: Tensor, dpc: Tensor, s: Tensor) -> Tensor:
    r"""
    Create the shift contribution for all diagonal elements of the quadrupole
    integral.

    We start with the quadrupole integral generated by the ``r0`` moment
    operator:

    .. math::

        Q_{xx}^{r0} = \langle i | (r_x - r0)^2 | j \rangle = \langle i | r_x^2 | j \rangle

    Now, we shift the integral to ``r_j`` yielding the quadrupole integral
    center on the respective atoms:

    .. math::

        \begin{align}
            Q_{xx} &= \langle i | (r_x - r_{xj})^2 | j \rangle \\
            &= \langle i | r_x^2 | j \rangle - 2 \langle i | r_{xj} r_x | j \rangle + \langle i | r_{xj}^2 | j \rangle \\
            &= Q_{xx}^{r0} - 2 r_{xj} \langle i | r_x | j \rangle + r_{xj}^2 \langle i | j \rangle \\
            &= Q_{xx}^{r0} - 2 r_{xj} D_{x}^{r0} + r_{xj}^2 S_{ij}
        \end{align}

    Parameters
    ----------
    c : Tensor
        Cartesian component.
    dpc : Tensor
        Cartesian component of dipole integral (`r0` operator).
    s : Tensor
        Overlap integral.

    Returns
    -------
    Tensor
        Shift contribution for diagonals of quadrupole integral.
    """
    shift_1 = -2 * einsum("...j,...ij->...ij", c, dpc)
    shift_2 = einsum("...j,...j,...ij->...ij", c, c, s)
    return shift_1 + shift_2


def _shift_offdiag(
    a: Tensor, b: Tensor, dpa: Tensor, dpb: Tensor, s: Tensor
) -> Tensor:
    r"""
    Create the shift contribution for all off-diagonal elements of the
    quadrupole integral.

    .. math::

        \begin{align}
            Q_{ab} &= \langle i | (r_a - r_{aj})(r_b - r_{bj}) | j \rangle \\
            &= \langle i | r_a r_b | j \rangle - \langle i | r_a r_{bj} | j \rangle - \langle i | r_{aj} r_b | j \rangle + \langle i | r_{aj} r_{bj} | j \rangle \\
            &= Q_{ab}^{r0} - r_{bj} \langle i | r_a | j \rangle - r_{aj} \langle i | r_b | j \rangle + r_{aj} r_{bj} \langle i | j \rangle \\
            &= Q_{ab}^{r0} - r_{bj} D_a^{r0} - r_{aj} D_b^{r0} + r_{aj} r_{bj} S_{ij}
        \end{align}

    Parameters
    ----------
    a : Tensor
        First cartesian component.
    b : Tensor
        Second cartesian component.
    dpa : Tensor
        First cartesian component of dipole integral (r0 operator).
    dpb : Tensor
        Second cartesian component of dipole integral (r0 operator).
    s : Tensor
        Overlap integral.

    Returns
    -------
    Tensor
        Shift contribution of off-diagonal elements of quadrupole integral.
    """
    shift_ab_1 = -einsum("...j,...ij->...ij", b, dpa)
    shift_ab_2 = -einsum("...j,...ij->...ij", a, dpb)
    shift_ab_3 = einsum("...j,...j,...ij->...ij", a, b, s)

    return shift_ab_1 + shift_ab_2 + shift_ab_3

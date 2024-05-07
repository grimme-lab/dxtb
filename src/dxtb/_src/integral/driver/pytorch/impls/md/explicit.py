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
Explicit McMurchie-Davidson
===========================

Calculation of overlap integrals using the McMurchie-Davidson algorithm.

- L. E. McMurchie, E. R. Davidson, One- and two-electron integrals over
  cartesian gaussian functions, *J. Comput. Phys.*, **1978**, *26*, 218-231.
  (`DOI <https://doi.org/10.1016/0021-9991(78)90092-X>`__)

Here, the E-coefficients are explicitly written down to avoid custom autograd
functions.

For the gradients, check out the following papers.

- T. Helgaker, P. R. Taylor, On the evaluation of derivatives of Gaussian
  integrals, *Theor. Chim. Acta*, **1992**, *83*, 177.
  (`DOI <https://doi.org/10.1007/BF01132826>`__)
- K. Doll, V. R. Saunders, N. M. Harrison, Analytical Hartree–Fock gradients
  for periodic systems, *Int. J. Quantum Chem.*, **2001** *82*, 1-13.
  (`DOI <https://doi.org/10.1002/1097-461X(2001)82:1%3C1::AID-QUA1017%3E3.0.CO;2-W>`__)
"""

from __future__ import annotations

from functools import partial
from math import pi, sqrt

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import (
    CGTOAzimuthalQuantumNumberError,
    IntegralTransformError,
)

from .trafo import NLM_CART, TRAFO

__all__ = ["md_explicit", "md_explicit_gradient"]

sqrtpi3 = sqrt(pi) ** 3


def md_explicit(
    angular: tuple[Tensor, Tensor],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Calculate overlap integrals using (explicit) McMurchie-Davidson algorithm.

    Parameters
    ----------
    angular : (Tensor, Tensor)
        Angular momentum of the shell pair(s).
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents of the shell pair(s).
    coeff : (Tensor, Tensor)
        Contraction coefficients of the shell pair(s).
    vec : Tensor
        Displacement vector between shell pair(s) of shape (nvec, 3).

    Returns
    -------
    Tensor
        Overlap integrals for shell pair(s).
    """
    li, lj = angular

    try:
        itrafo = TRAFO[li].type(vec.dtype).to(vec.device)
        jtrafo = TRAFO[lj].type(vec.dtype).to(vec.device)
    except IndexError as e:
        raise IntegralTransformError() from e

    # expontents and related variables required for integral
    ai, aj = alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-2)
    ci, cj = coeff[0].unsqueeze(-1), coeff[1].unsqueeze(-2)
    eij = ai + aj
    oij = 1.0 / eij
    xij = 0.5 * oij

    # p * (R_A - R_B)² with p = a*b/(a+b)
    r2 = einsum("...i,...i->...", vec, vec)  # vec.pow(2).sum(-1)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-2)

    # K_AB * [Gaussian integral (√(pi/(a+b))) in 3D] * c_A * c_B
    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    # ss does not require E-coefficients (e000 = 1)
    if li == 0 and lj == 0:
        s3d = sij.sum((-2, -1), keepdim=True)
    else:
        rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
        rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

        # e0: (li+1, lj+1, nat, 3, ai, aj)
        if li == 0:
            e0 = ecoeffs_s(lj, xij, rpi, rpj)
        elif li == 1:
            e0 = ecoeffs_p(lj, xij, rpi, rpj)
        elif li == 2:
            e0 = ecoeffs_d(lj, xij, rpi, rpj)
        elif li == 3:
            e0 = ecoeffs_f(lj, xij, rpi, rpj)
        else:
            raise CGTOAzimuthalQuantumNumberError(li)

        nlmi = NLM_CART[li]
        nlmj = NLM_CART[lj]

        # Collect E-coefficients for each cartesian direction for first (i)
        # center. Getting the E-coefficients for the three directions from
        # `NLM_CART` replaces the first dimension with the number of
        # cartesian basis functions of the first orbital (i), which finally
        # yields the following shape: (ncarti, lj+1, nbatch, 3 ai, aj)
        e0x = e0[nlmi[:, 0]]
        e0y = e0[nlmi[:, 1]]
        e0z = e0[nlmi[:, 2]]

        # Collect E-coefficients for each cartesian direction for second (j)
        # center. Getting the E-coefficients for the three directions from
        # `NLM_CART` replaces the second dimension with the number of
        # cartesian basis functions of the second orbital (j). Additionally,
        # we selecting the cartesian directions eliminating the `-3`rd
        # dimension, which ultimately yields the following shape:
        # (ncarti, ncartj, nbatch, ai, aj)
        sx = e0x[:, nlmj[:, 0], ..., 0, :, :]  # type: ignore
        sy = e0y[:, nlmj[:, 1], ..., 1, :, :]  # type: ignore
        sz = e0z[:, nlmj[:, 2], ..., 2, :, :]  # type: ignore

        # First, we multiply sx, sy and sz with sij using the inidces a (ai)
        # and b (aj). Then, we sum over the alphas (a and b), reducing the
        # tensor to (ncarti, ncartj, nbatch). Since the batch dimension is
        # conventionally the first dimension, we cycle the indices and obtain
        # the final shape: (nbatch, ncarti, ncartj)
        s3d = einsum(
            "ij...ab,ij...ab,ij...ab,...ab->...ij",
            *(sx, sy, sz, sij),
            optimize=[(0, 1), (2, 0), (0, 1)],  # fastest contraction path
        )

        # OLD: This is the loop-based version of the above indexing atrocities.
        # I left it here, as it may be better to understand...
        #
        # for mli in range(ncarti):
        #     mi = nlmi[mli, :]
        #     _e0x = e0[mi[0]]
        #     _e0y = e0[mi[1]]
        #     _e0z = e0[mi[2]]
        #
        #     for mlj in range(ncartj):
        #         mj = nlmj[mlj, :]
        #         x = _e0x[mj[0]][..., 0, :, :]
        #         y = _e0y[mj[1]][..., 1, :, :]
        #         z = _e0z[mj[2]][..., 2, :, :]
        #
        #         s3d[..., mli, mlj] += (sij * x * y * z).sum((-2, -1))

    # transform to cartesian basis functions (itrafo * S * jtrafo^T)
    o = itrafo @ s3d @ jtrafo.mT  # einsum("...ij,...jk,...lk->...il", i, s, j)

    # Previously, I removed small values for numerical stability of the SCF
    # (and some portions are also faster) by using the following expression
    # `torch.where(torch.abs(o) < eps, eps, o)`. This, however, leads to wrong
    # gradients (gradcheck fails), because the gradient of `abs()` is not
    # defined for zero. A related issue concerning `abs()` was already raised at
    # https://github.com/pytorch/pytorch/issues/7172
    return o


def md_explicit_gradient(
    angular: tuple[Tensor, Tensor],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    r"""
    Overlap gradient of two orbitals.

    .. math::

        F^{i,j}_t &= 2 a E^{i+1,j}_t - iE^{i-1,j}_t \\
        F^{i,j}_t &= 2 b E^{i,j+1}_t - jE^{i,j-1}_t

    Parameters
    ----------
    angular : tuple[Tensor, Tensor]
        Angular momenta of orbital i and j.
    alpha : tuple[Tensor, Tensor]
        Exponents of GTOs of orbital i and j.
    coeff : tuple[Tensor, Tensor]
        Coefficients in contracted GTOs of orbital i and j.
    vec : Tensor
        Distance vectors for all unique overlap pairs.

    Returns
    -------
    Tensor
        Overlap gradient of shape `(nbatch, 3, ncarti, ncartj)`.

    Raises
    ------
    IntegralTransformError, CGTOAzimuthalQuantumNumberError
        Unsupported angular momentum.
    """
    li, lj = angular

    try:
        itrafo = TRAFO[li].type(vec.dtype).to(vec.device)
        jtrafo = TRAFO[lj].type(vec.dtype).to(vec.device)
    except IndexError as e:
        raise IntegralTransformError() from e

    ai, aj = alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-2)
    ci, cj = coeff[0].unsqueeze(-1), coeff[1].unsqueeze(-2)
    eij = ai + aj
    oij = 1.0 / eij
    xij = 0.5 * oij

    # p * (R_A - R_B)² with p = a*b/(a+b)
    r2 = einsum("...i,...i->...", vec, vec)  # vec.pow(2).sum(-1)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-2)

    # K_AB * Gaussian integral (√(pi/(a+b))) in 3D * c_A * c_B
    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
    rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

    nlmi = NLM_CART[li]
    nlmj = NLM_CART[lj]

    # calc E function for all (ai, aj)-combis for all vecs in batch
    if li == 0:
        e0, d0 = de_s(lj, xij, rpi, rpj, ai, aj)
    elif li == 1:
        e0, d0 = de_p(lj, xij, rpi, rpj, ai, aj)
    elif li == 2:
        e0, d0 = de_d(lj, xij, rpi, rpj, ai, aj)
    elif li == 3:
        e0, d0 = de_f(lj, xij, rpi, rpj, ai, aj)
    else:
        raise CGTOAzimuthalQuantumNumberError(li)

    # Collect E-coefficients for each cartesian direction for first (i)
    # center. Getting the E-coefficients for the three directions from
    # `NLM_CART` replaces the first dimension with the number of
    # cartesian basis functions of the first orbital (i), which finally
    # yields the following shape: (ncarti, lj+1, nbatch, 3, ai, aj)
    e0x = e0[nlmi[:, 0]]
    e0y = e0[nlmi[:, 1]]
    e0z = e0[nlmi[:, 2]]

    d0x = d0[nlmi[:, 0]]
    d0y = d0[nlmi[:, 1]]
    d0z = d0[nlmi[:, 2]]

    # Collect E-coefficients for each cartesian direction for second (j)
    # center. Getting the E-coefficients for the three directions from
    # `NLM_CART` replaces the second dimension with the number of
    # cartesian basis functions of the second orbital (j). Additionally,
    # we are selecting the cartesian directions eliminating the `-3`rd
    # dimension, which ultimately yields the following shape:
    # (ncarti, ncartj, nbatch, ai, aj)
    sx = e0x[:, nlmj[:, 0], ..., 0, :, :]  # type: ignore
    sy = e0y[:, nlmj[:, 1], ..., 1, :, :]  # type: ignore
    sz = e0z[:, nlmj[:, 2], ..., 2, :, :]  # type: ignore

    dx = d0x[:, nlmj[:, 0], ..., 0, :, :]  # type: ignore
    dy = d0y[:, nlmj[:, 1], ..., 1, :, :]  # type: ignore
    dz = d0z[:, nlmj[:, 2], ..., 2, :, :]  # type: ignore

    # First, we multiply sx, sy and sz with sij using the indices a (ai)
    # and b (aj). Then, we sum over the alphas (a and b), reducing the
    # tensor to (ncarti, ncartj, nbatch). Since the batch dimension is
    # conventionally the first dimension, we cycle the indices and obtain
    # the final shape: (nbatch, ncarti, ncartj)
    _einsum = partial(
        einsum,
        "ij...ab,ij...ab,ij...ab,...ab->...ij",
        optimize=[(0, 1), (2, 0), (0, 1)],  # fastest contraction path
    )
    x = _einsum(dx, sy, sz, sij)
    y = _einsum(sx, dy, sz, sij)
    z = _einsum(sx, sy, dz, sij)

    # For the full cartesian gradient, we stack cartesian the contributions in
    # the first dimension, i.e., before the batch: (nbatch, 3, ncarti, ncartj).
    # Although the cartesian dimension always goes last, we apply `torch.tril`
    # or `torch.triu` later, which only works on the leading 2D sub-matrices of
    # higher-dimensional tensors. Hence, we reshape later.
    ds3d = torch.stack([x, y, z], dim=-3)

    # transform to spherical basis functions (itrafo * dS * jtrafo^T)
    grad = einsum("...ij,...xjk,...lk->...xil", itrafo, ds3d, jtrafo)

    # sometimes grad is not contiguous anymore (important for performance?)
    if not grad.is_contiguous():
        return grad.contiguous()
    return grad

    # OLD: This is the loop-based version of the above indexing atrocities.
    #      I left it here, as it may be better to understand...
    #      The loop-free version yielded speed-ups of ~10% for autograd.
    #      Maybe, one could check the efficiency of einsum vs matmul...
    #
    # ncarti = torch.div((li + 1) * (li + 2), 2, rounding_mode="floor")
    # ncartj = torch.div((lj + 1) * (lj + 2), 2, rounding_mode="floor")
    #
    # # cartesian overlap and overlap gradient with possible batch dimension
    # s3d = vec.new_zeros(*[*vec.shape[:-1], ncarti, ncartj])
    # ds3d = vec.new_zeros(*[*vec.shape[:-1], 3, ncarti, ncartj])
    #
    # for mli in range(ncarti):
    #     mi = NLM_CART[li][mli, :]
    #     for mlj in range(ncartj):
    #         mj = NLM_CART[lj][mlj, :]
    #
    #         sx = e0[mi[0]][mj[0]][..., 0, :, :]
    #         sy = e0[mi[1]][mj[1]][..., 1, :, :]
    #         sz = e0[mi[2]][mj[2]][..., 2, :, :]
    #
    #         dx = d0[mi[0]][mj[0]][..., 0, :, :]
    #         dy = d0[mi[1]][mj[1]][..., 1, :, :]
    #         dz = d0[mi[2]][mj[2]][..., 2, :, :]
    #
    #         # calculating overlap for free
    #         s3d[..., mli, mlj] += (sij * sx * sy * sz).sum((-2, -1))
    #
    #         # [bs, 3, nacrti, nacrtj]
    #         ds3d[..., :, mli, mlj] += torch.stack(
    #             [
    #                 (sij * dx * sy * sz).sum((-2, -1)),
    #                 (sij * sx * dy * sz).sum((-2, -1)),
    #                 (sij * sx * sy * dz).sum((-2, -1)),
    #             ],
    #             dim=-1,
    #         )


def ecoeffs_s(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> Tensor:
    """
    Explicitly calculate E-coefficients for s-orbitals with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.

    Returns
    -------
    Tensor
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """

    # ss -> not required since 0
    if lj == 0:
        return (
            torch.zeros_like(rpi, requires_grad=rpi.requires_grad)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    e011 = xij
    e100 = rpi
    e010 = rpj
    e000 = torch.ones_like(e100)

    # sp
    if lj == 1:
        return torch.stack([e000, e010]).unsqueeze(0)

    # sd
    if lj == 2:
        e020 = rpj * e010 + e011

        return torch.stack([e000, e010, e020]).unsqueeze(0)

    # sf
    if lj == 3:
        e020 = rpj * e010 + e011

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021

        return torch.stack([e000, e010, e020, e030]).unsqueeze(0)

    raise CGTOAzimuthalQuantumNumberError(lj)


def de_s(
    lj: Tensor,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    ai: Tensor,
    aj: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Explicitly calculate E-coefficients and their derivatives for s-orbitals
    with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    ai : Tensor
        Gaussian exponents for Gaussian `i`.
    aj : Tensor
        Gaussian exponents for Gaussian `j`.

    Returns
    -------
    tuple[Tensor, Tensor]
        "Matrix" of E-coefficients and their gradients. The shape depends on
        the angular momenta involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    a = 2 * ai
    b = 2 * aj
    e000 = torch.ones_like(rpi)
    e011 = xij  # * e000
    e100 = rpi
    e010 = rpj
    f000 = a * e100

    # ss
    if lj == 0:
        return (
            e000.unsqueeze(0).unsqueeze(0),
            f000.unsqueeze(0).unsqueeze(0),
        )

    # sp
    if lj == 1:
        e020 = rpj * e010 + e011
        f010 = e000 - b * e020

        return (
            torch.stack([e000, e010]).unsqueeze(0),
            torch.stack([f000, f010]).unsqueeze(0),
        )

    # sd
    if lj == 2:
        e020 = rpj * e010 + e011

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e020 = rpj * e010 + e011

        # derivatives
        f010 = e000 - b * e020
        f020 = 2 * e010 - b * e030

        return (
            torch.stack([e000, e010, e020]).unsqueeze(0),
            torch.stack([f000, f010, f020]).unsqueeze(0),
        )

    # sf
    if lj == 3:
        e020 = rpj * e010 + e011

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e020 = rpj * e010 + e011

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e040 = rpj * e030 + e031

        # derivatives
        f010 = e000 - b * e020
        f020 = 2 * e010 - b * e030
        f030 = 3 * e020 - b * e040

        return (
            torch.stack([e000, e010, e020, e030]).unsqueeze(0),
            torch.stack([f000, f010, f020, f030]).unsqueeze(0),
        )

    raise CGTOAzimuthalQuantumNumberError(lj)


def ecoeffs_p(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> Tensor:
    """
    Explicitly calculate E-coefficients for p-orbitals with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.

    Returns
    -------
    Tensor
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    e000 = torch.ones_like(rpi)
    e101 = xij
    e011 = xij
    e100 = rpi
    e010 = rpj

    # ps
    if lj == 0:
        # return [[e000], [e100]]
        return torch.stack(
            [
                e000.unsqueeze(0),
                e100.unsqueeze(0),
            ]
        )
    # pp
    if lj == 1:
        e110 = rpj * e100 + e101

        return torch.stack(
            [
                torch.stack([e000, e010]),
                torch.stack([e100, e110]),
            ]
        )
    # pd
    if lj == 2:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e120 = rpj * e110 + e111

        return torch.stack(
            [
                torch.stack([e000, e010, e020]),
                torch.stack([e100, e110, e120]),
            ]
        )
    # pf
    if lj == 3:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e120 = rpj * e110 + e111

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e130 = rpi * e030 + e031

        return torch.stack(
            [
                torch.stack([e000, e010, e020, e030]),
                torch.stack([e100, e110, e120, e130]),
            ]
        )

    raise CGTOAzimuthalQuantumNumberError(lj)


def de_p(
    lj: Tensor,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    ai: Tensor,
    aj: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Explicitly calculate E-coefficients and their derivatives for p-orbitals
    with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    ai : Tensor
        Gaussian exponents for Gaussian `i`.
    aj : Tensor
        Gaussian exponents for Gaussian `j`.

    Returns
    -------
    tuple[Tensor, Tensor]
        "Matrix" of E-coefficients and their gradients. The shape depends on
        the angular momenta involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    a = 2 * ai
    b = 2 * aj
    e000 = torch.ones_like(rpi)
    e101 = xij  # * e000
    e011 = xij  # * e000
    e100 = rpi
    e010 = rpj
    f000 = a * e100

    # ps
    if lj == 0:
        e200 = rpi * e100 + e101
        f100 = a * e200 - e000

        return (
            torch.stack(
                [
                    e000.unsqueeze(0),
                    e100.unsqueeze(0),
                ]
            ),
            torch.stack(
                [
                    f000.unsqueeze(0),
                    f100.unsqueeze(0),
                ]
            ),
        )

    # pp
    if lj == 1:
        e110 = rpj * e100 + e101

        e200 = rpi * e100 + e101
        f100 = a * e200 - e000

        # derivatives
        e020 = rpj * e010 + e011
        f010 = e000 - b * e020

        e111 = xij * e100 + rpj * e101
        e210 = rpi * e110 + e111
        f110 = a * e210 - e010

        return (
            torch.stack(
                [
                    torch.stack([e000, e010]),
                    torch.stack([e100, e110]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010]),
                    torch.stack([f100, f110]),
                ]
            ),
        )

    # pd
    if lj == 2:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e120 = rpj * e110 + e111

        # derivatives
        e200 = rpi * e100 + e101
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        e210 = rpi * e110 + e111
        f110 = a * e210 - e010

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        f020 = 2 * e010 - b * e030

        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112
        e220 = rpj * e210 + e211
        f120 = a * e220 - e020

        return (
            torch.stack(
                [
                    torch.stack([e000, e010, e020]),
                    torch.stack([e100, e110, e120]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010, f020]),
                    torch.stack([f100, f110, f120]),
                ]
            ),
        )

    # pf
    if lj == 3:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e120 = rpj * e110 + e111

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e130 = rpi * e030 + e031

        # derivatives
        e200 = rpi * e100 + e101
        f100 = a * e200 - e000

        e020 = rpj * e010 + e011
        f010 = e000 - b * e020

        e111 = xij * e100 + rpj * e101
        e210 = rpi * e110 + e111
        f110 = a * e210 - e010

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        f020 = 2 * e010 - b * e030

        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211
        f120 = a * e220 - e020

        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e040 = rpj * e030 + e031
        f030 = 3 * e020 - b * e040

        e130 = rpi * e030 + e031
        e032 = xij * e021 + rpj * e022
        e131 = xij * e030 + rpi * e031 + 2 * e032
        e230 = rpi * e130 + e131
        f130 = a * e230 - e030

        return (
            torch.stack(
                [
                    torch.stack([e000, e010, e020, e030]),
                    torch.stack([e100, e110, e120, e130]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010, f020, f030]),
                    torch.stack([f100, f110, f120, f130]),
                ]
            ),
        )

    raise CGTOAzimuthalQuantumNumberError(lj)


def ecoeffs_d(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> Tensor:
    """
    Explicitly calculate E-coefficients for d-orbitals with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.

    Returns
    -------
    Tensor
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    e101 = xij
    e011 = xij
    e100 = rpi
    e010 = rpj
    e110 = rpj * e100 + e101
    e000 = torch.ones_like(e100)

    e200 = rpi * e100 + e101

    # ds
    if lj == 0:
        return torch.stack(
            [
                e000.unsqueeze(0),
                e100.unsqueeze(0),
                e200.unsqueeze(0),
            ]
        )
    # dp
    if lj == 1:
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111

        return torch.stack(
            [
                torch.stack([e000, e010]),
                torch.stack([e100, e110]),
                torch.stack([e200, e210]),
            ]
        )
    # dd
    if lj == 2:
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        return torch.stack(
            [
                torch.stack([e000, e010, e020]),
                torch.stack([e100, e110, e120]),
                torch.stack([e200, e210, e220]),
            ]
        )
    # df
    if lj == 3:
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e130 = rpi * e030 + e031
        e032 = xij * e021 + rpj * e022
        e131 = xij * e030 + rpi * e031 + 2 * e032
        e230 = rpi * e130 + e131

        return torch.stack(
            [
                torch.stack([e000, e010, e020, e030]),
                torch.stack([e100, e110, e120, e130]),
                torch.stack([e200, e210, e220, e230]),
            ]
        )

    raise CGTOAzimuthalQuantumNumberError(lj)


def de_d(
    lj: Tensor,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    ai: Tensor,
    aj: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Explicitly calculate E-coefficients and their derivatives for d-orbitals
    with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    ai : Tensor
        Gaussian exponents for Gaussian `i`.
    aj : Tensor
        Gaussian exponents for Gaussian `j`.

    Returns
    -------
    tuple[Tensor, Tensor]
        "Matrix" of E-coefficients and their gradients. The shape depends on
        the angular momenta involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    a = 2 * ai
    b = 2 * aj
    e000 = torch.ones_like(rpi)
    e101 = xij  # * e000
    e011 = xij  # * e000
    e100 = rpi
    e010 = rpj
    e200 = rpi * e100 + e101
    f000 = a * e100

    # ds
    if lj == 0:
        f100 = a * e200 - e000

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201
        f200 = a * e300 - 2 * e100

        return (
            torch.stack(
                [
                    e000.unsqueeze(0),
                    e100.unsqueeze(0),
                    e200.unsqueeze(0),
                ]
            ),
            torch.stack(
                [
                    f000.unsqueeze(0),
                    f100.unsqueeze(0),
                    f200.unsqueeze(0),
                ]
            ),
        )

    # dp
    if lj == 1:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111

        # derivatives
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201
        f200 = a * e300 - 2 * e100

        f110 = a * e210 - e010

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301
        f210 = a * e310 - 2 * e110

        return (
            torch.stack(
                [
                    torch.stack([e000, e010]),
                    torch.stack([e100, e110]),
                    torch.stack([e200, e210]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010]),
                    torch.stack([f100, f110]),
                    torch.stack([f200, f210]),
                ]
            ),
        )

    # dd
    if lj == 2:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        # derivatives
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201
        f200 = a * e300 - 2 * e100

        f110 = a * e210 - e010

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301
        f210 = a * e310 - 2 * e110

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        f020 = 2 * e010 - b * e030

        f120 = a * e220 - e020

        e302 = xij * e201 + rpi * e202
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e320 = rpj * e310 + e311

        f220 = a * e320 - 2 * e120

        return (
            torch.stack(
                [
                    torch.stack([e000, e010, e020]),
                    torch.stack([e100, e110, e120]),
                    torch.stack([e200, e210, e220]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010, f020]),
                    torch.stack([f100, f110, f120]),
                    torch.stack([f200, f210, f220]),
                ]
            ),
        )

    # df
    if lj == 3:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e130 = rpi * e030 + e031
        e032 = xij * e021 + rpj * e022
        e131 = xij * e030 + rpi * e031 + 2 * e032
        e230 = rpi * e130 + e131

        # derivatives
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201
        f200 = a * e300 - 2 * e100

        f110 = a * e210 - e010

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301
        f210 = a * e310 - 2 * e110

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        f020 = 2 * e010 - b * e030

        f120 = a * e220 - e020

        e302 = xij * e201 + rpi * e202
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e320 = rpj * e310 + e311
        f220 = a * e320 - 2 * e120

        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e040 = rpj * e030 + e031
        f030 = 3 * e020 - b * e040

        f130 = a * e230 - e030

        e033 = xij * e022
        e132 = xij * e031 + rpi * e032 + 3 * e033
        e231 = xij * e130 + rpi * e131 + 2 * e132
        e330 = rpi * e230 + e231
        f230 = a * e330 - 2 * e130

        return (
            torch.stack(
                [
                    torch.stack([e000, e010, e020, e030]),
                    torch.stack([e100, e110, e120, e130]),
                    torch.stack([e200, e210, e220, e230]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010, f020, f030]),
                    torch.stack([f100, f110, f120, f130]),
                    torch.stack([f200, f210, f220, f230]),
                ]
            ),
        )

    raise CGTOAzimuthalQuantumNumberError(lj)


def ecoeffs_f(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> Tensor:
    """
    Explicitly calculate E-coefficients for f-orbitals with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.

    Returns
    -------
    Tensor
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    e101 = xij
    e011 = xij
    e100 = rpi
    e010 = rpj
    e000 = torch.ones_like(e100)

    e200 = rpi * e100 + e101

    e201 = xij * e100 + rpi * e101
    e300 = rpi * e200 + e201

    # fs
    if lj == 0:
        return torch.stack(
            [
                e000.unsqueeze(0),
                e100.unsqueeze(0),
                e200.unsqueeze(0),
                e300.unsqueeze(0),
            ]
        )
    # fp
    if lj == 1:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        return torch.stack(
            [
                torch.stack([e000, e010]),
                torch.stack([e100, e110]),
                torch.stack([e200, e210]),
                torch.stack([e300, e310]),
            ]
        )
    # fd
    if lj == 2:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        e302 = xij * e201 + rpi * e202
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e320 = rpj * e310 + e311

        return torch.stack(
            [
                torch.stack([e000, e010, e020]),
                torch.stack([e100, e110, e120]),
                torch.stack([e200, e210, e220]),
                torch.stack([e300, e310, e320]),
            ]
        )
    # ff
    if lj == 3:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e202 = xij * e101
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301
        e130 = rpi * e030 + e031

        e302 = xij * e201 + rpi * e202
        e032 = xij * e021 + rpj * e022
        e131 = xij * e030 + rpi * e031 + 2 * e032
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e320 = rpj * e310 + e311
        e230 = rpi * e130 + e131

        e033 = xij * e022
        e132 = xij * e031 + rpi * e032 + 3 * e033
        e231 = xij * e130 + rpi * e131 + 2 * e132
        e330 = rpi * e230 + e231

        return torch.stack(
            [
                torch.stack([e000, e010, e020, e030]),
                torch.stack([e100, e110, e120, e130]),
                torch.stack([e200, e210, e220, e230]),
                torch.stack([e300, e310, e320, e330]),
            ]
        )

    raise CGTOAzimuthalQuantumNumberError(lj)


def de_f(
    lj: Tensor,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    ai: Tensor,
    aj: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Explicitly calculate E-coefficients and their derivatives for f-orbitals
    with s/p/d/f-orbitals.

    Parameters
    ----------
    lj : Tensor
        Angular momentum
    xij : Tensor
        Prefactor. (`1/2p` with `p` from Gaussian product theorem)
    rpi : Tensor
        Distance between aufpunkt of Gaussian `i` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    rpj : Tensor
        Distance between aufpunkt of Gaussian `j` and new Gaussian `p`.
        The new Gaussian stems from the product of the two Gaussians.
    ai : Tensor
        Gaussian exponents for Gaussian `i`.
    aj : Tensor
        Gaussian exponents for Gaussian `j`.

    Returns
    -------
    tuple[Tensor, Tensor]
        "Matrix" of E-coefficients and their gradients. The shape depends on
        the angular momenta involved.

    Raises
    ------
    CGTOAzimuthalQuantumNumberError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    a = 2 * ai
    b = 2 * aj
    e000 = torch.ones_like(rpi)
    e101 = xij  # * e000
    e011 = xij  # * e000
    e100 = rpi
    e010 = rpj
    e200 = rpi * e100 + e101
    f000 = a * e100

    # fs
    if lj == 0:
        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e400 = rpi * e300 + e301

        # derivatives
        f100 = a * e200 - e000
        f200 = a * e300 - 2 * e100
        f300 = a * e400 - 3 * e200

        return (
            torch.stack(
                [
                    e000.unsqueeze(0),
                    e100.unsqueeze(0),
                    e200.unsqueeze(0),
                    e300.unsqueeze(0),
                ]
            ),
            torch.stack(
                [
                    f000.unsqueeze(0),
                    f100.unsqueeze(0),
                    f200.unsqueeze(0),
                    f300.unsqueeze(0),
                ]
            ),
        )

    # fp
    if lj == 1:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        # derivatives
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        f200 = a * e300 - 2 * e100

        e400 = rpi * e300 + e301
        f300 = a * e400 - 3 * e200

        f110 = a * e210 - e010
        f210 = a * e310 - 2 * e110

        e302 = xij * e201 + rpi * e202
        e021 = xij * e010 + rpj * e011
        e022 = xij * e011
        e032 = xij * e021 + rpj * e022
        e311 = xij * e300 + rpj * e301 + 2 * e302

        e410 = rpi * e310 + e311
        f310 = a * e410 - 3 * e210

        return (
            torch.stack(
                [
                    torch.stack([e000, e010]),
                    torch.stack([e100, e110]),
                    torch.stack([e200, e210]),
                    torch.stack([e300, e310]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010]),
                    torch.stack([f100, f110]),
                    torch.stack([f200, f210]),
                    torch.stack([f300, f310]),
                ]
            ),
        )

    # fd
    if lj == 2:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111

        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112
        e220 = rpj * e210 + e211

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        e302 = xij * e201 + rpi * e202
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e320 = rpj * e310 + e311

        # derivatives
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        f200 = a * e300 - 2 * e100

        e400 = rpi * e300 + e301
        f300 = a * e400 - 3 * e200

        f110 = a * e210 - e010
        f210 = a * e310 - 2 * e110

        e302 = xij * e201 + rpi * e202
        e021 = xij * e010 + rpj * e011
        e022 = xij * e011
        e032 = xij * e021 + rpj * e022
        e311 = xij * e300 + rpj * e301 + 2 * e302

        e410 = rpi * e310 + e311
        f310 = a * e410 - 3 * e210

        e030 = rpj * e020 + e021
        f020 = 2 * e010 - b * e030

        f120 = a * e220 - e020

        e320 = rpj * e310 + e311
        f220 = a * e320 - 2 * e120

        e303 = xij * e202
        e312 = xij * e301 + rpj * e302 + 3 * e303
        e321 = xij * e310 + rpj * e311 + 2 * e312
        e420 = rpi * e320 + e321
        f320 = a * e420 - 3 * e220

        return (
            torch.stack(
                [
                    torch.stack([e000, e010, e020]),
                    torch.stack([e100, e110, e120]),
                    torch.stack([e200, e210, e220]),
                    torch.stack([e300, e310, e320]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010, f020]),
                    torch.stack([f100, f110, f120]),
                    torch.stack([f200, f210, f220]),
                    torch.stack([f300, f310, f320]),
                ]
            ),
        )

    # ff
    if lj == 3:
        e110 = rpj * e100 + e101
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111

        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112
        e220 = rpj * e210 + e211

        e201 = xij * e100 + rpi * e101
        e300 = rpi * e200 + e201

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e202 = xij * e101
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301
        e130 = rpi * e030 + e031

        e032 = xij * e021 + rpj * e022
        e302 = xij * e201 + rpi * e202
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e131 = xij * e030 + rpi * e031 + 2 * e032
        e320 = rpj * e310 + e311
        e230 = rpi * e130 + e131

        e033 = xij * e022
        e132 = xij * e031 + rpi * e032 + 3 * e033
        e231 = xij * e130 + rpi * e131 + 2 * e132
        e330 = rpi * e230 + e231

        # derivatives
        f100 = a * e200 - e000
        f010 = e000 - b * e020

        f200 = a * e300 - 2 * e100

        e400 = rpi * e300 + e301
        f300 = a * e400 - 3 * e200

        f110 = a * e210 - e010
        f210 = a * e310 - 2 * e110

        e302 = xij * e201 + rpi * e202
        e021 = xij * e010 + rpj * e011
        e022 = xij * e011
        e032 = xij * e021 + rpj * e022
        e311 = xij * e300 + rpj * e301 + 2 * e302

        e410 = rpi * e310 + e311
        f310 = a * e410 - 3 * e210

        e030 = rpj * e020 + e021
        f020 = 2 * e010 - b * e030

        e040 = rpj * e030 + e031
        f030 = 3 * e020 - b * e040

        f120 = a * e220 - e020
        f130 = a * e230 - e030

        e320 = rpj * e310 + e311
        f220 = a * e320 - 2 * e120
        f230 = a * e330 - 2 * e130

        e303 = xij * e202
        e312 = xij * e301 + rpj * e302 + 3 * e303
        e321 = xij * e310 + rpj * e311 + 2 * e312
        e420 = rpi * e320 + e321
        f320 = a * e420 - 3 * e220

        e313 = xij * e302 + rpj * e303
        e322 = xij * e311 + rpj * e312 + 3 * e313
        e331 = xij * e320 + rpj * e321 + 2 * e322
        e430 = rpi * e330 + e331
        f330 = a * e430 - 3 * e230

        return (
            torch.stack(
                [
                    torch.stack([e000, e010, e020, e030]),
                    torch.stack([e100, e110, e120, e130]),
                    torch.stack([e200, e210, e220, e230]),
                    torch.stack([e300, e310, e320, e330]),
                ]
            ),
            torch.stack(
                [
                    torch.stack([f000, f010, f020, f030]),
                    torch.stack([f100, f110, f120, f130]),
                    torch.stack([f200, f210, f220, f230]),
                    torch.stack([f300, f310, f320, f330]),
                ]
            ),
        )

    raise CGTOAzimuthalQuantumNumberError(lj)

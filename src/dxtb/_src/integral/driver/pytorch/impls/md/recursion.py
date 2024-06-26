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
Calculation of overlap integrals using the McMurchie-Davidson algorithm.

- L. E. McMurchie, E. R. Davidson, One- and two-electron integrals over
  cartesian gaussian functions, *J. Comput. Phys.*, **1978**, *26*, 218-231.
  (`DOI <https://doi.org/10.1016/0021-9991(78)90092-X>`__)
"""

from __future__ import annotations

from math import pi, sqrt

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Any, Callable, Tensor
from dxtb._src.typing.exceptions import IntegralTransformError

from ......utils import t2int
from .trafo import NLM_CART, TRAFO

__all__ = ["md_recursion", "md_recursion_gradient"]


sqrtpi3 = sqrt(pi) ** 3


@torch.jit.script
def _e_function(E: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> Tensor:
    """
    Calculate E-coefficients for McMurchie-Davidson algorithm. Rather than computing
    them recursively, they are computed iteratively in this implementation.
    The number of coefficients is determined by the requested shape.

    Parameters
    ----------
    E : Tensor
        E-coefficients for the given shell pair, will be modified in-place.
    xij : Tensor
        One over two times the product Gaussian exponent of the two shells.
    rpi : Tensor
        Distance between the center of the first shell and the product Gaussian center.
    rpj : Tensor
        Distance between the center of the second shell and the product Gaussian center.

    Returns
    -------
    Tensor
        E-coefficients for the given shell pair.
    """

    E[:] = 0.0

    # do j = 0 and i = 0 (starting coefficient of recursion, zeroth order HP)
    E[..., 0, 0, 0] = 1.0

    # do j = 0 for all i > 0 (all orders of HP)
    for i in range(1, E.shape[-3]):
        # t = 0 (excludes t - 1 term)
        E[..., i, 0, 0] = rpi * E[..., i - 1, 0, 0] + E[..., i - 1, 0, 1]

        # t > 0
        for n in range(1, i):
            E[..., i, 0, n] = (
                xij * E[..., i - 1, 0, n - 1]
                + rpi * E[..., i - 1, 0, n]
                + (1 + n) * E[..., i - 1, 0, n + 1]
            )

        # t = tmax (excludes t + 1 term)
        E[..., i, 0, i] = xij * E[..., i - 1, 0, i - 1] + rpi * E[..., i - 1, 0, i]

    # do all j > 0 for all i's (all orders of HP)
    for j in range(1, E.shape[-2]):
        for i in range(0, E.shape[-3]):
            # t = 0 (excludes t - 1 term)
            E[..., i, j, 0] = rpj * E[..., i, j - 1, 0] + E[..., i, j - 1, 1]

            # t > 0
            for n in range(1, i + j):
                E[..., i, j, n] = (
                    xij * E[..., i, j - 1, n - 1]
                    + rpj * E[..., i, j - 1, n]
                    + (1 + n) * E[..., i, j - 1, n + 1]
                )

            # t = tmax (excludes t + 1 term)
            E[..., i, j, i + j] = (
                xij * E[..., i, j - 1, i + j - 1] + rpj * E[..., i, j - 1, i + j]
            )

    return E


_e_function_jit: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = torch.jit.trace(
    _e_function,
    (
        torch.rand((3, 5, 5, 11)),
        torch.tensor(1.0),
        torch.rand((3,)),
        torch.rand((3,)),
    ),
)  # type: ignore


@torch.jit.script
def _e_function_grad(
    E: Tensor,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    dxij: Tensor,
    drpi: Tensor,
    drpj: Tensor,
) -> Tensor:
    """
    Calculate derivative of E-coefficients for McMurchie-Davidson algorithm.

    Parameters
    ----------
    E: Tensor
        E-coefficients for the given shell pair.
    xij : Tensor
        One over two times the product Gaussian exponent of the two shells.
    rpi : Tensor
        Distance between the center of the first shell and the product Gaussian center.
    rpj : Tensor
        Distance between the center of the second shell and the product Gaussian center.
    dxij : Tensor
        Derivative of the product Gaussian exponent of the two shells.
    drpi : Tensor
        Derivative of the distance between the center of the first shell
        and the product Gaussian center.
    drpj : Tensor
        Derivative of the distance between the center of the second shell
        and the product Gaussian center.

    Returns
    -------
    Tensor
        Derivative of E-coefficients for the given shell pair.
    """

    dE = torch.zeros_like(E)
    for i in range(1, E.shape[-3]):
        dE[..., i, 0, 0] = (
            drpi * E[..., i - 1, 0, 0]
            + rpi * dE[..., i - 1, 0, 0]
            + dE[..., i - 1, 0, 1]
        )
        for n in range(1, i):
            dE[..., i, 0, n] = (
                dxij * E[..., i - 1, 0, n - 1]
                + xij * dE[..., i - 1, 0, n - 1]
                + drpi * E[..., i - 1, 0, n]
                + rpi * dE[..., i - 1, 0, n]
                + (1 + n) * dE[..., i - 1, 0, n + 1]
            )
        dE[..., i, 0, i] = (
            dxij * E[..., i - 1, 0, i - 1]
            + xij * dE[..., i - 1, 0, i - 1]
            + drpi * E[..., i - 1, 0, i]
            + rpi * dE[..., i - 1, 0, i]
        )
    for j in range(1, dE.shape[-2]):
        for i in range(0, dE.shape[-3]):
            dE[..., i, j, 0] = (
                drpj * E[..., i, j - 1, 0]
                + rpj * dE[..., i, j - 1, 0]
                + dE[..., i, j - 1, 1]
            )
            for n in range(1, i + j):
                dE[..., i, j, n] = (
                    dxij * E[..., i, j - 1, n - 1]
                    + xij * dE[..., i, j - 1, n - 1]
                    + drpj * E[..., i, j - 1, n]
                    + rpj * dE[..., i, j - 1, n]
                    + (1 + n) * dE[..., i, j - 1, n + 1]
                )
            dE[..., i, j, i + j] = (
                dxij * E[..., i, j - 1, i + j - 1]
                + xij * dE[..., i, j - 1, i + j - 1]
                + drpj * E[..., i, j - 1, i + j]
                + rpj * dE[..., i, j - 1, i + j]
            )

    return dE


_e_function_grad_jit: Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor
] = torch.jit.trace(
    _e_function_grad,
    (
        torch.rand((3, 5, 5, 11)),
        torch.tensor(1.0),
        torch.rand((3,)),
        torch.rand((3,)),
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
    ),
)  # type: ignore


class EFunction(torch.autograd.Function):
    """
    Autograd function for E-coefficients for McMurchie-Davidson algorithm.
    """

    @staticmethod
    def forward(
        ctx: Any,
        xij: Tensor,
        rpi: Tensor,
        rpj: Tensor,
        shape: tuple[int, ...],
    ) -> Tensor:
        """
        Calculate E-coefficients for McMurchie-Davidson algorithm.
        The number of coefficients is determined by the requested shape.

        Parameters
        ----------
        xij : Tensor
            One over two times the product Gaussian exponent of the two shells.
        rpi : Tensor
            Distance between the center of the first shell and the product
            Gaussian center.
        rpj : Tensor
            Distance between the center of the second shell and the product
            Gaussian center.
        shape : Tuple[..., int, int]
            Shape of the E-coefficients to calculate, the first dimensions
            identify the batch, the last two identify the angular momentum of
            the first and second shell.

        Returns
        -------
        Tensor
            E-coefficients for the given shell pair.
        """

        ctx.shape = (*shape, shape[-1] + shape[-2])
        E = _e_function_jit(xij.new_zeros(*ctx.shape), xij, rpi, rpj)

        ctx.save_for_backward(E, xij, rpi, rpj)
        return E

    @staticmethod
    def backward(
        ctx: Any,
        E_bar: Tensor,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
        """
        Calculate derivative of E-coefficients for McMurchie-Davidson algorithm.

        Parameters
        ----------
        E_bar: Tensor
            Derivative with respect to the E-coefficients for the given shell pair.

        Returns
        -------
        xij_bar : Tensor
            Derivative with respect to the product Gaussian exponent of the two shells.
        rpi_bar : Tensor
            Derivative with respect to the distance between the center of the first shell
            and the product Gaussian center.
        rpj_bar : Tensor
            Derivative with respect to the distance between the center of the second shell
            and the product Gaussian center.
        """

        E, xij, rpi, rpj = ctx.saved_tensors
        xij_grad, rpi_grad, rpj_grad, _ = ctx.needs_input_grad
        xij_bar = rpi_bar = rpj_bar = None

        # We want torched zeros and ones...
        _1 = E.new_ones(())
        _0 = E.new_zeros(())

        if xij_grad:
            xij_bar = torch.sum(
                E_bar * _e_function_grad_jit(E, xij, rpi, rpj, _1, _0, _0),
                (-3, -2, -1),
            )
        if rpi_grad:
            rpi_bar = torch.sum(
                E_bar * _e_function_grad_jit(E, xij, rpi, rpj, _0, _1, _0),
                (-3, -2, -1),
            )
        if rpj_grad:
            rpj_bar = torch.sum(
                E_bar * _e_function_grad_jit(E, xij, rpi, rpj, _0, _0, _1),
                (-3, -2, -1),
            )

        return xij_bar, rpi_bar, rpj_bar, None


# TODO-3.11: Fix typing with unpacking
e_function: Callable[
    [Tensor, Tensor, Tensor, tuple[tuple[int, ...], int, int, Tensor, Tensor]], Tensor
] = EFunction.apply  # type: ignore


def md_recursion(
    angular: tuple[Tensor, Tensor],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Calculate overlap integrals using (recursive) McMurchie-Davidson algorithm.

    Parameters
    ----------
    angular : (Tensor, Tensor)
        Angular momentum of the shell pair(s).
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents of the shell pair(s).
    coeff : (Tensor, Tensor)
        Contraction coefficients of the shell pair(s).
    vec : Tensor
        Displacement vector between shell pair(s).

    Returns
    -------
    Tensor
        Overlap integrals for shell pair(s).
    """

    # angular momenta and number of cartesian gaussian basis functions
    li, lj = angular
    ncarti = torch.div((li + 1) * (li + 2), 2, rounding_mode="floor")
    ncartj = torch.div((lj + 1) * (lj + 2), 2, rounding_mode="floor")
    r2 = torch.sum(vec.pow(2), -1)

    # transform from cartesian to spherical gaussians
    try:
        itrafo = TRAFO[li].type(vec.dtype).to(vec.device)
        jtrafo = TRAFO[lj].type(vec.dtype).to(vec.device)
    except IndexError as e:
        raise IntegralTransformError() from e

    # cartesian overlap (`vec.shape[:-1]` accounts for possible batch dimension)
    shape = [*vec.shape[:-1], ncarti, ncartj]
    s3d = vec.new_zeros(*shape)

    ai, aj = alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-2)
    ci, cj = coeff[0].unsqueeze(-1), coeff[1].unsqueeze(-2)
    eij = ai + aj
    oij = 1.0 / eij
    xij = 0.5 * oij

    # p * (R_A - R_B)² with p = a*b/(a+b)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-2)

    # K_AB * Gaussian integral (√(pi/(a+b))) in 3D * c_A * c_B
    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
    rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

    E = e_function(
        xij,
        rpi,
        rpj,
        (*vec.shape, ai.shape[-2], aj.shape[-1], li + 1, lj + 1),  # type: ignore
    )

    for mli in range(s3d.shape[-2]):
        mi = NLM_CART[li][mli, :]
        for mlj in range(s3d.shape[-1]):
            mj = NLM_CART[lj][mlj, :]
            s3d[..., mli, mlj] += (
                sij
                * E[..., 0, :, :, mi[0], mj[0], 0]
                * E[..., 1, :, :, mi[1], mj[1], 0]
                * E[..., 2, :, :, mi[2], mj[2], 0]
            ).sum((-2, -1))

    # transform to cartesian basis functions (itrafo * S * jtrafo^T)
    o = einsum("...ij,...jk,...lk->...il", itrafo, s3d, jtrafo)

    return o


def e_function_derivative(e: Tensor, ai: Tensor, li: Tensor, lj: Tensor) -> Tensor:
    """Calculate derivative of E coefficients."""
    _li, _lj = t2int(li), t2int(lj)

    de = torch.zeros((e.shape[0], 3, e.shape[-5], e.shape[-4], _li + 1, _lj + 1, 1))
    for k in range(e.shape[-4]):  # aj
        for i in range(li + 1):  # li
            for j in range(lj + 1):  # lj
                de[..., k, i, j, 0] = 2 * ai * e[..., k, i + 1, j, 0]
                if i > 0:
                    de[..., k, i, j, 0] -= i * e[..., k, i - 1, j, 0]
    return de


def md_recursion_gradient(
    angular: tuple[Tensor, Tensor],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Calculate overlap gradients using (recursive) McMurchie-Davidson algorithm.

    Parameters
    ----------
    angular : (Tensor, Tensor)
        Angular momentum of the shell pair(s).
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents of the shell pair(s).
    coeff : (Tensor, Tensor)
        Contraction coefficients of the shell pair(s).
    vec : Tensor
        Displacement vector between shell pair(s).

    Returns
    -------
    Tensor
        Overlap gradient for shell pair(s).
    """

    # angular momenta and number of cartesian gaussian basis functions
    li, lj = angular
    ncarti = torch.div((li + 1) * (li + 2), 2, rounding_mode="floor")
    ncartj = torch.div((lj + 1) * (lj + 2), 2, rounding_mode="floor")
    r2 = torch.sum(vec.pow(2), -1)

    # init transformation from cartesian to spherical gaussians
    try:
        itrafo = TRAFO[li].type(vec.dtype).to(vec.device)
        jtrafo = TRAFO[lj].type(vec.dtype).to(vec.device)
    except IndexError as e:
        raise IntegralTransformError() from e

    # cartesian overlap and overlap gradient
    s3d = vec.new_zeros(*[*vec.shape[:-1], ncarti, ncartj])
    ds3d = vec.new_zeros(*[*vec.shape[:-1], 3, ncarti, ncartj])

    ai, aj = alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-2)
    ci, cj = coeff[0].unsqueeze(-1), coeff[1].unsqueeze(-2)
    oij = 1.0 / (ai + aj)

    # p * (R_A - R_B)² with p = a*b/(a+b)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-2)

    # K_AB * Gaussian integral (√(pi/(a+b))) in 3D * c_A * c_B
    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    # NOTE: watch out for correct +/- vec (definition + argument)
    rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
    rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

    # for single gaussians (e.g. in tests)
    if len(vec.shape) == 1:
        vec = torch.unsqueeze(vec, 0)
        s3d = torch.unsqueeze(s3d, 0)
        ds3d = torch.unsqueeze(ds3d, 0)

    # calc E function for all (ai, aj)-combis for all vecs in batch
    E = e_function(
        0.5 * oij,
        rpi,
        rpj,
        (*vec.shape, ai.shape[-2], aj.shape[-1], li + 2, lj + 2),  # type: ignore
    )
    dE = e_function_derivative(E, alpha[0], li, lj)

    for mli in range(ncarti):
        mi = NLM_CART[li][mli, :]  # [3]
        for mlj in range(ncartj):
            mj = NLM_CART[lj][mlj, :]  # [3]

            e0 = E[..., 0, :, :, mi[0], mj[0], 0]
            e1 = E[..., 1, :, :, mi[1], mj[1], 0]
            e2 = E[..., 2, :, :, mi[2], mj[2], 0]

            d0 = dE[..., 0, :, :, mi[0], mj[0], 0]
            d1 = dE[..., 1, :, :, mi[1], mj[1], 0]
            d2 = dE[..., 2, :, :, mi[2], mj[2], 0]

            ds3d[..., :, mli, mlj] += torch.stack(
                [
                    (sij * d0 * e1 * e2).sum((-2, -1)),
                    (sij * e0 * d1 * e2).sum((-2, -1)),
                    (sij * e0 * e1 * d2).sum((-2, -1)),
                ],
                dim=-1,
            )  # [bs, 3]

    # transform to spherical basis functions (itrafo * S * jtrafo^T)
    return einsum("...ij,...xjk,...lk->...xil", itrafo, ds3d, jtrafo)

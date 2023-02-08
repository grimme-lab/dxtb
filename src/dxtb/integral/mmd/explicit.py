"""
Calculation of overlap integrals using the McMurchie-Davidson algorithm.

- L. E. McMurchie, E. R. Davidson, One- and two-electron integrals over
  cartesian gaussian functions, *J. Comput. Phys.*, **1978**, *26*, 218-231.
  (`DOI <https://doi.org/10.1016/0021-9991(78)90092-X>`__)
"""
from __future__ import annotations

from math import sqrt, pi

import torch

from ..._types import Tensor
from ...utils import IntegralTransformError
from ..constants import TRAFO, NLM_CART

sqrtpi3 = sqrt(pi) ** 3


def mmd_explicit(
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
        Displacement vector between shell pair(s).

    Returns
    -------
    Tensor
        Overlap integrals for shell pair(s).
    """

    li, lj = angular
    ncarti = torch.div((li + 1) * (li + 2), 2, rounding_mode="floor")
    ncartj = torch.div((lj + 1) * (lj + 2), 2, rounding_mode="floor")
    r2 = torch.sum(vec.pow(2), -1)

    shape = [*vec.shape[:-1], ncarti, ncartj]
    s3d = vec.new_zeros(*shape)

    try:
        itrafo = TRAFO[li].type(s3d.dtype).to(s3d.device)
        jtrafo = TRAFO[lj].type(s3d.dtype).to(s3d.device)
    except IndexError as e:
        raise IntegralTransformError() from e

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

    # ss does not require E-coefficients (e000 = 1)
    if li == 0 and lj == 0:
        s3d = sij.sum((-2, -1), keepdim=True)
    else:
        e101 = xij
        e011 = xij
        e100 = rpi
        e010 = rpj
        e110 = rpj * rpi + e101
        e000 = torch.ones_like(e100)

        # sp
        if li == 0 and lj == 1:
            e0 = [
                [e000, e010],
            ]
        # ps
        elif li == 1 and lj == 0:
            e0 = [
                [e000],
                [e100],
            ]
        # pp
        elif li == 1 and lj == 1:
            e0 = [
                [e000, e010],
                [e100, e110],
            ]
        # sd
        elif li == 0 and lj == 2:
            e020 = rpj * e010 + e011

            e0 = [
                [e000, e010, e020],
            ]
        # ds
        elif li == 2 and lj == 0:
            e200 = rpi * e100 + e101

            e0 = [
                [e000],
                [e100],
                [e200],
            ]
        # pd
        elif li == 1 and lj == 2:
            e111 = xij * e100 + rpj * e101

            e200 = rpi * e100 + e101
            e020 = rpj * e010 + e011
            e120 = rpj * e110 + e111

            e0 = [
                [e000, e010, e020],
                [e100, e110, e120],
            ]
        # dp
        elif li == 2 and lj == 1:
            e111 = xij * e100 + rpj * e101

            e200 = rpi * e100 + e101
            e020 = rpj * e010 + e011
            e210 = rpi * e110 + e111

            e0 = [
                [e000, e010],
                [e100, e110],
                [e200, e210],
            ]
        # dd
        elif li == 2 and lj == 2:
            e111 = xij * e100 + rpj * e101
            e112 = xij * e011
            e211 = xij * e110 + rpi * e111 + 2 * e112

            e200 = rpi * e100 + e101
            e020 = rpj * e010 + e011
            e210 = rpi * e110 + e111
            e120 = rpj * e110 + e111
            e220 = rpj * e210 + e211

            e0 = [
                [e000, e010, e020],
                [e100, e110, e120],
                [e200, e210, e220],
            ]
        else:
            raise RuntimeError(f"Unsupported angular momentum {li} {lj}.")

        for mli in range(ncarti):
            mi = NLM_CART[li][mli, :]
            for mlj in range(ncartj):
                mj = NLM_CART[lj][mlj, :]

                x = e0[mi[0]][mj[0]][..., 0, :, :]
                y = e0[mi[1]][mj[1]][..., 1, :, :]
                z = e0[mi[2]][mj[2]][..., 2, :, :]

                s3d[..., mli, mlj] += (sij * x * y * z).sum((-2, -1))

    # transform to cartesian basis functions (itrafo^T * S * jtrafo)
    o = torch.einsum("...ji,...jk,...kl->...il", itrafo, s3d, jtrafo)

    # remove small values
    eps = vec.new_tensor(torch.finfo(vec.dtype).eps)
    return torch.where(torch.abs(o) < eps, vec.new_tensor(0.0), o)

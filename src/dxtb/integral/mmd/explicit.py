"""
Calculation of overlap integrals using the McMurchie-Davidson algorithm.

- L. E. McMurchie, E. R. Davidson, One- and two-electron integrals over
  cartesian gaussian functions, *J. Comput. Phys.*, **1978**, *26*, 218-231.
  (`DOI <https://doi.org/10.1016/0021-9991(78)90092-X>`__)
"""
from __future__ import annotations

from math import pi, sqrt

import torch

from ..._types import Tensor
from ...utils import IntegralTransformError
from ..constants import NLM_CART, TRAFO

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
        if li == 0:
            e0 = ecoeffs_s(lj, xij, rpi, rpj)
        elif li == 1:
            e0 = ecoeffs_p(lj, xij, rpi, rpj)
        elif li == 2:
            e0 = ecoeffs_d(lj, xij, rpi, rpj)
        elif li == 3:
            e0 = ecoeffs_f(lj, xij, rpi, rpj)
        else:
            raise RuntimeError(f"Unsupported angular momentum {li}.")

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


def ecoeffs_s(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> list[list[Tensor]]:
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
    list[list[Tensor]]
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    RuntimeError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """

    # ss -> not required since 0
    if lj == 0:
        return [[torch.zeros_like(rpi)]]

    e011 = xij
    e100 = rpi
    e010 = rpj
    e000 = torch.ones_like(e100)

    # sp
    if lj == 1:
        return [
            [e000, e010],
        ]
    # sd
    if lj == 2:
        e020 = rpj * e010 + e011

        return [
            [e000, e010, e020],
        ]
    # sf
    if lj == 3:
        e020 = rpj * e010 + e011

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021

        return [
            [e000, e010, e020, e030],
        ]

    raise RuntimeError(f"Unsupported angular momentum {lj}.")


def ecoeffs_p(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> list[list[Tensor]]:
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
    list[list[Tensor]]
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    RuntimeError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    e101 = xij
    e011 = xij
    e100 = rpi
    e010 = rpj
    e110 = rpj * rpi + e101
    e000 = torch.ones_like(e100)

    # ps
    if lj == 0:
        return [
            [e000],
            [e100],
        ]
    # pp
    if lj == 1:
        return [
            [e000, e010],
            [e100, e110],
        ]
    # pd
    if lj == 3:
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e120 = rpj * e110 + e111

        return [
            [e000, e010, e020],
            [e100, e110, e120],
        ]
    # pf
    if lj == 2:
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e120 = rpj * e110 + e111

        e021 = xij * e010 + rpj * e011
        e030 = rpj * e020 + e021
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e130 = rpi * e030 + e031

        return [
            [e000, e010, e020, e030],
            [e100, e110, e120, e130],
        ]

    raise RuntimeError(f"Unsupported angular momentum {lj}.")


def ecoeffs_d(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> list[list[Tensor]]:
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
    list[list[Tensor]]
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    RuntimeError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    e101 = xij
    e011 = xij
    e100 = rpi
    e010 = rpj
    e110 = rpj * rpi + e101
    e000 = torch.ones_like(e100)

    e200 = rpi * e100 + e101

    # ds
    if lj == 0:
        return [
            [e000],
            [e100],
            [e200],
        ]
    # dp
    if lj == 1:
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111

        return [
            [e000, e010],
            [e100, e110],
            [e200, e210],
        ]
    # dd
    if lj == 2:
        e111 = xij * e100 + rpj * e101
        e112 = xij * e011
        e211 = xij * e110 + rpi * e111 + 2 * e112

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111
        e120 = rpj * e110 + e111
        e220 = rpj * e210 + e211

        return [
            [e000, e010, e020],
            [e100, e110, e120],
            [e200, e210, e220],
        ]
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

        return [
            [e000, e010, e020, e030],
            [e100, e110, e120, e130],
            [e200, e210, e220, e230],
        ]

    raise RuntimeError(f"Unsupported angular momentum {lj}.")


def ecoeffs_f(lj: Tensor, xij: Tensor, rpi: Tensor, rpj: Tensor) -> list:
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
    list[list[Tensor]]
        "Matrix" of E-coefficients. The shape depends on the angular momenta
        involved.

    Raises
    ------
    RuntimeError
        If angular momentum is not supported. Currently, the highest supported
        angular momentum is 3 (f-orbitals).
    """
    e101 = xij
    e011 = xij
    e100 = rpi
    e010 = rpj
    e110 = rpj * rpi + e101
    e000 = torch.ones_like(e100)

    e200 = rpi * e100 + e101

    e201 = xij * e200 + rpi * e101
    e300 = rpi * e200 + e201

    # fs
    if lj == 0:
        return [
            [e000],
            [e100],
            [e200],
            [e300],
        ]
    # fp
    if lj == 1:
        e111 = xij * e100 + rpj * e101

        e020 = rpj * e010 + e011
        e210 = rpi * e110 + e111

        e202 = xij * e101
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301

        return [
            [e000, e010],
            [e100, e110],
            [e200, e210],
            [e300, e310],
        ]
    # fd
    if lj == 2:
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
        e320 = rpi * e310 + e311

        return [
            [e000, e010, e020],
            [e100, e110, e120],
            [e200, e210, e220],
            [e300, e310, e320],
        ]
    # ff
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
        e202 = xij * e101
        e022 = xij * e011
        e031 = xij * e020 + rpj * e021 + 2 * e022
        e301 = xij * e200 + rpi * e201 + 2 * e202
        e310 = rpj * e300 + e301
        e130 = rpi * e030 + e031

        e302 = xij * e201 + rpi * e202
        e032 = xij * e021 + rpj * e022
        e311 = xij * e300 + rpj * e301 + 2 * e302
        e131 = xij * e030 + rpi * e031 + 2 * e032
        e320 = rpi * e310 + e311
        e230 = rpi * e130 + e131

        e033 = xij * e022
        e132 = xij * e031 + rpi * e032 + 3 * e033
        e231 = xij * e130 + rpi * e131 + 2 * e132
        e330 = rpi * e230 + e231

        return [
            [e000, e010, e020, e030],
            [e100, e110, e120, e130],
            [e200, e210, e220, e230],
            [e300, e310, e320, e330],
        ]

    raise RuntimeError(f"Unsupported angular momentum {lj}.")

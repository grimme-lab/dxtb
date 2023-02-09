"""
Calculation of overlap integrals using the McMurchie-Davidson algorithm.

- L. E. McMurchie, E. R. Davidson, One- and two-electron integrals over
  cartesian gaussian functions, *J. Comput. Phys.*, **1978**, *26*, 218-231.
  (`DOI <https://doi.org/10.1016/0021-9991(78)90092-X>`__)
"""
from __future__ import annotations

import math

import torch

from .._types import Any, Tensor
from ..utils import IntegralTransformError
from . import transform

sqrtpi = math.sqrt(math.pi)
sqrtpi3 = sqrtpi**3


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


_e_function_jit = torch.jit.trace(
    _e_function,
    (
        torch.rand((3, 5, 5, 11)),
        torch.tensor(1.0),
        torch.rand((3,)),
        torch.rand((3,)),
    ),
)


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


_e_function_grad_jit = torch.jit.trace(
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
)


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


e_function = EFunction.apply


nlm_cart = (
    torch.tensor(
        [
            [0, 0, 0],  # s
        ]
    ),
    torch.tensor(
        [
            # tblite order: x (+1), y (-1), z (0) in [-1, 0, 1] sorting
            [0, 1, 0],  # py
            [0, 0, 1],  # pz
            [1, 0, 0],  # px
        ]
    ),
    torch.tensor(
        [
            [2, 0, 0],  # dxx
            [1, 1, 0],  # dxy
            [1, 0, 1],  # dxz
            [0, 2, 0],  # dyy
            [0, 1, 1],  # dyz
            [0, 0, 2],  # dzz
        ]
    ),
    torch.tensor(
        [
            [3, 0, 0],  # fxxx
            [2, 1, 0],  # fxxy
            [2, 0, 1],  # fxxz
            [1, 2, 0],  # fxyy
            [1, 1, 1],  # fxyz
            [1, 0, 2],  # fxzz
            [0, 3, 0],  # fyyy
            [0, 2, 1],  # fyyz
            [0, 1, 2],  # fyzz
            [0, 0, 3],  # fzzz
        ]
    ),
    torch.tensor(
        [
            [4, 0, 0],  # gxxxx
            [3, 1, 0],  # gxxxy
            [3, 0, 1],  # gxxxz
            [2, 2, 0],  # gxxyy
            [2, 1, 1],  # gxxyz
            [2, 0, 2],  # gxxzz
            [1, 3, 0],  # gxyyy
            [1, 2, 1],  # gxyyz
            [1, 1, 2],  # gxyzz
            [1, 0, 3],  # gxzzz
            [0, 4, 0],  # gyyyy
            [0, 3, 1],  # gyyyz
            [0, 2, 2],  # gyyzz
            [0, 1, 3],  # gyzzz
            [0, 0, 4],  # gzzzz
        ]
    ),
)


def overlap(
    angular: tuple[Tensor, Tensor],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Calculate overlap integrals using McMurchie-Davidson algorithm.

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
        itrafo = transform.trafo[li].type(s3d.dtype).to(s3d.device)
        jtrafo = transform.trafo[lj].type(s3d.dtype).to(s3d.device)
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
    E = e_function(
        xij, rpi, rpj, (*vec.shape, ai.shape[-2], aj.shape[-1], li + 1, lj + 1)
    )
    for mli in range(s3d.shape[-2]):
        for mlj in range(s3d.shape[-1]):
            mi = nlm_cart[li][mli, :]
            mj = nlm_cart[lj][mlj, :]
            s3d[..., mli, mlj] += (
                sij
                * E[..., 0, :, :, mi[0], mj[0], 0]
                * E[..., 1, :, :, mi[1], mj[1], 0]
                * E[..., 2, :, :, mi[2], mj[2], 0]
            ).sum((-2, -1))

    # transform to cartesian basis functions (itrafo^T * S * jtrafo)
    o = torch.einsum("...ji,...jk,...kl->...il", itrafo, s3d, jtrafo)

    eps = vec.new_tensor(torch.finfo(vec.dtype).eps)
    g = torch.where(torch.abs(o) < eps, vec.new_tensor(0.0), o)
    return g


def e_function_derivative(e, ai, li, lj):
    # TODO: integrate into e_function

    de = torch.zeros((1, li + 3, lj + 3, 3))

    for m in range(3):
        for i in range(li + 1):
            for j in range(lj + 1):
                de[0, j, i, m] = 2 * ai * e[m, i + 1, j, 0]
                if i > 0:
                    de[0, j, i, m] -= i * e[m, i - 1, j, 0]

    return de


def overlap_gradient(angular: tuple, alpha: tuple, coeff: tuple, vec: Tensor):
    """Implementation identical to tblite overlap gradient."""

    # number of primitive gaussians
    nprim_i = len(alpha[0].nonzero(as_tuple=True)[0])
    nprim_j = len(alpha[1].nonzero(as_tuple=True)[0])

    # angular momenta and number of cartesian gaussian basis functions
    li, lj = angular
    ncarti = torch.div((li + 1) * (li + 2), 2, rounding_mode="floor")
    ncartj = torch.div((lj + 1) * (lj + 2), 2, rounding_mode="floor")

    # cartesian overlap and overlap gradient
    s3d = vec.new_zeros([ncarti, ncartj])
    ds3d = vec.new_zeros([3, ncarti, ncartj])

    r2 = torch.sum(vec.pow(2), -1)

    for ip in range(nprim_i):
        for jp in range(nprim_j):

            ai, aj = alpha[0][ip], alpha[1][jp]
            ci, cj = coeff[0][ip], coeff[1][jp]
            oij = 1.0 / (ai + aj)
            est = ai * aj * r2 * oij
            cc = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

            rpi = -vec * aj * oij
            rpj = +vec * ai * oij

            E = e_function(0.5 * oij, rpi, rpj, (3, li + 2, lj + 2))

            dE = e_function_derivative(E, ai, li, lj)

            for mli in range(ncarti):
                for mlj in range(ncartj):
                    mi = nlm_cart[li][mli, :]  # [3]
                    mj = nlm_cart[lj][mlj, :]  # [3]

                    e0 = torch.tensor(
                        [
                            E[..., 0, mi[0], mj[0], 0],
                            E[..., 1, mi[1], mj[1], 0],
                            E[..., 2, mi[2], mj[2], 0],
                        ]
                    )
                    d0 = torch.tensor(
                        [
                            dE[0, mj[0], mi[0], 0],
                            dE[0, mj[1], mi[1], 1],
                            dE[0, mj[2], mi[2], 2],
                        ]
                    )

                    # NOTE: calculating overlap for free
                    s3d[mli, mlj] += cc * torch.prod(e0)

                    grad = torch.tensor(
                        [
                            d0[0] * e0[1] * e0[2],
                            e0[0] * d0[1] * e0[2],
                            e0[0] * e0[1] * d0[2],
                        ]
                    )
                    ds3d[:, mli, mlj] += cc * grad

    # transform from cartesian to spherical gaussians
    try:
        itrafo = transform.trafo[li].type(ds3d.dtype).to(ds3d.device)
        jtrafo = transform.trafo[lj].type(ds3d.dtype).to(ds3d.device)
    except IndexError as e:
        raise IntegralTransformError() from e

    # transform to spherical basis functions (itrafo^T * S * jtrafo)
    sphr = [
        torch.einsum("...ji,...jk,...kl->...il", itrafo, ds3d[k, :, :], jtrafo)
        for k in range(ds3d.shape[-3])
    ]

    return torch.stack(sphr)  # [3, norb, norb]

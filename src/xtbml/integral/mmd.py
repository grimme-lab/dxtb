"""
Calculation of overlap integrals using the McMurchie-Davidson algorithm.

- L. E. McMurchie, E. R. Davidson, One- and two-electron integrals over
  cartesian gaussian functions, *J. Comput. Phys.*, **1978**, *26*, 218-231.
  (`DOI <https://doi.org/10.1016/0021-9991(78)90092-X>`__)
"""

import math
import torch

from . import transform
from ..exceptions import IntegralTransformError
from ..typing import Tensor, Tuple

sqrtpi = math.sqrt(math.pi)
sqrtpi3 = sqrtpi**3


@torch.jit.script
def e_function(
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    E: Tensor,
) -> Tensor:
    """
    Calculate E-coefficients for McMurchie-Davidson algorithm. Rather than computing
    them recursively, they are computed iteratively in this implementation.

    Parameters
    ----------
    xij : Tensor
        One over two times the product Gaussian exponent of the two shells.
    rpi : Tensor
        Distance between the center of the first shell and the product Gaussian center.
    rpj : Tensor
        Distance between the center of the second shell and the product Gaussian center.
    E : Tensor
        E-coefficients for the given shell pair.

    Returns
    -------
    Tensor
        E-coefficients for the given shell pair.
    """

    E[:] = 0.0
    E[..., 0, 0, 0] = 1.0
    for i in range(1, E.shape[-3]):
        E[..., i, 0, 0] = rpi * E[..., i - 1, 0, 0] + E[..., i - 1, 0, 1]
        for n in range(1, i):
            E[..., i, 0, n] = (
                xij * E[..., i - 1, 0, n - 1]
                + rpi * E[..., i - 1, 0, n]
                + (1 + n) * E[..., i - 1, 0, n + 1]
            )
        E[..., i, 0, i] = xij * E[..., i - 1, 0, i - 1] + rpi * E[..., i - 1, 0, i]
    for j in range(1, E.shape[-2]):
        for i in range(0, E.shape[-3]):
            E[..., i, j, 0] = rpj * E[..., i, j - 1, 0] + E[..., i, j - 1, 1]
            for n in range(1, i + j):
                E[..., i, j, n] = (
                    xij * E[..., i, j - 1, n - 1] + rpj * E[..., i, j - 1, n]
                )
                if n < E.shape[-1] - 1:
                    E[..., i, j, n].add_((1 + n) * E[..., i, j - 1, n + 1])
            E[..., i, j, i + j] = (
                xij * E[..., i, j - 1, i + j - 1] + rpj * E[..., i, j - 1, i + j]
            )

    return E


e_function_traced = torch.jit.trace(
    e_function,
    (
        torch.tensor(1.0),
        torch.rand((3,)),
        torch.rand((3,)),
        torch.rand((3, 5, 5, 11)),
    ),
)


nlm_cart = (
    torch.tensor(
        [
            [0, 0, 0],  # s
        ]
    ),
    torch.tensor(
        [
            [1, 0, 0],  # px
            [0, 1, 0],  # py
            [0, 0, 1],  # pz
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
    angular: Tuple[Tensor, Tensor],
    alpha: Tuple[Tensor, Tensor],
    coeff: Tuple[Tensor, Tensor],
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

    s3d = vec.new_zeros((*vec.shape[:-1], ncarti, ncartj))
    E = vec.new_zeros((*vec.shape, li + 1, lj + 1, li + lj + 1))

    try:
        itrafo = transform.trafo[li].type(s3d.dtype).to(s3d.device)
        jtrafo = transform.trafo[lj].type(s3d.dtype).to(s3d.device)
    except IndexError:
        raise IntegralTransformError()

    for ai, ci in zip(alpha[0], coeff[0]):
        for aj, cj in zip(alpha[1], coeff[1]):
            eij = ai + aj
            oij = 1.0 / eij
            xij = 0.5 * oij
            est = ai * aj * oij * r2
            sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj
            rpi = +vec * aj * oij
            rpj = -vec * ai * oij
            e_function_traced(xij, rpi, rpj, E)
            for mli in range(s3d.shape[-2]):
                for mlj in range(s3d.shape[-1]):
                    mi = nlm_cart[li][mli, :]
                    mj = nlm_cart[lj][mlj, :]
                    s3d[..., mli, mlj].add_(
                        sij
                        * E[..., 0, mi[0], mj[0], 0]
                        * E[..., 1, mi[1], mj[1], 0]
                        * E[..., 2, mi[2], mj[2], 0]
                    )

    overlap = torch.einsum("...ji,...jk,...kl->...il", itrafo, s3d, jtrafo)
    return overlap

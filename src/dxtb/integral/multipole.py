"""
Multipole Integrals
===================

Calculation and modification of multipole integrals.
"""

from __future__ import annotations

import torch

from .._types import Tensor
from ..utils import batch
from . import libcint as intor
from .overlap import OverlapLibcint


def multipole(
    driver: intor.LibcintWrapper | list[intor.LibcintWrapper],
    overlap: OverlapLibcint,
    intstring: str,
) -> Tensor:
    """
    Calculation of multipole integral. The integral is properly
    normalized, using the diagonal of the overlap integral.

    Parameters
    ----------
    driver : intor.LibcintWrapper
        Integral driver (libcint interface).
    norm : Tensor
        Norm of the overlap integral.

    Returns
    -------
    Tensor
        Normalized multipole integral.
    """

    # TODO: Better exception msg ("add to dxtblibs")
    # allowed_mps = ("j", "jj", "jjj")
    # if intstring not in allowed_mps:
    # raise ValueError("Unknown integral string provided.")

    def _mpint(driver: intor.LibcintWrapper, norm: Tensor) -> Tensor:
        return torch.einsum(
            "...ij,i,j->...ij", intor.int1e(intstring, driver), norm, norm
        )

    if isinstance(driver, list):
        # check batched mode again just to be sure
        if overlap.numbers.ndim != 2:
            raise RuntimeError(
                "Integral driver seems to be batched but atomic numbers not."
            )

        return batch.pack(
            [
                _mpint(drv, batch.deflate(overlap.norm[_batch]))
                for _batch, drv in enumerate(driver)
            ]
        )

    assert isinstance(driver, intor.LibcintWrapper)
    return _mpint(driver, overlap.norm)


def dipole(
    driver: intor.LibcintWrapper | list[intor.LibcintWrapper], overlap: OverlapLibcint
) -> Tensor:
    """
    Short-cut for dipole integral.

    Parameters
    ----------
    driver : intor.LibcintWrapper | list[intor.LibcintWrapper]
        Integral driver.
    overlap : OverlapLibcint
        Instance of the libcint overlap integral.

    Returns
    -------
    Tensor
        Dipole integral.
    """
    return multipole(driver, overlap, "r0")


def quadrupole(
    driver: intor.LibcintWrapper | list[intor.LibcintWrapper], overlap: OverlapLibcint
) -> Tensor:
    """
    Short-cut for quadrupole integral.

    Parameters
    ----------
    driver : intor.LibcintWrapper | list[intor.LibcintWrapper]
        Integral driver.
    overlap : OverlapLibcint
        Instance of the libcint overlap integral.

    Returns
    -------
    Tensor
        Quadrupole integral.
    """
    return multipole(driver, overlap, "r0r0")


################################################################################


def traceless(qpint: Tensor) -> Tensor:
    """
    Make a quadrupole tensor traceless.

    Parameters
    ----------
    qpint : Tensor
        Quadrupole moment tensor of shape `(..., 9, n, n)`.

    Returns
    -------
    Tensor
        Traceless Quadrupole moment tensor of shape
        `(..., 6, n, n)`.

    Raises
    ------
    RuntimeError
        Supplied quadrupole integral is no 3x3 tensor.

    Note
    ----
    First the quadrupole tensor is reshaped to be symmetric.
    Due to symmetry, only the lower triangular matrix is used.

    xx xy xz       0 1 2      0
    yx yy yz  <=>  3 4 5  ->  3 4
    zx zy zz       6 7 8      6 7 8
    """

    if qpint.shape[-3] == 9:
        # (..., 9, norb, norb) -> (..., 3, 3, norb, norb)
        shp = qpint.shape
        qpint = qpint.view(*shp[:-3], 3, 3, *shp[-2:])

        # trace: (..., 3, 3, norb, norb) -> (..., norb, norb)
        tr = 0.5 * torch.einsum("...iijk->...jk", qpint)

        return torch.stack(
            [
                1.5 * qpint[..., 0, 0, :, :] - tr,  # xx
                1.5 * qpint[..., 1, 0, :, :],  # yx
                1.5 * qpint[..., 1, 1, :, :] - tr,  # yy
                1.5 * qpint[..., 2, 0, :, :],  # zx
                1.5 * qpint[..., 2, 1, :, :],  # zy
                1.5 * qpint[..., 2, 2, :, :] - tr,  # zz
            ],
            dim=-3,
        )

    raise RuntimeError(f"Quadrupole integral must be 3x3 tensor but is {qpint.shape}")


def shift_r0_rj(r0: Tensor, overlap: Tensor, pos: Tensor) -> Tensor:
    r"""
    Shift the centering of the dipole integral (moment operator) from the origin
    (`r0 = r - (0, 0, 0)`) to atoms (ket index, `rj = r - r_j`).

    .. math::
        D &= D^{r_j}
          &= \langle i | r_j | j \rangle
          &= \langle i | r | j \rangle - r_j \langle i | j \rangle \\
          &= \langle i | r_0 | j \rangle - r_j S_{ij}
          &= D^{r_0} - r_j S_{ij}

    Parameters
    ----------
    r0 : Tensor
        Origin centered dipole integral.
    overlap : Tensor
        Overlap integral.
    pos : Tensor
        Orbital-resolved atomic positions.

    Raises
    ------
    RuntimeError
        Shape mismatch between `positions` and `overlap`.
        The positions must be orbital-resolved.

    Returns
    -------
    Tensor
        Second-index (ket) atom-centered dipole integral.
    """
    if pos.shape[-2] != overlap.shape[-1]:
        raise RuntimeError(
            "Shape mismatch between positions and overlap integral. "
            "The position tensor must be spread to orbital-resolution."
        )

    shift = torch.einsum("...jx,...ij->...xij", pos, overlap)
    return r0 - shift


def shift_r0r0_rjrj(r0r0: Tensor, r0: Tensor, overlap: Tensor, pos: Tensor) -> Tensor:
    """
    Shift the centering of the quadrupole integral (moment operator) from the
    origin (`r0 = r - (0, 0, 0)`) to atoms (ket index, `rj = r - r_j`).

    Parameters
    ----------
    r0r0 : Tensor
        Origin-centered quadrupole integral.
    r0 : Tensor
        Origin-centered dipole integral.
    overlap : Tensor
        Monopole integral (overlap).
    pos : Tensor
        Orbital-resolved atomic positions.

    Raises
    ------
    RuntimeError
        Shape mismatch between `positions` and `overlap`.
        The positions must be orbital-resolved.

    Returns
    -------
    Tensor
        Second-index (ket) atom-centered quadrupole integral.
    """
    if pos.shape[-2] != overlap.shape[-1]:
        raise RuntimeError(
            "Shape mismatch between positions and overlap integral. "
            "The position tensor must be spread to orbital-resolution."
        )

    # cartesian components for convenience
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    dpx = r0[..., 0, :, :]
    dpy = r0[..., 1, :, :]
    dpz = r0[..., 2, :, :]

    # construct shift contribution from dipole and monopole (overlap) moments
    shift_xx = shift_diagonal(x, dpx, overlap)
    shift_yy = shift_diagonal(y, dpy, overlap)
    shift_zz = shift_diagonal(z, dpz, overlap)
    shift_yx = shift_offdiag(y, x, dpy, dpx, overlap)
    shift_zx = shift_offdiag(z, x, dpz, dpx, overlap)
    shift_zy = shift_offdiag(z, y, dpz, dpy, overlap)

    # collect the trace of shift contribution
    tr = 0.5 * (shift_xx + shift_yy + shift_zz)

    return torch.stack(
        [
            r0r0[..., 0, :, :] + 1.5 * shift_xx - tr,  # xx
            r0r0[..., 1, :, :] + 1.5 * shift_yx,  # yx
            r0r0[..., 2, :, :] + 1.5 * shift_yy - tr,  # yy
            r0r0[..., 3, :, :] + 1.5 * shift_zx,  # zx
            r0r0[..., 4, :, :] + 1.5 * shift_zy,  # zy
            r0r0[..., 5, :, :] + 1.5 * shift_zz - tr,  # zz
        ],
        dim=-3,
    )


def shift_diagonal(c: Tensor, dpc: Tensor, s: Tensor) -> Tensor:
    r"""
    Create the shift contribution for all diagonal elements of the quadrupole
    integral.

    We start with the quadrupole integral generated by the `r0` moment operator:

    .. math::
        Q_{xx}^{r0} = \langle i | (r_x - r0)^2 | j \rangle = \langle i | r_x^2 | j \rangle

    Now, we shift the integral to `r_j` yielding the quadrupole integral center
    on the respective atoms:

    .. math::
        Q_{xx} &= \langle i | (r_x - r_{xj})^2 | j \rangle
               &= \langle i | r_x^2 | j \rangle - 2 \langle i | r_{xj} r_x | j \rangle + \langle i | r_{xj}^2 | j \rangle
               &= Q_{xx}^{r0} - 2 r_{xj} \langle i | r_x | j \rangle + r_{xj}^2 \langle i | j \rangle
               &= Q_{xx}^{r0} - 2 r_{xj} D_{x}^{r0} + r_{xj}^2 S_{ij}

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
    shift_1 = -2 * torch.einsum("...j,...ij->...ij", c, dpc)
    shift_2 = torch.einsum("...j,...j,...ij->...ij", c, c, s)
    return shift_1 + shift_2


def shift_offdiag(a: Tensor, b: Tensor, dpa: Tensor, dpb: Tensor, s: Tensor) -> Tensor:
    r"""
    Create the shift contribution for all off-diagonal elements of the
    quadrupole integral.

    .. math::
        Q_{ab} &= \langle i | (r_a - r_{aj})(r_b - r_{bj}) | j \rangle
               &= \langle i | r_a r_b | j \rangle - \langle i | r_a r_{bj} | j \rangle - \langle i | r_{aj} r_b | j \rangle + \langle i | r_{aj} r_{bj} | j \rangle \\
               &= Q_{ab}^{r0} - r_{bj} \langle i | r_a | j \rangle - r_{aj} \langle i | r_b | j \rangle + r_{aj} r_{bj} \langle i | j \rangle \\
               &= Q_{ab}^{r0} - r_{bj} D_a^{r0} - r_{aj} D_b^{r0} + r_{aj} r_{bj} S_{ij}

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
    shift_ab_1 = -torch.einsum("...j,...ij->...ij", b, dpa)
    shift_ab_2 = -torch.einsum("...j,...ij->...ij", a, dpb)
    shift_ab_3 = torch.einsum("...j,...j,...ij->...ij", a, b, s)

    return shift_ab_1 + shift_ab_2 + shift_ab_3

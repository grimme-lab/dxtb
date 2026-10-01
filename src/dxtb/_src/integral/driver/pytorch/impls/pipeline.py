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
Shared 3D assembly
==================

The 3D assembly, contraction and spherical transform of one class of shell
pairs, parameterized over a 1D ``kernel`` matching the ``compute_1d``
contract (McMurchie-Davidson or Obara-Saika).

``assemble_overlap_1d`` is the overlap (``emax == 0``),
``assemble_overlap_gradient_1d`` its derivative with respect to the bra
center, and ``assemble_multipole_1d`` the raw dipole (3) and quadrupole
(9 components, row-major) integrals about a common origin. All consume
per-class ``(angular, alpha, coeff, vec)`` inputs; the enumeration and
grouping of the shell pairs, screening, chunking and the scatter into the AO
matrix are done by ``impls/pairs.py``.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Callable

import torch

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import IntegralTransformError

from .md.trafo import NLM_CART, TRAFO

__all__ = [
    "assemble_overlap_1d",
    "assemble_overlap_gradient_1d",
    "assemble_multipole_1d",
    "Kernel1D",
]

sqrtpi3 = sqrt(pi) ** 3

Kernel1D = Callable[..., Tensor]


def _transforms(angular: tuple[int, int], vec: Tensor) -> tuple[Tensor, Tensor]:
    """Cartesian-to-spherical transforms of the bra and ket shells."""
    try:
        itrafo = TRAFO[angular[0]].type(vec.dtype).to(vec.device)
        jtrafo = TRAFO[angular[1]].type(vec.dtype).to(vec.device)
    except IndexError as e:
        raise IntegralTransformError() from e
    return itrafo, jtrafo


def _primitive_pairs(
    alpha: tuple[Tensor, Tensor], coeff: tuple[Tensor, Tensor], vec: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Quantities of all primitive pairs: the bra exponents ``ai`` (shape
    ``(nprimi, 1)``), ``xij = 1 / (2p)``, the displacements ``rpi = P - A``
    and ``rpj = P - B`` (shape ``(nvec, 3, nprimi, nprimj)``), and the
    prefactor ``sij`` of the 3D overlap of two s primitives times the
    contraction coefficients (shape ``(nvec, nprimi, nprimj)``).
    """
    ai, aj = alpha[0].unsqueeze(-1), alpha[1].unsqueeze(-2)
    ci, cj = coeff[0].unsqueeze(-1), coeff[1].unsqueeze(-2)
    oij = 1.0 / (ai + aj)
    xij = 0.5 * oij

    # no `einsum` (opt_einsum), which `torch.compile` cannot trace
    r2 = (vec * vec).sum(-1)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-2)
    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
    rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij
    return ai, xij, rpi, rpj, sij


def _per_axis(table: Tensor, angular: tuple[int, int]) -> list[Tensor]:
    """
    Gather the 1D factors of every Cartesian component pair from a table
    ``[i, j, ..., axis, p, q]``: one tensor ``(ncarti, ncartj, ..., p, q)``
    per axis. One broadcasting fancy-index call per axis gathers the bra and
    ket components at once.
    """
    nlmi = NLM_CART[angular[0]].to(table.device)
    nlmj = NLM_CART[angular[1]].to(table.device)
    return [
        table[nlmi[:, ax, None], nlmj[None, :, ax]].select(-3, ax)
        for ax in range(3)
    ]


def assemble_overlap_1d(
    kernel: Kernel1D,
    angular: tuple[int, int],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Assemble a shell pair's overlap from any ``compute_1d``-contract kernel.

    Parameters
    ----------
    kernel : Kernel1D
        A function matching the ``compute_1d(la, lb, emax, xij, rpi, rpj)``
        contract (e.g. MD's or OS's), called only for ``emax == 0`` here.
    angular : (int, int)
        Angular momentum of the shell pair(s).
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents of the shell pair(s).
    coeff : (Tensor, Tensor)
        Contraction coefficients of the shell pair(s).
    vec : Tensor
        Displacement vector between shell pair(s) of shape ``(nvec, 3)``.

    Returns
    -------
    Tensor
        Overlap integrals for the shell pair(s).
    """
    li, lj = angular
    itrafo, jtrafo = _transforms(angular, vec)
    _, xij, rpi, rpj, sij = _primitive_pairs(alpha, coeff, vec)

    if li == 0 and lj == 0:
        s3d = sij.sum((-2, -1), keepdim=True)
    else:
        sx, sy, sz = _per_axis(kernel(li, lj, 0, xij, rpi, rpj), angular)

        # fixed contraction, written out (no `einsum` path handling)
        s3d = (sx * sy * sz * sij).sum((-2, -1)).movedim((0, 1), (-2, -1))

    return itrafo @ s3d @ jtrafo.mT


def assemble_overlap_gradient_1d(
    kernel: Kernel1D,
    angular: tuple[int, int],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Derivative of a shell pair's overlap with respect to the bra center
    :math:`A` (the derivative with respect to the ket center is its
    negative).

    The derivative of a 1D primitive,
    :math:`\\partial_{A_x} (x - A_x)^i e^{-a (x - A_x)^2}
    = 2a (x - A_x)^{i+1} e^{\\ldots} - i (x - A_x)^{i-1} e^{\\ldots}`,
    turns the 1D overlap table of ``(la + 1, lb)`` into the 1D derivative
    table, so every kernel provides the gradient without derivative-specific
    code.

    Parameters
    ----------
    kernel : Kernel1D
        A function matching the ``compute_1d`` contract, called only for
        ``emax == 0``.
    angular : (int, int)
        Angular momentum of the shell pair(s).
    alpha : (Tensor, Tensor)
        Primitive Gaussian exponents of the shell pair(s).
    coeff : (Tensor, Tensor)
        Contraction coefficients of the shell pair(s).
    vec : Tensor
        ``B - A`` displacement, shape ``(nvec, 3)``.

    Returns
    -------
    Tensor
        Gradient of shape ``(nvec, 3, nsph_a, nsph_b)``.
    """
    li, lj = angular
    itrafo, jtrafo = _transforms(angular, vec)
    ai, xij, rpi, rpj, sij = _primitive_pairs(alpha, coeff, vec)

    # e1: (li+2, lj+1, nvec, 3, p, q)
    e1 = kernel(li + 1, lj, 0, xij, rpi, rpj)
    rows = [2.0 * ai * e1[1]]
    for i in range(1, li + 1):
        rows.append(2.0 * ai * e1[i + 1] - i * e1[i - 1])

    s = _per_axis(e1[: li + 1], angular)
    d = _per_axis(torch.stack(rows), angular)

    out = []
    for ax, (u, v) in enumerate(((1, 2), (0, 2), (0, 1))):
        cart = (d[ax] * s[u] * s[v] * sij).sum((-2, -1))
        out.append(itrafo @ cart.movedim((0, 1), (-2, -1)) @ jtrafo.mT)

    return torch.stack(out, dim=1)


DIPOLE_COMPONENTS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
"""Per-axis multipole order of each dipole component (x, y, z)."""

QUADRUPOLE_COMPONENTS = tuple(
    tuple(int(a == ax) + int(b == ax) for ax in range(3))
    for a in range(3)
    for b in range(3)
)
"""
Per-axis multipole order of each of the 9 raw quadrupole components,
row-major ``(x, y, z) x (x, y, z)`` (xx, xy, xz, yx, ...), matching libcint's
``int1e("r0r0")`` ordering.
"""


def assemble_multipole_1d(
    kernel: Kernel1D,
    angular: tuple[int, int],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
    pos_a: Tensor,
    components: tuple[tuple[int, int, int], ...],
    origin: Tensor | None = None,
) -> Tensor:
    """
    Assemble multipole integrals ``<a| x^ex y^ey z^ez |b>`` (about a common
    origin) for a shell pair from a ``compute_1d``-contract kernel with
    ``emax > 0``.

    Parameters
    ----------
    kernel : Kernel1D
        Kernel supporting ``emax > 0`` and the ``rpc`` keyword (OS).
    angular : (int, int)
        Angular momentum of the shell pair(s).
    alpha, coeff : (Tensor, Tensor)
        Primitive exponents and contraction coefficients.
    vec : Tensor
        ``B - A`` displacement, shape ``(nvec, 3)`` (same convention as
        :func:`assemble_overlap_1d`).
    pos_a : Tensor
        Absolute position of center ``A``, shape ``(nvec, 3)``.
    components : tuple of (int, int, int)
        Per-axis multipole orders, one entry per output component, e.g.
        :data:`DIPOLE_COMPONENTS` or :data:`QUADRUPOLE_COMPONENTS`.
    origin : Tensor | None, optional
        Multipole origin ``C`` (shape ``(3,)`` or ``(nvec, 3)``). Defaults to
        the Cartesian origin, matching libcint's ``r0`` operators.

    Returns
    -------
    Tensor
        Integrals of shape ``(nvec, ncomp, nsph_a, nsph_b)``.
    """
    li, lj = angular
    # plain loops: dynamo (torch 2.4) cannot trace a nested generator here
    emax = 0
    for comp in components:
        for order in comp:
            emax = max(emax, order)

    itrafo, jtrafo = _transforms(angular, vec)
    _, xij, rpi, rpj, sij = _primitive_pairs(alpha, coeff, vec)

    a_minus_c = pos_a if origin is None else pos_a - origin
    rpc = rpi + a_minus_c.unsqueeze(-1).unsqueeze(-1)

    e0 = kernel(li, lj, emax, xij, rpi, rpj, rpc=rpc)
    if emax == 0:
        # the kernels omit the (singleton) multipole axis for the overlap
        e0 = e0.unsqueeze(2)

    # per-axis tables: (ncarti, ncartj, e, nvec, nprimi, nprimj)
    tables = _per_axis(e0, angular)

    out = []
    for ex, ey, ez in components:
        prod = tables[0][:, :, ex] * tables[1][:, :, ey] * tables[2][:, :, ez]
        cart = (prod * sij).sum((-2, -1))  # (ncarti, ncartj, nvec)
        cart = cart.permute(2, 0, 1)
        out.append(itrafo @ cart @ jtrafo.mT)

    return torch.stack(out, dim=1)

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

One 3D-assembly/contraction/spherical-transform pipeline that every 1D kernel
(McMurchie-Davidson, Obara-Saika) plugs into. It is the pipeline of
``md_explicit`` (``impls/md/explicit.py``), parameterized over a ``kernel``
callable matching the ``compute_1d`` contract.

``assemble_overlap_1d`` is the overlap (``emax == 0``) assembly of one class
of shell pairs and ``assemble_multipole_1d`` its generalization to the raw
dipole (3) and quadrupole (9 components, row-major) integrals about a common
origin. Both consume per-class ``(angular, alpha, coeff, vec)`` inputs; the
enumeration and grouping of the shell pairs, screening, chunking and the
scatter into the AO matrix are done by ``impls/pairs.py``. Every kernel takes
``(xij, rpi, rpj[, xpc])``.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Callable

import torch
from tad_mctc.math import einsum

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import IntegralTransformError

from .md.trafo import NLM_CART, TRAFO

__all__ = ["assemble_overlap_1d", "assemble_multipole_1d", "Kernel1D"]

sqrtpi3 = sqrt(pi) ** 3

Kernel1D = Callable[..., Tensor]


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

    r2 = einsum("...i,...i->...", vec, vec)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-2)

    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    if li == 0 and lj == 0:
        s3d = sij.sum((-2, -1), keepdim=True)
    else:
        rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
        rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

        e0 = kernel(li, lj, 0, xij, rpi, rpj)

        nlmi = NLM_CART[li].to(vec.device)
        nlmj = NLM_CART[lj].to(vec.device)

        # one broadcasting fancy-index call per axis (gathers dims i and j at
        # once) instead of two chained ones; see assemble_multipole_1d
        sx = e0[nlmi[:, 0, None], nlmj[None, :, 0], ..., 0, :, :]  # type: ignore
        sy = e0[nlmi[:, 1, None], nlmj[None, :, 1], ..., 1, :, :]  # type: ignore
        sz = e0[nlmi[:, 2, None], nlmj[None, :, 2], ..., 2, :, :]  # type: ignore

        # fixed contraction, written out (no `einsum` path handling)
        s3d = (sx * sy * sz * sij).sum((-2, -1)).movedim((0, 1), (-2, -1))

    return itrafo @ s3d @ jtrafo.mT


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

    r2 = (vec * vec).sum(-1)
    est = ai * aj * oij * r2.unsqueeze(-1).unsqueeze(-1)
    sij = torch.exp(-est) * sqrtpi3 * torch.pow(oij, 1.5) * ci * cj

    rpi = +vec.unsqueeze(-1).unsqueeze(-1) * aj * oij
    rpj = -vec.unsqueeze(-1).unsqueeze(-1) * ai * oij

    a_minus_c = pos_a if origin is None else pos_a - origin
    rpc = rpi + a_minus_c.unsqueeze(-1).unsqueeze(-1)

    e0 = kernel(li, lj, emax, xij, rpi, rpj, rpc=rpc)
    if emax == 0:
        # the kernels omit the (singleton) multipole axis for the overlap
        e0 = e0.unsqueeze(2)

    nlmi = NLM_CART[li].to(vec.device)
    nlmj = NLM_CART[lj].to(vec.device)

    # per-axis tables: (ncarti, ncartj, e, nvec, nprimi, nprimj). One
    # broadcasting fancy-index call per axis gathers the bra and ket
    # components at once (e0 carries the full primitive-pair volume).
    tables = []
    for ax in range(3):
        t = e0[nlmi[:, ax, None], nlmj[None, :, ax]]
        tables.append(t.select(-3, ax))

    out = []
    for ex, ey, ez in components:
        prod = tables[0][:, :, ex] * tables[1][:, :, ey] * tables[2][:, :, ez]
        cart = (prod * sij).sum((-2, -1))  # (ncarti, ncartj, nvec)
        cart = cart.permute(2, 0, 1)
        out.append(itrafo @ cart @ jtrafo.mT)

    return torch.stack(out, dim=1)

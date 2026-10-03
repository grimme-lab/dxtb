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

``assemble_multipole_1d`` assembles the overlap (one component), the raw
dipole (3) and the raw quadrupole (9 components, row-major) integrals about a
common origin; ``assemble_overlap_1d`` is its single-component form. All consume
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

from .trafo import NLM_CART, TRAFO

__all__ = [
    "assemble_overlap_1d",
    "assemble_multipole_1d",
    "OVERLAP_COMPONENTS",
    "Kernel1D",
]

sqrtpi3 = sqrt(pi) ** 3

Kernel1D = Callable[..., Tensor]


def _transforms(
    angular: tuple[int, int], vec: Tensor
) -> tuple[Tensor | None, Tensor | None]:
    """
    Cartesian-to-spherical transforms of the bra and ket shells. ``None`` for
    s and p shells, whose transform is the identity.
    """
    out: list[Tensor | None] = []
    for l in angular:
        try:
            trafo = TRAFO[l]
        except IndexError as e:
            raise IntegralTransformError() from e

        out.append(
            None
            if l <= 1
            else torch.as_tensor(trafo, dtype=vec.dtype, device=vec.device)
        )

    return out[0], out[1]


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

    The axis is picked first with ``unbind``, whose backward is a single
    ``stack``. Selecting it after the gather (``.select(-3, ax)``) produces a
    long chain of ``select_scatter`` operations in the backward graph, which
    inductor miscompiles for ``compile(jacrev(...))`` of the quadrupole.
    """
    nlmi = torch.as_tensor(NLM_CART[angular[0]], device=table.device)
    nlmj = torch.as_tensor(NLM_CART[angular[1]], device=table.device)
    by_axis = table.unbind(-3)
    return [
        by_axis[ax][nlmi[:, ax, None], nlmj[None, :, ax]] for ax in range(3)
    ]


OVERLAP_COMPONENTS = ((0, 0, 0),)
"""Per-axis multipole order of the overlap (a single, zeroth-order component)."""

DIPOLE_COMPONENTS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
"""Per-axis multipole order of each dipole component (x, y, z)."""

QUADRUPOLE_COMPONENTS = (
    (2, 0, 0),  # xx
    (1, 1, 0),  # xy
    (1, 0, 1),  # xz
    (1, 1, 0),  # yx
    (0, 2, 0),  # yy
    (0, 1, 1),  # yz
    (1, 0, 1),  # zx
    (0, 1, 1),  # zy
    (0, 0, 2),  # zz
)
"""
Per-axis multipole order of each of the 9 raw quadrupole components,
row-major ``(x, y, z) x (x, y, z)`` (xx, xy, xz, yx, ...), matching
libcint's ``int1e("r0r0")`` ordering.
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

    # unique components (e.g., xy and yx are identical): contract only once
    unique = list(dict.fromkeys(components))

    if emax == 0 and li == 0 and lj == 0:
        # s-s overlap: no recursion needed
        s3d = sij.sum((-2, -1)).reshape(1, 1, -1)
        cart = [s3d] * len(unique)
    else:
        rpc = None
        if emax > 0:
            a_minus_c = pos_a if origin is None else pos_a - origin
            rpc = rpi + a_minus_c.unsqueeze(-1).unsqueeze(-1)

        e0 = kernel(li, lj, emax, xij, rpi, rpj, rpc=rpc)

        # per-axis tables: (ncarti, ncartj, e, nvec, nprimi, nprimj)
        tables = _per_axis(e0, angular)

        cart = []
        for ex, ey, ez in unique:
            prod = (
                tables[0][:, :, ex] * tables[1][:, :, ey] * tables[2][:, :, ez]
            )
            cart.append((prod * sij).sum((-2, -1)))  # (ncarti, ncartj, nvec)

    # single batched spherical transformation: (nvec, nuniq, ncarti, ncartj)
    cart = torch.stack(cart, dim=0).permute(3, 0, 1, 2)
    sph = cart
    if itrafo is not None:
        sph = itrafo @ sph
    if jtrafo is not None:
        sph = sph @ jtrafo.mT

    if len(unique) == len(components):
        return sph
    return sph[:, [unique.index(c) for c in components]]


def assemble_overlap_1d(
    kernel: Kernel1D,
    angular: tuple[int, int],
    alpha: tuple[Tensor, Tensor],
    coeff: tuple[Tensor, Tensor],
    vec: Tensor,
) -> Tensor:
    """
    Overlap of a shell pair: :func:`assemble_multipole_1d` with the single
    zeroth-order component.

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
        Overlap integrals of shape ``(nvec, nsph_a, nsph_b)``.
    """
    # a single (unbatched) displacement of shape ``(3,)`` is also accepted
    single = vec.ndim == 1
    if single:
        vec = vec.unsqueeze(0)

    # the multipole origin is irrelevant for the overlap
    out = assemble_multipole_1d(
        kernel, angular, alpha, coeff, vec, vec, OVERLAP_COMPONENTS
    )[:, 0]
    return out[0] if single else out

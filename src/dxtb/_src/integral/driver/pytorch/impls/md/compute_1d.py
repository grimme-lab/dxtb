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
Shared McMurchie-Davidson 1D kernel
====================================

Single entry point for the McMurchie-Davidson Hermite-expansion table used
by the overlap assembly. This is the shared ``compute_1d`` contract,
specialized to the McMurchie-Davidson algorithm and extracted, not rewritten,
from ``impls/md/explicit.py``'s inline E-coefficient dispatch so that the
overlap's numerics are unchanged.

Only ``emax == 0`` (the overlap case) is implemented. Multipole moments
(``emax > 0``) are provided by ``compute_1d_md_hermite``
(``impls/md/hermite.py``); calling this with ``emax > 0`` raises
``NotImplementedError`` rather than silently returning a wrong table.

Implementation note on axes: the three Cartesian axes (x, y, z) share the
same primitive-pair exponent data (``xij``) and differ only in the
center-to-Gaussian-product distances (``rpi``/``rpj``). ``ecoeffs_s/p/d/f``
already batch all three axes in one call via a shared trailing axis
dimension of size 3, which is more efficient than three separate calls with
identical primitive/exponent tensors. ``compute_1d`` therefore accepts
``rpi``/``rpj`` with that axis dimension already present (as the existing
code does) and returns the fused-axis table, rather than being called once
per axis.
"""

from __future__ import annotations

from dxtb._src.typing import Tensor
from dxtb._src.typing.exceptions import CGTOAzimuthalQuantumNumberError

from .explicit import ecoeffs_d, ecoeffs_f, ecoeffs_p, ecoeffs_s

__all__ = ["compute_1d"]


def compute_1d(
    la: int, lb: int, emax: int, xij: Tensor, rpi: Tensor, rpj: Tensor
) -> Tensor:
    """
    McMurchie-Davidson 1D (per-Hermite-index) table, fused over the three
    Cartesian axes.

    Parameters
    ----------
    la : int
        Angular momentum of the first (bra) center.
    lb : int
        Angular momentum of the second (ket) center.
    emax : int
        Maximum multipole order. Only ``0`` (overlap) is implemented.
    xij : Tensor
        Prefactor ``1 / (2p)`` with ``p = a + b`` from the Gaussian product
        theorem, broadcastable against ``rpi``/``rpj``.
    rpi : Tensor
        Distance between the Gaussian product center and center ``A``,
        shape ``(..., 3, nprimi, nprimj)``.
    rpj : Tensor
        Distance between the Gaussian product center and center ``B``,
        shape ``(..., 3, nprimi, nprimj)``.

    Returns
    -------
    Tensor
        Table indexed ``[i, j, ..., axis, nprimi, nprimj]`` for
        ``i <= la``, ``j <= lb``.

    Raises
    ------
    NotImplementedError
        If ``emax > 0``. Multipole tables are provided by
        ``impls/md/hermite.py`` (``compute_1d_md_hermite``).
    CGTOAzimuthalQuantumNumberError
        If ``la`` is not one of the supported angular momenta (s, p, d, f).
    """
    if emax != 0:
        raise NotImplementedError(
            "compute_1d: emax > 0 (multipole moments) is not supported by "
            "this explicit (hand-unrolled) table, which is overlap-only "
            "(emax == 0). Use `compute_1d_md_hermite` "
            "(impls/md/hermite.py) for multipoles."
        )

    if la == 0:
        return ecoeffs_s(lb, xij, rpi, rpj)
    if la == 1:
        return ecoeffs_p(lb, xij, rpi, rpj)
    if la == 2:
        return ecoeffs_d(lb, xij, rpi, rpj)
    if la == 3:
        return ecoeffs_f(lb, xij, rpi, rpj)

    raise CGTOAzimuthalQuantumNumberError(la)

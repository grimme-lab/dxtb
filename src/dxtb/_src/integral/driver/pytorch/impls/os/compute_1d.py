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
Obara-Saika 1D kernel (three-index vertical recursion)
=======================================================

Implements the three-index Obara-Saika recursion,

.. math::

    I_{i+1,j,e} = X_{PA} I_{i,j,e} + \\frac{1}{2p}\\left(i\\,I_{i-1,j,e}
    + j\\,I_{i,j-1,e} + e\\,I_{i,j,e-1}\\right)

with the same form for raising ``j`` (via :math:`X_{PB}`) and ``e`` (via
:math:`X_{PC}`), and start value :math:`I_{000} = 1` (unitless -- the
:math:`\\sqrt{\\pi/p}\\cdot\\exp(-\\mu X_{AB}^2)` 3D prefactor is applied by
the caller, exactly as for the McMurchie-Davidson ``compute_1d``, so the two
kernels are drop-in interchangeable for the same shell-pair class).

``emax`` up to 2 (overlap, dipole, quadrupole) is supported; ``emax > 0``
needs the multipole-origin distance ``rpc`` (:math:`X_{PC}`).

The two-index (i, j) DP table is filled with the standard Obara-Saika
overlap recursion, expressed with ``j`` as a spectator when raising ``i``:

.. math::

    T_{i,j} = X_{PA}\\,T_{i-1,j} + \\frac{1}{2p}\\left((i-1)\\,T_{i-2,j}
    + j\\,T_{i-1,j-1}\\right)

filled along ``i`` first for ``j = 0``, then along ``j`` for ``i = 0``, then
the general ``(i, j)`` cells, all with Python-int loop bounds (``la``,
``lb`` are Python ints, never traced tensors) so the recursion is
straight-line code once ``la``/``lb`` are fixed, as required for
``torch.compile``.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor

__all__ = ["compute_1d_os"]


def compute_1d_os(
    la: int,
    lb: int,
    emax: int,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    rpc: Tensor | None = None,
) -> Tensor:
    """
    Obara-Saika 1D table, fused over the three Cartesian axes, matching
    ``compute_1d``'s contract and inputs (``xij`` is ``1/(2p)``,
    ``rpi``/``rpj`` are :math:`X_{PA}`/:math:`X_{PB}`, all with a shared
    trailing axis dimension of size 3).

    Parameters
    ----------
    la : int
        Angular momentum of the first (bra) center.
    lb : int
        Angular momentum of the second (ket) center.
    emax : int
        Maximum multipole order (``0`` overlap, ``1`` dipole, ``2``
        quadrupole).
    xij : Tensor
        ``1 / (2p)`` with ``p = a + b``, broadcastable against ``rpi``/``rpj``.
    rpi : Tensor
        :math:`X_{PA}`, shape ``(..., 3, nprimi, nprimj)``.
    rpj : Tensor
        :math:`X_{PB}`, shape ``(..., 3, nprimi, nprimj)``.
    rpc : Tensor | None, optional
        :math:`X_{PC}` (product center minus multipole origin), same shape
        as ``rpi``. Required if ``emax > 0``.

    Returns
    -------
    Tensor
        For ``emax == 0``: ``[i, j, ..., axis, nprimi, nprimj]`` (matching
        ``compute_1d``). For ``emax > 0``: ``[i, j, e, ..., axis, nprimi,
        nprimj]`` with ``e <= emax``.

    Raises
    ------
    ValueError
        If ``emax > 0`` and ``rpc`` is not given.
    """
    if emax > 0 and rpc is None:
        raise ValueError("compute_1d_os: `rpc` is required for emax > 0.")

    # T[(i, j, e)]; missing keys (negative indices) contribute nothing
    t: dict[tuple[int, int, int], Tensor] = {
        (0, 0, 0): xij.new_ones(()).expand(rpi.shape)
    }

    def get(i: int, j: int, e: int) -> Tensor | None:
        return t.get((i, j, e))

    # small multiples of 1/(2p): `xk[n] = n / (2p)`, so that every recursion
    # term is one fused `addcmul` instead of a multiply and an add
    xk = [xij * n for n in range(la + lb + emax + 1)]

    def step(base: Tensor, n: int, prev: Tensor | None) -> Tensor:
        return base if prev is None else torch.addcmul(base, xk[n], prev)

    def mul(r: Tensor, key: tuple[int, int, int]) -> Tensor:
        # `t[(0, 0, 0)]` is all ones: skip the multiplication
        return r if key == (0, 0, 0) else r * t[key]

    for e in range(emax + 1):
        for i in range(la + 1):
            for j in range(lb + 1):
                if (i, j, e) in t:
                    continue

                if i > 0:
                    term = mul(rpi, (i - 1, j, e))
                    term = step(term, i - 1, get(i - 2, j, e))
                    term = step(term, j, get(i - 1, j - 1, e))
                    term = step(term, e, get(i - 1, j, e - 1))
                elif j > 0:
                    term = mul(rpj, (0, j - 1, e))
                    term = step(term, j - 1, get(0, j - 2, e))
                    term = step(term, e, get(0, j - 1, e - 1))
                else:
                    assert rpc is not None
                    term = mul(rpc, (0, 0, e - 1))
                    term = step(term, e - 1, get(0, 0, e - 2))

                t[(i, j, e)] = term

    # a single flat `stack` instead of nested per-axis ones, which copy the
    # whole intermediate again on every level
    shape = t[(0, 0, 0)].shape
    if emax == 0:
        vals = [t[(i, j, 0)] for i in range(la + 1) for j in range(lb + 1)]
        return torch.stack(vals, dim=0).reshape(la + 1, lb + 1, *shape)

    vals = [
        t[(i, j, e)]
        for i in range(la + 1)
        for j in range(lb + 1)
        for e in range(emax + 1)
    ]
    return torch.stack(vals, dim=0).reshape(la + 1, lb + 1, emax + 1, *shape)

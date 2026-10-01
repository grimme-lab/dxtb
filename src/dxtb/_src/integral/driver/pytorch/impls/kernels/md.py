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
McMurchie-Davidson 1D kernel with Hermite moments
=================================================

General McMurchie-Davidson ``compute_1d`` for the overlap and multipole
moments: Hermite expansion coefficients :math:`E^{ij}_t` by the standard
recursion and Hermite moments :math:`M^e_t` about the multipole origin
:math:`C`,

.. math::

    E^{i+1,j}_t = \\frac{1}{2p} E^{ij}_{t-1} + X_{PA} E^{ij}_t
    + (t+1) E^{ij}_{t+1},

.. math::

    I_{i,j,e} = \\sum_t E^{ij}_t M^e_t, \\quad
    M^{e+1}_t = t M^e_{t-1} + X_{PC} M^e_t + \\frac{1}{2p} M^e_{t+1},

(Helgaker, Jorgensen, Olsen, *Molecular Electronic-Structure Theory*, ch. 9),
with the unitless start values :math:`E^{00}_0 = M^0_0 = 1`; the
:math:`\\sqrt{\\pi/p}\\exp(-\\mu X_{AB}^2)` factor is applied by the caller, as
for every kernel. Unlike the explicit E-coefficients of the legacy code
(``impls/legacy/explicit.py``), this is not hand-unrolled per angular
momentum: it works for any ``la``, ``lb`` with Python-int loops and no in-place
writes.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Tensor

__all__ = ["compute_1d_md_hermite"]


def compute_1d_md_hermite(
    la: int,
    lb: int,
    emax: int,
    xij: Tensor,
    rpi: Tensor,
    rpj: Tensor,
    rpc: Tensor | None = None,
) -> Tensor:
    """
    McMurchie-Davidson 1D table, fused over the three Cartesian axes; same
    inputs and output layout as ``compute_1d_os``.

    Raises
    ------
    ValueError
        If ``emax > 0`` and ``rpc`` is not given.
    """
    if emax > 0 and rpc is None:
        raise ValueError(
            "compute_1d_md_hermite: `rpc` is required for emax > 0."
        )

    one = xij.new_ones(()).expand(rpi.shape)

    def total(terms: list[Tensor]) -> Tensor:
        acc = terms[0]
        for term in terms[1:]:
            acc = acc + term
        return acc

    # E[(i, j)] = list over t = 0 .. i + j
    e_tab: dict[tuple[int, int], list[Tensor]] = {(0, 0): [one]}

    def step(prev: list[Tensor], x: Tensor) -> list[Tensor]:
        n = len(prev)
        out = []
        for t in range(n + 1):
            terms = []
            if t >= 1:
                terms.append(xij * prev[t - 1])
            if t < n:
                terms.append(x * prev[t])
            if t + 1 < n:
                terms.append((t + 1) * prev[t + 1])
            out.append(total(terms))
        return out

    for i in range(1, la + 1):
        e_tab[(i, 0)] = step(e_tab[(i - 1, 0)], rpi)
    for i in range(la + 1):
        for j in range(1, lb + 1):
            e_tab[(i, j)] = step(e_tab[(i, j - 1)], rpj)

    # Hermite moments M[e] = list over t = 0 .. e
    m_tab: list[list[Tensor]] = [[one]]
    for e in range(emax):
        assert rpc is not None
        prev = m_tab[e]
        n = len(prev)
        out = []
        for t in range(n + 1):
            terms = []
            if t >= 1:
                terms.append(t * prev[t - 1])
            if t < n:
                terms.append(rpc * prev[t])
            if t + 1 < n:
                terms.append(xij * prev[t + 1])
            out.append(total(terms))
        m_tab.append(out)

    def contract(i: int, j: int, e: int) -> Tensor:
        ets, mts = e_tab[(i, j)], m_tab[e]
        return total([ets[t] * mts[t] for t in range(min(len(ets), len(mts)))])

    # a single flat `stack` instead of nested per-axis ones, which copy the
    # whole intermediate again on every level
    shape = rpi.shape
    if emax == 0:
        vals = [e_tab[(i, j)][0] for i in range(la + 1) for j in range(lb + 1)]
        return torch.stack(vals, dim=0).reshape(la + 1, lb + 1, *shape)

    vals = [
        contract(i, j, e)
        for i in range(la + 1)
        for j in range(lb + 1)
        for e in range(emax + 1)
    ]
    return torch.stack(vals, dim=0).reshape(la + 1, lb + 1, emax + 1, *shape)

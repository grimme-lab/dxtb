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
1D kernels (``compute_1d`` contract) and the assembly of one class of shell
pairs.
"""

from __future__ import annotations

import math

import pytest
import torch

from dxtb._src.integral.driver.pytorch.impls.kernels import (
    ALGORITHMS,
    get_kernel,
)
from dxtb._src.integral.driver.pytorch.impls.kernels.md import (
    compute_1d_md_hermite,
)
from dxtb._src.integral.driver.pytorch.impls.kernels.os import compute_1d_os
from dxtb._src.integral.driver.pytorch.impls.pipeline import (
    DIPOLE_COMPONENTS,
    QUADRUPOLE_COMPONENTS,
    assemble_multipole_1d,
    assemble_overlap_1d,
)
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE


def pair_quantities(dd: DD) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``xij``, ``rpi``, ``rpj`` and ``rpc`` of random primitive pairs."""
    gen = torch.Generator(device="cpu").manual_seed(0)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=gen, device="cpu").to(**dd)

    a = (rand(3) * 3 + 0.1).unsqueeze(-1)
    b = (rand(4) * 3 + 0.1).unsqueeze(-2)
    vec = (rand(1, 3) * 2 - 1).unsqueeze(-1).unsqueeze(-1)

    xij = 0.5 / (a + b)
    rpi = vec * b / (a + b)
    rpj = -vec * a / (a + b)
    rpc = rpi + rand(3, 1, 1)
    return xij, rpi, rpj, rpc


def class_inputs(dd: DD) -> tuple[Tensor, ...]:
    """Exponents, coefficients, displacements and positions of one class."""
    gen = torch.Generator(device="cpu").manual_seed(1)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=gen, device="cpu").to(**dd)

    return (
        rand(2) + 0.3,
        rand(3) + 0.3,
        rand(2),
        rand(3),
        rand(2, 3) - 0.5,
        rand(2, 3),
    )


@pytest.mark.parametrize("la", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("lb", [0, 1, 2, 3, 4])
def test_md_matches_os(la: int, lb: int) -> None:
    """The shipped bases have l <= 2; the kernels are checked up to g."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    xij, rpi, rpj, rpc = pair_quantities(dd)

    for emax in range(3):
        ref = compute_1d_os(la, lb, emax, xij, rpi, rpj, rpc)
        out = compute_1d_md_hermite(la, lb, emax, xij, rpi, rpj, rpc)
        assert (out - ref).abs().max() <= 1e-13 * max(
            1.0, float(ref.abs().max())
        )


def test_one_center_closed_form() -> None:
    """
    For coincident centers, the table holds the moments of a centered
    Gaussian, :math:`(n-1)!! / (2p)^{n/2}` for even :math:`n = i + j` and
    zero for odd :math:`n`.
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    p = 1.7
    xij = torch.full((1, 1, 1, 1), 0.5 / p, **dd)
    zero = torch.zeros((1, 1, 1, 1), **dd)

    table = compute_1d_os(2, 2, 0, xij, zero, zero)

    for i in range(3):
        for j in range(3):
            n = i + j
            ref = 0.0
            if n % 2 == 0:
                ref = math.prod(range(n - 1, 0, -2)) / (2.0 * p) ** (n / 2)
            assert table[i, j].item() == pytest.approx(ref, abs=1e-14)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_overlap_is_zeroth_multipole(algorithm: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    kernel = get_kernel(algorithm)
    a, b, c, d, vec, pos_a = class_inputs(dd)

    for la, lb in [(0, 0), (1, 2), (2, 2)]:
        ref = assemble_overlap_1d(kernel, (la, lb), (a, b), (c, d), vec)
        out = assemble_multipole_1d(
            kernel, (la, lb), (a, b), (c, d), vec, pos_a, ((0, 0, 0),)
        )
        assert torch.allclose(out[:, 0], ref, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_extreme_exponents_against_arbitrary_precision(algorithm: str) -> None:
    """A tight (1e4) and a diffuse (1e-2) s primitive: <s|r|s> = S P."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50

    dd: DD = {"dtype": torch.double, "device": DEVICE}
    a, b = 1.0e4, 1.0e-2
    ra, rb = [0.1, -0.2, 0.3], [0.15, -0.1, 0.25]
    ab = [y - x for x, y in zip(ra, rb)]

    out = assemble_multipole_1d(
        get_kernel(algorithm),
        (0, 0),
        (torch.tensor([a], **dd), torch.tensor([b], **dd)),
        (torch.ones(1, **dd), torch.ones(1, **dd)),
        torch.tensor([ab], **dd),
        torch.tensor([ra], **dd),
        DIPOLE_COMPONENTS,
    )[0, :, 0, 0]

    p = mp.mpf(a) + mp.mpf(b)
    s = (mp.pi / p) ** mp.mpf("1.5") * mp.exp(
        -mp.mpf(a) * mp.mpf(b) / p * sum(mp.mpf(x) ** 2 for x in ab)
    )
    for k in range(3):
        ref = s * (mp.mpf(a) * mp.mpf(ra[k]) + mp.mpf(b) * mp.mpf(rb[k])) / p
        assert out[k].item() == pytest.approx(float(ref), rel=1e-12)


@pytest.mark.grad
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_gradcheck(algorithm: str) -> None:
    """First and second derivatives w.r.t. exponents, coefficients,
    displacements, positions and the multipole origin."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    kernel = get_kernel(algorithm)
    inputs = tuple(t.requires_grad_(True) for t in class_inputs(dd))
    origin = torch.tensor([0.3, -0.2, 0.1], **dd, requires_grad=True)

    def overlap(a, b, c, d, vec):
        return assemble_overlap_1d(kernel, (2, 1), (a, b), (c, d), vec)

    def quadrupole(a, b, c, d, vec, pos_a, origin):
        return assemble_multipole_1d(
            kernel,
            (2, 1),
            (a, b),
            (c, d),
            vec,
            pos_a,
            QUADRUPOLE_COMPONENTS,
            origin,
        )

    for func, args in ((overlap, inputs[:5]), (quadrupole, (*inputs, origin))):
        assert torch.autograd.gradcheck(func, args, atol=1e-6)
        # the full second-order check takes about 10 s per kernel
        assert torch.autograd.gradgradcheck(
            func, args, atol=1e-5, fast_mode=True
        )


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported for this Python/torch combination",
)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_traceable_without_graph_breaks(algorithm: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    a, b, c, d, vec, pos_a = class_inputs(dd)
    kernel = get_kernel(algorithm)

    def f(vec: Tensor, pos_a: Tensor) -> Tensor:
        return assemble_multipole_1d(
            kernel, (2, 2), (a, b), (c, d), vec, pos_a, QUADRUPOLE_COMPONENTS
        )

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(f)(vec, pos_a)
    torch._dynamo.reset()

    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

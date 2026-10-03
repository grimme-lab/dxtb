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
Integral screening: the shell-pair bound is a rigorous upper bound of the
integral block (overlap, dipole, quadrupole; any multipole origin, scaled
exponents), and the screened matrices differ from the unscreened ones by less
than the threshold.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck

from dxtb import GFN2_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.impls.kernels.os import compute_1d_os
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    _bounds,
    _normalized,
    assemble_matrix,
    prepare,
    select_pairs,
)
from dxtb._src.integral.driver.pytorch.impls.pipeline import (
    DIPOLE_COMPONENTS,
    QUADRUPOLE_COMPONENTS,
)
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples

COMPONENTS = {
    "S": None,
    "D": DIPOLE_COMPONENTS,
    "Q": QUADRUPOLE_COMPONENTS,
}

# SiH4 + NH3 cover s, p and d shells
NAMES = ("SiH4", "NH3", "H2O", "LiH")


def _setup(name: str, dd: DD, ascale: float):
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    alphas, coeffs = Basis(numbers, par, ihelp, **dd).create_cgtos()
    return ihelp, [a * ascale for a in alphas], coeffs, positions


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("ascale", [1.0, 0.3])
@pytest.mark.parametrize("shift", [0.0, 50.0])
@pytest.mark.parametrize("op", COMPONENTS)
def test_bound(name: str, ascale: float, shift: float, op: str) -> None:
    """The bound is never smaller than the largest element of a block."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _setup(name, dd, ascale)
    positions = positions + shift

    comps = COMPONENTS[op]
    emax = 0 if comps is None else max(max(c) for c in comps)
    mat = assemble_matrix(
        compute_1d_os, ihelp, alphas, coeffs, positions, comps
    )

    plan = prepare(ihelp, positions.device)
    coeffs = _normalized(compute_1d_os, plan, alphas, coeffs)
    off = (2 * ihelp.angular + 1).cumsum(0) - (2 * ihelp.angular + 1)

    for cl in plan.classes:
        inter = plan.atom[cl.ib] != plan.atom[cl.jk]  # different atoms
        if not inter.any():
            continue
        ib, jk = cl.ib[inter], cl.jk[inter]
        bound = _bounds(
            positions[plan.atom[ib]],
            positions[plan.atom[jk]],
            alphas[cl.ub],
            alphas[cl.uk],
            coeffs[cl.ub],
            coeffs[cl.uk],
            (cl.lb, cl.lk),
            (emax,),
            None,
        )[0]
        for n, (i, j) in enumerate(zip(ib.tolist(), jk.tolist())):
            blk = mat[
                :,
                off[i] : off[i] + 2 * cl.lb + 1,
                off[j] : off[j] + 2 * cl.lk + 1,
            ]
            assert blk.abs().max() <= bound[n]


def test_bounds_orders_consistent() -> None:
    """Bounds of several orders at once equal the bounds of each order."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _setup("SiH4", dd, 1.0)

    plan = prepare(ihelp, positions.device)
    coeffs = _normalized(compute_1d_os, plan, alphas, coeffs)

    for cl in plan.classes:
        args = (
            positions[plan.atom[cl.ib]],
            positions[plan.atom[cl.jk]],
            alphas[cl.ub],
            alphas[cl.uk],
            coeffs[cl.ub],
            coeffs[cl.uk],
            (cl.lb, cl.lk),
        )
        both = _bounds(*args, (0, 1, 2), None)
        for i, e in enumerate((0, 1, 2)):
            assert torch.equal(both[i], _bounds(*args, (e,), None)[0])


def _two_molecules(dd: DD, gap: float, ascale: float):
    """SiH4 and a smaller NH3-like fragment ``gap`` bohr apart."""
    numbers = torch.tensor([14, 1, 1, 1, 1, 7, 1, 1, 1], device=DEVICE)
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.6, 1.6, 1.6],
            [-1.6, -1.6, 1.6],
            [-1.6, 1.6, -1.6],
            [1.6, -1.6, -1.6],
        ],
        **dd,
    )
    shift = torch.tensor([gap, 0.0, 0.0], **dd)
    positions = torch.cat([base, base[:4] * 0.6 + shift])
    ihelp = IndexHelper.from_numbers(numbers, par)
    alphas, coeffs = Basis(numbers, par, ihelp, **dd).create_cgtos()
    return ihelp, [a * ascale for a in alphas], coeffs, positions


@pytest.mark.parametrize("op", COMPONENTS)
@pytest.mark.parametrize("threshold", [1e-4, 1e-8, 1e-12])
def test_error(op: str, threshold: float) -> None:
    """Dropped elements are smaller than the threshold."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 16.0, 1.0)

    comps = COMPONENTS[op]
    args = (compute_1d_os, ihelp, alphas, coeffs, positions, comps)
    ref = assemble_matrix(*args)
    scr = assemble_matrix(*args, screening_threshold=threshold)

    # the test is only meaningful if something was dropped
    assert ((scr == 0) & (ref != 0)).any()
    assert (scr - ref).abs().max() <= threshold


@pytest.mark.parametrize("threshold", [1e-4, 1e-3])
def test_positive_definite(threshold: float) -> None:
    """The overlap of a diffuse basis stays positive definite."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 32.0, 0.3)

    args = (compute_1d_os, ihelp, alphas, coeffs, positions, None)
    ref = assemble_matrix(*args)
    scr = assemble_matrix(*args, screening_threshold=threshold)

    assert ((scr == 0) & (ref != 0)).any()
    assert torch.linalg.eigvalsh(scr[0]).min() > 0


@pytest.mark.parametrize("op", ["S", "Q"])
def test_gradcheck(op: str) -> None:
    """Gradients of the screened matrix agree with finite differences."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 16.0, 1.0)
    comps = COMPONENTS[op]
    threshold = 1e-6

    ref = assemble_matrix(
        compute_1d_os, ihelp, alphas, coeffs, positions, comps
    )
    scr = assemble_matrix(
        compute_1d_os,
        ihelp,
        alphas,
        coeffs,
        positions,
        comps,
        screening_threshold=threshold,
    )
    assert ((scr == 0) & (ref != 0)).any()

    def func(pos: Tensor) -> Tensor:
        return assemble_matrix(
            compute_1d_os,
            ihelp,
            alphas,
            coeffs,
            pos,
            comps,
            screening_threshold=threshold,
        )

    pos = positions.clone().requires_grad_(True)
    assert dgradcheck(func, pos, atol=1e-6)


def _matrix(ihelp, alphas, coeffs, comps, **kwargs):
    return lambda pos: assemble_matrix(
        compute_1d_os, ihelp, alphas, coeffs, pos, comps, **kwargs
    )


@pytest.mark.parametrize("op", COMPONENTS)
def test_pairs_equal_threshold(op: str) -> None:
    """A selection gives the same matrix as the threshold it was made with."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 16.0, 1.0)
    comps = COMPONENTS[op]
    threshold = 1e-8

    pairs = select_pairs(
        compute_1d_os, ihelp, alphas, coeffs, positions, threshold
    )
    by_threshold = _matrix(
        ihelp, alphas, coeffs, comps, screening_threshold=threshold
    )(positions)
    by_pairs = _matrix(ihelp, alphas, coeffs, comps, pairs=pairs)(positions)
    full = _matrix(ihelp, alphas, coeffs, comps)(positions)

    assert torch.equal(by_pairs, by_threshold)
    assert ((by_pairs == 0) & (full != 0)).any()


def test_pairs_arguments() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 16.0, 1.0)
    pairs = select_pairs(compute_1d_os, ihelp, alphas, coeffs, positions, 1e-8)

    with pytest.raises(ValueError):
        _matrix(
            ihelp, alphas, coeffs, None, pairs=pairs, screening_threshold=1e-8
        )(positions)
    with pytest.raises(ValueError):
        _matrix(ihelp, alphas, coeffs, None, pairs=pairs[:-1])(positions)


@pytest.mark.parametrize("op", ["S", "Q"])
def test_vmap(op: str) -> None:
    """
    With a selection over the whole batch, ``vmap`` equals the loop and stays
    within the threshold of the unscreened matrices, while ``vmap`` of the
    threshold itself cannot work (data-dependent size).
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 16.0, 1.0)
    comps = COMPONENTS[op]
    threshold = 1e-8

    batch = torch.stack([positions, positions + 0.05 * torch.sin(positions)])
    pairs = select_pairs(compute_1d_os, ihelp, alphas, coeffs, batch, threshold)

    screened = _matrix(ihelp, alphas, coeffs, comps, pairs=pairs)
    full = _matrix(ihelp, alphas, coeffs, comps)

    # `vmap` uses batched matmuls, whose summation order may differ from the
    # loop at the level of rounding errors (platform dependent)
    out = torch.vmap(screened)(batch)
    loop = torch.stack([screened(b) for b in batch])
    assert torch.allclose(out, loop, atol=1e-13, rtol=0.0)
    ref = torch.stack([full(b) for b in batch])
    assert (out - ref).abs().max() <= threshold
    assert ((out == 0) & (ref != 0)).any()

    with pytest.raises(RuntimeError):
        torch.vmap(
            _matrix(ihelp, alphas, coeffs, comps, screening_threshold=threshold)
        )(batch)


def test_jacrev_pairs() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    ihelp, alphas, coeffs, positions = _two_molecules(dd, 16.0, 1.0)
    pairs = select_pairs(compute_1d_os, ihelp, alphas, coeffs, positions, 1e-8)
    f = _matrix(ihelp, alphas, coeffs, None, pairs=pairs)
    g = _matrix(ihelp, alphas, coeffs, None, screening_threshold=1e-8)

    def scalar(fn):
        return lambda pos: (fn(pos) ** 2).sum()

    assert torch.equal(
        torch.func.jacrev(scalar(f))(positions),
        torch.func.jacrev(scalar(g))(positions),
    )

    # the test suite disables the TorchScript JIT, but torch scripts its
    # forward-mode decompositions on first use
    jit_was_enabled = torch.jit._state._enabled.enabled  # type: ignore
    torch.jit._state.enable()  # type: ignore
    try:
        fwd = torch.func.jacfwd(scalar(f))(positions)
        hess = torch.func.jacfwd(torch.func.jacrev(scalar(f)))(positions)
    finally:
        if not jit_was_enabled:
            torch.jit._state.disable()  # type: ignore

    assert torch.allclose(
        fwd, torch.func.jacrev(scalar(g))(positions), atol=1e-9, rtol=1e-12
    )
    assert torch.allclose(
        hess,
        torch.func.jacrev(torch.func.jacrev(scalar(g)))(positions),
        atol=1e-8,
        rtol=1e-12,
    )


def _compile_backend() -> str:
    """
    ``inductor`` needs a C++ compiler (missing, e.g., on CI runners without
    MSVC); the graph capture is checked with ``aot_eager`` otherwise.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from torch._inductor.cpp_builder import get_cpp_compiler

        get_cpp_compiler()
    except Exception:  # pylint: disable=broad-exception-caught
        return "aot_eager"
    return "inductor"


@pytest.mark.skipif(
    not torch._dynamo.is_dynamo_supported(),
    reason="torch.compile is not supported for this Python/torch combination",
)
def test_compile() -> None:
    """``fullgraph`` compilation, unscreened and with a pair selection."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([8, 1, 1, 8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 1.1, 0.0],
            [-1.4, 1.1, 0.0],
            [16.5, 0.3, 0.2],
            [17.9, 1.4, 0.2],
            [15.1, 1.4, 0.2],
        ],
        **dd,
    )
    ihelp = IndexHelper.from_numbers(numbers, par)
    alphas, coeffs = Basis(numbers, par, ihelp, **dd).create_cgtos()
    plan = prepare(ihelp, positions.device)

    pairs = select_pairs(
        compute_1d_os, ihelp, alphas, coeffs, positions, 1e-8, plan=plan
    )
    for kwargs in ({}, {"pairs": pairs}):
        fn = _matrix(ihelp, alphas, coeffs, None, plan=plan, **kwargs)
        torch._dynamo.reset()
        out = torch.compile(fn, fullgraph=True, backend=_compile_backend())(
            positions
        )
        assert torch.allclose(out, fn(positions), atol=1e-13, rtol=0.0)

    # the data-dependent selection by threshold is refused with a hint
    torch._dynamo.reset()
    with pytest.raises(Exception, match="select_pairs"):
        torch.compile(
            _matrix(
                ihelp,
                alphas,
                coeffs,
                None,
                plan=plan,
                screening_threshold=1e-8,
            ),
            fullgraph=True,
            backend=_compile_backend(),
        )(positions)

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
Tests of the implicit fixed-point Function on toy problems with a known
answer (unrolled iterations of many steps, started at the solution).
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from dxtb import OutputHandler
from dxtb._src.scf.implicit.fixed_point import (
    AdjointOptions,
    _grad_leaves,
    equilibrium,
)

DD = {"device": torch.device("cpu"), "dtype": torch.double}
FWD = {"f_tol": 1e-13, "x_tol": 1e-13, "maxiter": 500, "method": "broyden1"}


class Toy:
    """``g(x, W, p) = 0.4 tanh(W x) + p`` (contractive for small ``W``)."""

    def __init__(self, n: int = 5, batch: int | None = None, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        shape = (n, n) if batch is None else (batch, n, n)
        self.W = (
            0.5 * torch.randn(shape, generator=gen, **DD)
        ).requires_grad_()
        pshape = (n,) if batch is None else (batch, n)
        self.p = torch.randn(pshape, generator=gen, **DD).requires_grad_()
        self.x0 = torch.zeros(pshape, **DD)

        W, p = self.W, self.p

        def g(x: torch.Tensor) -> torch.Tensor:
            return 0.4 * torch.tanh((W @ x.unsqueeze(-1)).squeeze(-1)) + p

        self.g = g  # plain function: closes over the parameters only

    def implicit(self, **kw) -> torch.Tensor:
        x, _ = equilibrium(
            self.g, self.x0, batched=self.p.dim() == 2, **FWD, **kw
        )
        return x

    def unrolled(self, steps: int = 400) -> torch.Tensor:
        """Reference: differentiate through all iterations."""
        with torch.no_grad():
            x = self.implicit()
        for _ in range(steps):
            x = self.g(x)
        return x


def close(a: torch.Tensor, b: torch.Tensor, rtol: float) -> bool:
    """Max abs deviation relative to the largest reference entry (the forward
    solve is only converged to ~1e-11, which limits the comparison)."""
    return (a - b).abs().max().item() < rtol * max(1.0, b.abs().max().item())


def loss(x: torch.Tensor) -> torch.Tensor:
    return (x**3).sum() + (x[..., 0] * x[..., 1]).sum()


@pytest.mark.parametrize("batch", [None, 3])
def test_first_derivative(batch: int | None) -> None:
    """Gradient equals the gradient of the unrolled iteration."""
    t = Toy(batch=batch)
    gi = torch.autograd.grad(loss(t.implicit()), [t.W, t.p])
    gu = torch.autograd.grad(loss(t.unrolled()), [t.W, t.p])
    for a, b in zip(gi, gu):
        assert close(a, b, 1e-9)


@pytest.mark.parametrize("batch", [None, 3])
def test_second_derivative_is_exact(batch: int | None) -> None:
    """Hessian-vector products equal those of the unrolled iteration."""
    t = Toy(batch=batch)
    dp = torch.randn(
        t.p.shape, generator=torch.Generator().manual_seed(1), **DD
    )

    def hvp(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        (g,) = torch.autograd.grad(loss(x), t.p, create_graph=True)
        return torch.autograd.grad((g * dp).sum(), [t.W, t.p])

    for a, b in zip(hvp(t.implicit()), hvp(t.unrolled())):
        assert close(a, b, 1e-8)


def test_dense_fallback_matches() -> None:
    """Forcing the dense fallback (no Anderson iterations) gives the same."""
    t = Toy()
    ref = torch.autograd.grad(loss(t.implicit()), [t.W, t.p])
    fb = torch.autograd.grad(
        loss(t.implicit(bck_options={"maxiter": 0})), [t.W, t.p]
    )
    for a, b in zip(ref, fb):
        assert close(a, b, 1e-9)


def test_batch_entries_converge_at_different_rates() -> None:
    """A stiff batch entry does not spoil an easy one."""
    t = Toy(batch=2)
    with torch.no_grad():
        t.W[1] *= 1.8  # closer to the edge of contractivity
    gi = torch.autograd.grad(loss(t.implicit()), [t.W, t.p])
    gu = torch.autograd.grad(loss(t.unrolled(2000)), [t.W, t.p])
    for a, b in zip(gi, gu):
        assert close(a, b, 1e-7)


def test_no_grad_bypasses_function() -> None:
    """Without grad mode, the plain solution is returned."""
    t = Toy()
    with torch.no_grad():
        x = t.implicit()
    assert x.grad_fn is None and not x.requires_grad


def test_no_parameter_requires_grad() -> None:
    """Nothing to differentiate: the plain solution is returned."""
    t = Toy()
    t.W.requires_grad_(False)
    t.p.requires_grad_(False)
    assert not t.implicit().requires_grad


def test_leaves_are_found() -> None:
    """All autograd leaves behind ``g`` are the inputs of the Function."""
    t = Toy()
    ids = {id(v) for v in _grad_leaves(t.g(t.x0))}
    assert ids == {id(t.W), id(t.p)}


def test_options_ignore_unknown_keys() -> None:
    """Keys that only xitorch knew (e.g. ``posdef``) are ignored."""
    o = AdjointOptions.from_mapping({"posdef": True, "maxiter": 7, "m": 3})
    assert (o.maxiter, o.history) == (7, 3)


def test_options_accept_xitorch_names() -> None:
    """xitorch's ``max_niter`` sets the iteration limit."""
    assert AdjointOptions.from_mapping({"max_niter": 11}).maxiter == 11


def test_options_warn_on_ignored_keys() -> None:
    """Options without effect are reported; ``posdef`` (always set) is not."""
    OutputHandler.clear_warnings()
    AdjointOptions.from_mapping({"posdef": True, "method": "gmres"})
    msgs = [msg for msg, _ in OutputHandler.warnings]
    assert any("'method'" in m for m in msgs)
    assert not any("'posdef'" in m for m in msgs)


@pytest.mark.parametrize("create_graph", [False, True])
def test_no_reference_cycle(create_graph: bool) -> None:
    """Output and graph are freed by reference counting alone (no gc)."""
    gc.collect()
    gc.disable()
    try:
        t = Toy()
        x = t.implicit()
        probe = weakref.ref(x)
        (g,) = torch.autograd.grad(loss(x), t.p, create_graph=create_graph)
        if create_graph:
            g.sum().backward()
        del x, g
        assert probe() is None
    finally:
        gc.enable()


def test_closure_reaches_no_downstream_state() -> None:
    """
    ``g`` is stored on the Function. Nothing it can reach may hold the output
    of the solve, or the Function would keep itself alive through the graph.
    """
    t = Toy()
    x = t.implicit()
    seen: set[int] = set()
    stack = [t.g]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        assert obj is not x
        stack.extend(gc.get_referents(obj))


def test_dense_fallback_second_order_and_batched() -> None:
    """The dense fallback also works in the double VJP and for batches."""
    t = Toy(batch=2)
    dp = torch.randn(
        t.p.shape, generator=torch.Generator().manual_seed(1), **DD
    )

    def hvp(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        (g,) = torch.autograd.grad(loss(x), t.p, create_graph=True)
        return torch.autograd.grad((g * dp).sum(), [t.W, t.p])

    ref = hvp(t.unrolled())
    OutputHandler.clear_warnings()
    fb = hvp(t.implicit(bck_options={"maxiter": 0}))
    assert any("dense" in msg for msg, _ in OutputHandler.warnings)
    for a, b in zip(fb, ref):
        assert close(a, b, 1e-8)


def test_small_upstream_gradient() -> None:
    """The adjoint tolerance scales with the gradient (no early stop)."""
    t = Toy()
    x = t.implicit()
    (g1,) = torch.autograd.grad(x, t.p, torch.ones_like(x), retain_graph=True)
    (g2,) = torch.autograd.grad(x, t.p, 1e-9 * torch.ones_like(x))
    assert close(g2 * 1e9, g1, 1e-8)


def test_float32() -> None:
    """Single precision runs through the same code path."""
    t = Toy()
    with torch.no_grad():
        t.W.data = t.W.data.float()
        t.p.data = t.p.data.float()
        t.x0 = t.x0.float()
    x, _ = equilibrium(t.g, t.x0, f_tol=1e-6, x_tol=1e-6, maxiter=200)
    (g,) = torch.autograd.grad(x.sum(), t.p)
    assert g.dtype == torch.float32 and torch.isfinite(g).all()


@pytest.mark.xfail(
    strict=True,
    reason="known limit: the implicit term is attached to autograd leaves, "
    "not to non-leaf intermediates derived from them",
)
def test_nonleaf_intermediate() -> None:
    """Derivative w.r.t. a non-leaf tensor that ``g`` closes over."""
    W = torch.tensor([[0.3, 0.1], [0.0, 0.2]], **DD)
    p0 = torch.tensor([0.5, -0.2], **DD).requires_grad_()
    p = p0 * 1.0  # non-leaf

    def g(x: torch.Tensor) -> torch.Tensor:
        return 0.4 * torch.tanh(W @ x) + p

    x, _ = equilibrium(g, torch.zeros(2, **DD), **FWD)
    (d_inter,) = torch.autograd.grad((x**2).sum(), p, retain_graph=True)
    (d_leaf,) = torch.autograd.grad((x**2).sum(), p0)
    assert close(d_inter, d_leaf, 1e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_entry_with_zero_gradient(dtype: torch.dtype) -> None:
    """
    A system of the batch that receives no gradient (e.g., one row of a
    batched Jacobian) neither breaks nor stalls the adjoint solve.
    """
    t = Toy(batch=2)
    with torch.no_grad():
        t.W.data = t.W.data.to(dtype)
        t.p.data = t.p.data.to(dtype)
        t.x0 = t.x0.to(dtype)
    fwd = (
        {**FWD, "f_tol": 1e-6, "x_tol": 1e-6} if dtype == torch.float32 else FWD
    )
    x, _ = equilibrium(t.g, t.x0, batched=True, **fwd)

    OutputHandler.clear_warnings()
    gi = torch.autograd.grad(loss(x[0]), [t.W, t.p])
    assert not any("dense" in msg for msg, _ in OutputHandler.warnings)

    gu = torch.autograd.grad(loss(t.unrolled()[0]), [t.W, t.p])
    rtol = 1e-4 if dtype == torch.float32 else 1e-8  # adjoint atol 1e-9
    for a, b in zip(gi, gu):
        assert (a[1] == 0).all()
        assert close(a, b, rtol)


def test_repeated_backward() -> None:
    """Backward through a retained graph gives the same gradient twice."""
    t = Toy()
    out = loss(t.implicit())
    g1 = torch.autograd.grad(out, [t.W, t.p], retain_graph=True)
    g2 = torch.autograd.grad(out, [t.W, t.p])
    for a, b in zip(g1, g2):
        assert close(a, b, 1e-12)


def test_iterations_count_forward_solve_only() -> None:
    """The iteration count excludes the evaluations needed for gradients."""
    t = Toy()
    calls = 0

    def g(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return t.g(x)

    with torch.no_grad():
        _, n_plain = equilibrium(g, t.x0, **FWD)
    assert n_plain == calls > 1

    calls = 0
    x, n_grad = equilibrium(g, t.x0, **FWD)
    torch.autograd.grad(loss(x), t.p)
    assert n_grad == n_plain < calls


def test_no_extra_evaluation_without_parameters() -> None:
    """Grad mode on, but nothing requires grad: only the forward iterations."""
    t = Toy()
    t.W.requires_grad_(False)
    t.p.requires_grad_(False)
    calls = 0

    def g(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return t.g(x)

    _, niter = equilibrium(g, t.x0, **FWD)
    assert calls == niter


def test_method_none_means_default() -> None:
    """``method=None`` selects the default solver (as in xitorch)."""
    t = Toy()
    x, _ = equilibrium(t.g, t.x0, **{**FWD, "method": None})
    assert close(x, t.implicit(), 1e-12)


def test_default_tolerance_follows_forward_tolerance() -> None:
    """The adjoint tolerance defaults to 1% of the forward ``f_tol``."""
    tol = AdjointOptions(f_tol=1e-6).tolerances
    assert tol(torch.double) == (1e-8, 0.0)
    assert tol(torch.float32) == (100 * torch.finfo(torch.float32).eps, 0.0)
    assert AdjointOptions(f_tol=1e-14).tolerances(torch.double) == (1e-10, 0.0)

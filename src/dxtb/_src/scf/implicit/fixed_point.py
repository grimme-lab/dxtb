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
SCF Implicit: Fixed-Point Solver
================================

Implicit differentiation of a fixed point :math:`x^* = g(x^*, p)` with exact
second derivatives and without the memory leak of ``xitorch``'s
``RootFinder``.

The forward solve runs without autograd. The backward pass applies the implicit
function theorem: for an incoming gradient :math:`v` it solves the adjoint
system :math:`(I - J^T) \\lambda = v` with :math:`J = \\partial g/\\partial x`
at :math:`x^*` and returns :math:`\\lambda^T \\partial g/\\partial p`. The
adjoint solve is itself a ``torch.autograd.Function``
(:class:`_AdjointSolve`), whose backward yields the derivatives with respect
to :math:`x^*` and :math:`p` (double VJP). Together with saving the output of
the first Function, this makes the result differentiable a second time
(Hessians) without approximation.

Design rules (see also the closure test in ``test_fixed_point.py``):

- The map ``g`` must be *stateless*: it may not read or write any object that
  later holds the output of the solve (e.g., the SCF data). Everything that can
  carry gradients is reached through tensors that ``g`` closes over; their
  autograd leaves are discovered from the graph of one evaluation of ``g`` and
  become the inputs of the Function. Derivatives with respect to autograd
  leaves (positions, fields, parameters) are therefore complete. Known limit:
  the *implicit* part is attached to leaves, not to non-leaf intermediates
  derived from them (see ``test_fixed_point.py``).
- The Function keeps ``g`` on ``ctx`` (needed in the backward). This is safe
  only under the previous rule: ``ctx`` is part of the graph of the output, and
  a path from ``g`` back to the output would be a reference cycle through the
  C++ graph, which the garbage collector cannot free.

The structure follows JAX's ``custom_root``/``custom_linear_solve``, ``jaxopt``
(``custom_fixed_point``) and ``torchopt`` (``diff.implicit``), all Apache-2.0.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from dxtb._src.exlibs.xitorch._utils.misc import get_method
from dxtb._src.exlibs.xitorch.optimize.rootfinder import _RF_METHODS
from dxtb import OutputHandler
from dxtb._src.typing import Any, Callable, Mapping, Tensor

__all__ = ["AdjointOptions", "equilibrium"]



@dataclass
class AdjointOptions:
    """Options of the adjoint (linear) solve."""

    maxiter: int = 200
    """Maximum number of Anderson iterations before the dense fallback."""

    atol: float | None = None
    """
    Tolerance on the residual, relative to the largest entry of the right-hand
    side (the system is linear in it). Defaults to ``1e-9`` for double and
    ``1e-4`` for lower precision. Converged if
    ``max|res| <= (atol + rtol) * max|rhs|`` for every system.
    """

    rtol: float | None = None
    """Second relative tolerance, added to ``atol``. Defaults to ``atol``."""

    history: int = 10
    """Number of previous iterates used in the Anderson mixing."""

    batched: bool = False
    """Whether the first dimension enumerates independent systems."""

    @classmethod
    def from_mapping(
        cls, opts: Mapping[str, Any] | None, batched: bool = False
    ) -> "AdjointOptions":
        """Create from a dict, ignoring unknown keys (e.g., ``posdef``)."""
        opts = dict(opts or {})
        return cls(
            maxiter=int(opts.get("maxiter", cls.maxiter)),
            atol=opts.get("atol"),
            rtol=opts.get("rtol"),
            history=int(opts.get("m", cls.history)),
            batched=batched,
        )

    def tolerances(self, dtype: torch.dtype) -> tuple[float, float]:
        """Absolute and relative tolerance for a given dtype."""
        eps = torch.finfo(dtype).eps
        atol = self.atol if self.atol is not None else (1e-9 if eps < 1e-10 else 1e-4)
        rtol = self.rtol if self.rtol is not None else atol
        return float(atol), float(rtol)


class _AdjointConvergenceError(RuntimeError):
    """Adjoint iterations did not converge (triggers the dense fallback)."""


##############################################################################
# linear solves


def _anderson(
    fcn: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    maxiter: int,
    atol: float,
    rtol: float,
    scale: Tensor | None = None,
    m: int = 5,
    beta: float = 1.0,
    reg: float = 1e-12,
) -> Tensor:
    """
    Anderson acceleration for ``x = fcn(x)`` on tensors of shape ``(B, n)``.

    The rows are independent systems and are checked for convergence
    separately (all must converge). Raises ``_AdjointConvergenceError`` if not
    converged within ``maxiter`` iterations.
    """
    xs: list[Tensor] = []
    gs: list[Tensor] = []
    x = x0
    for _ in range(maxiter):
        g = fcn(x)
        res = g - x

        # tolerances relative to the size of the right-hand side (the system
        # is linear in it), so that small gradients are not "converged" early
        ref = g.abs().amax(-1) if scale is None else scale
        if bool((res.abs().amax(-1) <= (atol + rtol) * ref).all()):
            return g

        xs.append(x)
        gs.append(g)
        xs, gs = xs[-m:], gs[-m:]

        f = torch.stack([gi - xi for gi, xi in zip(gs, xs)], dim=-2)  # B,k,n
        eye = torch.eye(f.shape[-2], dtype=f.dtype, device=f.device)
        hmat = f @ f.transpose(-1, -2)
        # scale-invariant regularization (residuals may be tiny for small rhs)
        dscale = hmat.diagonal(dim1=-2, dim2=-1).mean(-1)
        dscale = dscale.clamp_min(torch.finfo(f.dtype).tiny)
        hmat = hmat + reg * dscale[:, None, None] * eye
        y = torch.linalg.solve(hmat, torch.ones_like(f[..., 0]).unsqueeze(-1))
        alpha = (y / y.sum(-2, keepdim=True)).transpose(-1, -2)  # B,1,k

        gmat = torch.stack(gs, dim=-2)
        xmat = torch.stack(xs, dim=-2)
        x = (alpha @ (beta * gmat + (1.0 - beta) * xmat)).squeeze(-2)

    raise _AdjointConvergenceError("Adjoint iterations did not converge.")


def _dense(op: Callable[[Tensor], Tensor], rhs: Tensor) -> Tensor:
    """
    Solve ``(I - A) u = rhs`` for a linear map ``A`` by building it column by
    column. Fallback and reference for small systems.
    """
    n = rhs.numel()
    eye = torch.eye(n, dtype=rhs.dtype, device=rhs.device)
    cols = [op(eye[i].reshape(rhs.shape)).reshape(-1) for i in range(n)]
    amat = torch.stack(cols, dim=-1)
    return torch.linalg.solve(eye - amat, rhs.reshape(-1)).reshape(rhs.shape)


def _solve(
    op: Callable[[Tensor], Tensor], rhs: Tensor, opts: AdjointOptions
) -> Tensor:
    """
    Solve ``u = rhs + op(u)`` (``op`` linear) with Anderson acceleration and
    dense fallback.
    """
    shape = rhs.shape
    nb = shape[0] if opts.batched else 1
    atol, rtol = opts.tolerances(rhs.dtype)
    scale = rhs.reshape(nb, -1).abs().amax(-1)

    def step(u: Tensor) -> Tensor:
        return (rhs + op(u.reshape(shape))).reshape(nb, -1)

    try:
        u = _anderson(
            step,
            rhs.reshape(nb, -1),
            maxiter=opts.maxiter,
            atol=atol,
            rtol=rtol,
            scale=scale,
            m=opts.history,
        )
        return u.reshape(shape)
    except _AdjointConvergenceError:
        OutputHandler.warn(
            "Adjoint iterations of the implicit SCF gradient did not converge "
            f"within {opts.maxiter} iterations. Falling back to a dense solve."
        )
        return _dense(op, rhs)


##############################################################################
# functions


def _vjp_setup(
    fcn: Callable[[Tensor], Tensor], x: Tensor
) -> tuple[Tensor, Tensor, Callable[[Tensor], Tensor]]:
    """Evaluate ``g`` at a detached copy of ``x`` and return ``(z, f, J^T .)``."""
    z = x.detach().requires_grad_(True)
    f = fcn(z)

    def jtu(u: Tensor) -> Tensor:
        return torch.autograd.grad(f, z, u, retain_graph=True)[0]

    return z, f, jtu


class _AdjointSolve(torch.autograd.Function):
    """Solve ``(I - J^T) lam = v``; differentiable w.r.t. ``v``, ``x``, ``p``."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        v: Tensor,
        x: Tensor,
        fcn: Callable[[Tensor], Tensor],
        opts: AdjointOptions,
        *params: Tensor,
    ) -> Tensor:
        with torch.enable_grad():
            _, _, jtu = _vjp_setup(fcn, x)
            lam = _solve(jtu, v.detach(), opts).detach()

        ctx.fcn = fcn
        ctx.opts = opts
        ctx.save_for_backward(lam, x, *params)
        return lam

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_lam: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore[override]
        lam, x, *params = ctx.saved_tensors
        fcn, opts = ctx.fcn, ctx.opts
        grad_lam = grad_lam.detach()

        with torch.enable_grad():
            z = x.detach().requires_grad_(True)
            f = fcn(z)

            # mu solves (I - J) mu = grad_lam, with J mu from a double VJP
            w = torch.zeros_like(f, requires_grad=True)
            jtw = torch.autograd.grad(f, z, w, create_graph=True)[0]

            def ju(u: Tensor) -> Tensor:
                return torch.autograd.grad(jtw, w, u, retain_graph=True)[0]

            mu = _solve(ju, grad_lam, opts).detach()

            # d/d(x, p) of  lam^T J(x, p) mu  (lam, mu fixed)
            jtl = torch.autograd.grad(f, z, lam.detach(), create_graph=True)[0]
            s = (jtl * mu).sum()
            grads = torch.autograd.grad(
                s, [z, *params], retain_graph=True, allow_unused=True
            )

        return (mu, grads[0], None, None, *grads[1:])


_LOCAL = threading.local()
"""
Thread-local record of the node whose backward is currently evaluating the
*partial* derivative of ``g`` with respect to the parameters at fixed ``x*``.

In the differentiable case, ``g`` is evaluated at the (graph-connected)
output, so ``autograd.grad(f, params)`` would also walk through the output's
node, i.e., re-enter the very backward that is running and add the total
instead of the partial derivative. Only that one node is skipped (identified by
its ``ctx``); other implicit solves in the graph behave normally. The graph of
the result still contains the dependence on ``x*``, which the second derivative
needs.
"""


class _ImplicitFixedPoint(torch.autograd.Function):
    """Identity in the forward; implicit function theorem in the backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x_star: Tensor,
        fcn: Callable[[Tensor], Tensor],
        opts: AdjointOptions,
        first: tuple[Tensor, Tensor],
        *params: Tensor,
    ) -> Tensor:
        out = x_star.clone()
        ctx.fcn = fcn
        ctx.opts = opts
        # Evaluation of `g` at `x*` from the parameter discovery. Its graph
        # ends at a detached leaf and the parameters, i.e., it holds nothing
        # downstream of the output, and it saves one evaluation in the
        # backward without `create_graph`.
        ctx.first = first
        # saving the output (not as plain attribute) creates no reference cycle
        ctx.save_for_backward(out, *params)
        return out

    @staticmethod
    def backward(ctx: Any, v: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore[override]
        if getattr(_LOCAL, "running", None) is ctx:
            return (None,) * (4 + len(ctx.saved_tensors) - 1)

        x, *params = ctx.saved_tensors
        fcn, opts = ctx.fcn, ctx.opts

        # `create_graph=True` of the outer call is signaled by grad mode
        cg = torch.is_grad_enabled()

        with torch.enable_grad():
            # Only in the differentiable case, `x` carries its dependence on
            # `params` (through this Function), which yields the exact second
            # derivative. Otherwise, it must not re-enter this node.
            if cg:
                z = x
                f = fcn(z)
            else:
                z, f = ctx.first
            if cg:
                lam = _AdjointSolve.apply(v, z, fcn, opts, *params)
            else:  # no graph needed: reuse this evaluation of `g`
                lam = _solve(
                    lambda u: torch.autograd.grad(f, z, u, retain_graph=True)[
                        0
                    ],
                    v.detach(),
                    opts,
                )
            previous = getattr(_LOCAL, "running", None)
            _LOCAL.running = ctx
            try:
                grads = torch.autograd.grad(
                    f,
                    params,
                    lam,
                    retain_graph=True,
                    create_graph=cg,
                    allow_unused=True,
                )
            finally:
                _LOCAL.running = previous

        return (None, None, None, None, *grads)


def _grad_leaves(t: Tensor) -> list[Tensor]:
    """All autograd leaves that require grad and are reachable from ``t``."""
    leaves: dict[int, Tensor] = {}
    seen: set[int] = set()
    stack = [t.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))

        var = getattr(node, "variable", None)  # AccumulateGrad
        if var is not None:
            leaves[id(var)] = var
            continue
        stack.extend(nxt for nxt, _ in node.next_functions)
    return list(leaves.values())


def equilibrium(
    fcn: Callable[[Tensor], Tensor],
    y0: Tensor,
    bck_options: Mapping[str, Any] | None = None,
    on_converged: Callable[[], None] | None = None,
    batched: bool = False,
    **fwd_options: Any,
) -> Tensor:
    """
    Solve the fixed-point equation ``y = fcn(y)`` with implicit gradients.

    Parameters
    ----------
    fcn : Callable[[Tensor], Tensor]
        Stateless fixed-point map (see module docstring). All differentiable
        inputs are captured by ``fcn`` as tensors.
    y0 : Tensor
        Initial guess (treated as constant).
    bck_options : Mapping[str, Any] | None, optional
        Options of the adjoint solve, see :meth:`AdjointOptions.from_mapping`
        (keys ``maxiter``, ``atol``, ``rtol``, ``m``).
    on_converged : Callable[[], None] | None, optional
        Called right after the forward solve, before any additional evaluation
        of ``fcn`` (e.g., to snapshot an iteration counter).
    batched : bool
        Whether the first dimension enumerates independent systems (the
        adjoint convergence is then checked per system).
    **fwd_options : Any
        Options of the forward solver (``method`` and the options of the
        vendored xitorch root solvers, i.e., the same solver and iteration
        count as before).

    Returns
    -------
    Tensor
        Solution ``y*``. If gradients are required, it is connected to the
        graph of all parameters of ``fcn`` and differentiable twice.
    """
    fwd = dict(fwd_options)
    solver = get_method("rootfinder", _RF_METHODS, fwd.pop("method", "broyden1"))

    # root of `y - g(y)` (same function and solver as xitorch's equilibrium)
    def root(y: Tensor) -> Tensor:
        return y - fcn(y)

    with torch.no_grad():
        x_star = solver(root, y0.detach(), (), **fwd)
    x_star = x_star.detach()

    if on_converged is not None:
        on_converged()

    if not torch.is_grad_enabled():
        return x_star

    # discover all parameters (autograd leaves) that `fcn` depends on
    z = x_star.clone().requires_grad_(True)
    f = fcn(z)
    params = [p for p in _grad_leaves(f) if p is not z]
    if not params:
        return x_star

    opts = AdjointOptions.from_mapping(bck_options, batched=batched)
    return _ImplicitFixedPoint.apply(x_star, fcn, opts, (z, f), *params)

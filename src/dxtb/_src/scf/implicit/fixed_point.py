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
  leaves (positions, fields, parameters) are therefore complete.
- The Function keeps ``g`` on ``ctx`` (needed in the backward). This is safe
  only under the previous rule: ``ctx`` is part of the graph of the output, and
  a path from ``g`` back to the output would be a reference cycle through the
  C++ graph, which the garbage collector cannot free.

Known limits:

- The *implicit* part is attached to leaves, not to non-leaf intermediates
  (e.g., ``pos0 + d`` or ``result.integrals.hcore``); their derivatives miss
  it silently (see ``test_fixed_point.py``).
- Third and higher derivatives are not supported.

The structure follows JAX's ``custom_root``/``custom_linear_solve``, ``jaxopt``
(``custom_fixed_point``) and ``torchopt`` (``diff.implicit``), all Apache-2.0.
"""

from __future__ import annotations

import gc
import types
from dataclasses import dataclass

import torch
from torch.autograd.function import once_differentiable

from dxtb import OutputHandler
from dxtb._src.exlibs.xitorch._utils.misc import get_method
from dxtb._src.exlibs.xitorch.optimize.rootfinder import _RF_METHODS
from dxtb._src.typing import Any, Callable, Mapping, Tensor

__all__ = ["AdjointOptions", "equilibrium"]


_SILENT_KEYS = frozenset({"posdef"})
"""Options of xitorch's backward that the SCF always sets; ignored silently."""


@dataclass
class AdjointOptions:
    """Options of the adjoint (linear) solve."""

    maxiter: int = 200
    """Maximum number of Anderson iterations before the dense fallback."""

    atol: float | None = None
    """
    Tolerance on the residual, relative to the largest entry of the right-hand
    side (the system is linear in it). Converged if
    ``max|res| <= (atol + rtol) * max|rhs|`` for every system. Defaults to
    ``max(1e-10, 1e-2 * f_tol, 100 * eps)``: the gradient cannot be more
    accurate than the forward solve, nor than the precision in use.
    """

    rtol: float | None = None
    """Second tolerance, added to ``atol``. Defaults to ``0``."""

    f_tol: float | None = None
    """Tolerance of the forward solve (default ``1e-7``), sets ``atol``."""

    history: int = 10
    """Number of previous iterates used in the Anderson mixing."""

    batched: bool = False
    """Whether the first dimension enumerates independent systems."""

    @classmethod
    def from_mapping(
        cls, opts: Mapping[str, Any] | None, batched: bool = False
    ) -> AdjointOptions:
        """
        Create from a dict of options.

        Recognized keys are ``maxiter`` (or xitorch's ``max_niter``),
        ``atol``, ``rtol`` and ``m`` (history). Other keys have no effect and
        are reported with a warning, except ``posdef``, which the SCF always
        sets for xitorch's former backward solve.
        """
        opts = dict(opts or {})
        known = {"maxiter", "max_niter", "atol", "rtol", "m"}
        ignored = sorted(set(opts) - known - _SILENT_KEYS)
        if ignored:
            names = ", ".join(repr(k) for k in ignored)
            OutputHandler.warn(
                f"Options {names} of the implicit SCF backward have no effect "
                "(recognized: 'maxiter', 'atol', 'rtol', 'm')."
            )

        maxiter = opts.get("maxiter", opts.get("max_niter", cls.maxiter))
        return cls(
            maxiter=int(maxiter),
            atol=opts.get("atol"),
            rtol=opts.get("rtol"),
            history=int(opts.get("m", cls.history)),
            batched=batched,
        )

    def tolerances(self, dtype: torch.dtype) -> tuple[float, float]:
        """Tolerances ``(atol, rtol)`` for a given dtype."""
        if self.atol is not None:
            atol = self.atol
        else:
            f_tol = 1e-7 if self.f_tol is None else self.f_tol
            atol = max(1e-10, 1e-2 * f_tol, 100 * torch.finfo(dtype).eps)
        rtol = 0.0 if self.rtol is None else self.rtol
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
    separately. A converged row is frozen and leaves the mixing, so that its
    (possibly vanishing) residuals cannot spoil the other rows. Raises
    ``_AdjointConvergenceError`` if not all rows converge within ``maxiter``
    iterations or the mixing breaks down.
    """
    tiny = torch.finfo(x0.dtype).tiny
    reg = max(reg, torch.finfo(x0.dtype).eps)

    xs: list[Tensor] = []
    gs: list[Tensor] = []
    x = x0
    out = x0
    active = torch.ones(x0.shape[0], dtype=torch.bool, device=x0.device)
    for _ in range(maxiter):
        g = fcn(x)
        res = g - x

        # tolerances relative to the size of the right-hand side (the system
        # is linear in it), so that small gradients are not "converged" early
        ref = g.abs().amax(-1) if scale is None else scale
        conv = res.abs().amax(-1) <= (atol + rtol) * ref
        out = torch.where((conv & active).unsqueeze(-1), g, out)
        active = active & ~conv
        if not bool(active.any()):
            return out

        xs.append(x)
        gs.append(g)
        xs, gs = xs[-m:], gs[-m:]

        # mixing coefficients of the active rows only
        idx = active.nonzero().squeeze(-1)
        gmat = torch.stack(gs, dim=-2)[idx]  # B',k,n
        xmat = torch.stack(xs, dim=-2)[idx]
        f = gmat - xmat

        # The coefficients are invariant to the scale of the residuals of a
        # row, which are normalized to avoid underflow for tiny right-hand
        # sides. The regularization is relative to the diagonal.
        f = f / f.abs().amax((-2, -1), keepdim=True).clamp_min(tiny)
        hmat = f @ f.transpose(-1, -2)
        eye = torch.eye(f.shape[-2], dtype=f.dtype, device=f.device)
        dscale = hmat.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(tiny)
        hmat = hmat + reg * dscale[:, None, None] * eye
        rhs = torch.ones_like(f[..., 0]).unsqueeze(-1)
        try:
            y = torch.linalg.solve(hmat, rhs)
        except torch.linalg.LinAlgError as e:
            raise _AdjointConvergenceError("Anderson mixing failed.") from e
        alpha = (y / y.sum(-2, keepdim=True)).transpose(-1, -2)  # B',1,k
        if not bool(torch.isfinite(alpha).all()):
            raise _AdjointConvergenceError("Anderson mixing failed.")

        mixed = (alpha @ (beta * gmat + (1.0 - beta) * xmat)).squeeze(-2)

        # converged rows keep their solution as input (their output is unused)
        x = out.index_put((idx,), mixed)

    raise _AdjointConvergenceError("Adjoint iterations did not converge.")


def _dense(op: Callable[[Tensor], Tensor], rhs: Tensor) -> Tensor:
    """
    Solve ``(I - A) u = rhs`` for a linear map ``A`` by building it column by
    column. Fallback and reference for small systems.

    Limit: one matrix for the whole batch (``rhs.numel()`` VJPs, memory
    ``numel^2``), which can exhaust memory for large or Fock-mode systems.
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


def _evaluate(
    fcn: Callable[[Tensor], Tensor], x: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate ``g`` at a detached copy ``z`` of ``x``; returns ``(z, g(z))``."""
    z = x.detach().requires_grad_(True)
    return z, fcn(z)


def _jtu(f: Tensor, z: Tensor) -> Callable[[Tensor], Tensor]:
    """The vector-Jacobian product ``u -> J^T u`` of ``f = g(z)``."""

    def jtu(u: Tensor) -> Tensor:
        return torch.autograd.grad(f, z, u, retain_graph=True)[0]

    return jtu


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
            z, f = _evaluate(fcn, x)
            lam = _solve(_jtu(f, z), v.detach(), opts).detach()

        ctx.fcn = fcn
        ctx.opts = opts
        ctx.save_for_backward(lam, x, *params)
        return lam

    @staticmethod
    @once_differentiable
    def backward(  # type: ignore[override]
        ctx: Any, grad_lam: Tensor
    ) -> tuple[Tensor | None, ...]:
        lam, x, *params = ctx.saved_tensors
        fcn, opts = ctx.fcn, ctx.opts
        grad_lam = grad_lam.detach()

        with torch.enable_grad():
            z, f = _evaluate(fcn, x)

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
        # downstream of the output, and it saves one evaluation in the first
        # backward without `create_graph`. It is released by the first
        # backward (a repeated backward evaluates `g` again).
        ctx.first = first
        # saving the output (not as plain attribute) creates no reference cycle
        ctx.save_for_backward(out, *params)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, v: Tensor
    ) -> tuple[Tensor | None, ...]:
        # Re-entry while this node computes the *partial* derivative of `g`
        # w.r.t. the parameters: in the differentiable case `g` is evaluated
        # at the (graph-connected) output, so `autograd.grad(f, params)` walks
        # through this node again and would add the total derivative. The
        # flag is on `ctx` (not thread-local): autograd may run the nested
        # call on another thread.
        if getattr(ctx, "running", False):
            return (None,) * (4 + len(ctx.saved_tensors) - 1)

        x, *params = ctx.saved_tensors
        fcn, opts = ctx.fcn, ctx.opts
        first, ctx.first = ctx.first, None

        # `create_graph=True` of the outer call is signaled by grad mode
        cg = torch.is_grad_enabled()

        with torch.enable_grad():
            if cg:
                # `x` carries its dependence on `params` (through this
                # Function), which yields the exact second derivative
                z, f = x, fcn(x)
                lam = _AdjointSolve.apply(v, z, fcn, opts, *params)
            else:
                # no graph needed: `x` must not re-enter this node, and the
                # evaluation from the forward is reused if still available
                z, f = first if first is not None else _evaluate(fcn, x)
                lam = _solve(_jtu(f, z), v.detach(), opts)

            # Limit: all discovered leaves are differentiated, also those the
            # current call does not ask for (no public autograd API tells).
            ctx.running = True
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
                ctx.running = False

        return (None, None, None, None, *grads)


def _grad_leaves(t: Tensor) -> list[Tensor]:
    """All autograd leaves that require grad and are reachable from ``t``."""
    leaves: dict[int, Tensor] = {}
    # Hold the nodes (not only their ids): older torch versions create a new
    # Python wrapper on every access, and the id of a freed one is reused.
    seen: dict[int, object] = {}
    stack = [t.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen[id(node)] = node

        var = getattr(node, "variable", None)  # AccumulateGrad
        if var is not None:
            leaves[id(var)] = var
            continue
        stack.extend(nxt for nxt, _ in node.next_functions)
    return list(leaves.values())


def _reaches_grad(fcn: Callable[[Tensor], Tensor]) -> bool:
    """
    Whether a tensor that requires grad is reachable from the closure of
    ``fcn`` (module globals are not followed; see the design rules).
    """
    seen: set[int] = set()
    stack: list[object] = [fcn]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (types.ModuleType, type)):
            continue
        seen.add(id(obj))
        if isinstance(obj, Tensor):
            if obj.requires_grad:
                return True
            continue
        if isinstance(obj, types.FunctionType):
            stack.extend(c.cell_contents for c in obj.__closure__ or ())
            continue
        stack.extend(gc.get_referents(obj))
    return False


def equilibrium(
    fcn: Callable[[Tensor], Tensor],
    y0: Tensor,
    bck_options: Mapping[str, Any] | None = None,
    batched: bool = False,
    **fwd_options: Any,
) -> tuple[Tensor, int]:
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
    batched : bool
        Whether the first dimension enumerates independent systems (the
        adjoint convergence is then checked per system).
    **fwd_options : Any
        Options of the forward solver (``method`` and the options of the
        vendored xitorch root solvers, i.e., the same solver and iteration
        count as before).

    Returns
    -------
    tuple[Tensor, int]
        Solution ``y*`` and the number of evaluations of ``fcn`` in the
        forward solve (the SCF iterations). If gradients are required, ``y*``
        is connected to the graph of all parameters of ``fcn`` and
        differentiable twice.
    """
    fwd = dict(fwd_options)
    method = fwd.pop("method", None) or "broyden1"
    solver = get_method("rootfinder", _RF_METHODS, method)

    # root of `y - g(y)` (same function and solver as xitorch's equilibrium)
    niter = 0

    def root(y: Tensor) -> Tensor:
        nonlocal niter
        niter += 1
        return y - fcn(y)

    with torch.no_grad():
        x_star = solver(root, y0.detach(), (), **fwd)
    x_star = x_star.detach()

    # skip the evaluation of `fcn` below if nothing can carry gradients
    if not torch.is_grad_enabled() or not _reaches_grad(fcn):
        return x_star, niter

    # discover all parameters (autograd leaves) that `fcn` depends on
    z = x_star.clone().requires_grad_(True)
    f = fcn(z)
    params = [p for p in _grad_leaves(f) if p is not z]
    if not params:
        return x_star, niter

    opts = AdjointOptions.from_mapping(bck_options, batched=batched)
    opts.f_tol = fwd.get("f_tol")
    out = _ImplicitFixedPoint.apply(x_star, fcn, opts, (z, f), *params)
    return out, niter

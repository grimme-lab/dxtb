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

Leak-free replacement for ``xitorch.optimize.equilibrium`` in the non-pure
implicit SCF.

The forward solve is run detached (no autograd graph). The implicit function
theorem is applied through a gradient hook on the output: for an upstream
gradient ``v``, the adjoint system ``(I - J^T) u = v`` is solved with
Anderson acceleration (dense fallback) and ``u`` replaces ``v`` on the way back
into a single differentiable evaluation of the fixed-point map.

The hook closes over exactly two tensors (``f_z`` and ``z``) and plain
scalars. It must never capture the fixed-point function, the SCF object or the
output, as this recreates the reference cycle through PyTorch's C++ graph
that causes the memory leak of ``xitorch``'s custom autograd function.

The adjoint/fixed-point structure follows ``jaxopt`` (``custom_fixed_point``,
Apache-2.0) and ``torchopt`` (``diff.implicit``, Apache-2.0).
"""

from __future__ import annotations

import torch

from dxtb._src.exlibs.xitorch import optimize as xto
from dxtb._src.typing import Any, Callable, Mapping, Tensor

__all__ = ["equilibrium"]


class _AdjointConvergenceError(RuntimeError):
    """Adjoint iterations did not converge (triggers the dense fallback)."""


def _anderson(
    fcn: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    maxiter: int,
    tol: float,
    m: int = 5,
    beta: float = 1.0,
    reg: float = 1e-12,
) -> Tensor:
    """
    Anderson acceleration for ``x = fcn(x)`` on tensors of shape ``(B, n)``.

    All operations are differentiable. Raises ``_AdjointConvergenceError`` if
    not converged within ``maxiter`` iterations.
    """
    xs: list[Tensor] = []
    gs: list[Tensor] = []
    x = x0
    for _ in range(maxiter):
        g = fcn(x)
        res = g - x

        if bool((res.abs().amax(-1) <= tol * (1.0 + g.abs().amax(-1))).all()):
            return g

        xs.append(x)
        gs.append(g)
        xs, gs = xs[-m:], gs[-m:]

        f = torch.stack([gi - xi for gi, xi in zip(gs, xs)], dim=-2)  # B,k,n
        eye = torch.eye(f.shape[-2], dtype=f.dtype, device=f.device)
        hmat = f @ f.transpose(-1, -2)
        hmat = hmat + reg * (1.0 + hmat.diagonal(dim1=-2, dim2=-1).mean()) * eye
        y = torch.linalg.solve(hmat, torch.ones_like(f[..., 0]).unsqueeze(-1))
        alpha = (y / y.sum(-2, keepdim=True)).transpose(-1, -2)  # B,1,k

        gmat = torch.stack(gs, dim=-2)
        xmat = torch.stack(xs, dim=-2)
        x = (alpha @ (beta * gmat + (1.0 - beta) * xmat)).squeeze(-2)

    raise _AdjointConvergenceError("Adjoint iterations did not converge.")


def _dense_adjoint(
    jtu: Callable[[Tensor], Tensor], v: Tensor, create_graph: bool
) -> Tensor:
    """
    Solve ``(I - J^T) u = v`` by building ``J^T`` column by column.

    Only meant as a fallback and reference for small systems.
    """
    n = v.numel()
    eye = torch.eye(n, dtype=v.dtype, device=v.device)
    cols = [jtu(eye[i].reshape(v.shape)).reshape(-1) for i in range(n)]
    jt = torch.stack(cols, dim=-1)
    u = torch.linalg.solve(eye - jt, v.reshape(-1))
    return u.reshape(v.shape) if create_graph else u.detach().reshape(v.shape)


def equilibrium(
    fcn: Callable[[Tensor], Tensor],
    y0: Tensor,
    bck_options: Mapping[str, Any] | None = None,
    on_converged: Callable[[], None] | None = None,
    **fwd_options: Any,
) -> Tensor:
    """
    Solve the fixed-point equation ``y = fcn(y)`` with implicit gradients.

    Drop-in for the subset of ``xitorch.optimize.equilibrium`` that is used by
    the non-pure implicit SCF.

    Parameters
    ----------
    fcn : Callable[[Tensor], Tensor]
        Fixed-point map. All differentiable inputs are captured by ``fcn``.
    y0 : Tensor
        Initial guess (treated as constant).
    bck_options : Mapping[str, Any] | None, optional
        Options of the adjoint solve. Recognized keys are ``maxiter``
        (default: 200), ``atol`` (default: 1e-10 for double precision, 1e-4
        otherwise) and ``m`` (Anderson history, default: 5). All others
        (e.g., ``posdef`` for xitorch) are ignored.
    on_converged : Callable[[], None] | None, optional
        Called right after the forward solve, before the additional
        evaluations of ``fcn`` that are required to attach the gradient. Can
        be used, e.g., to snapshot iteration counters.
    **fwd_options : Any
        Options of the forward solver (passed to xitorch's ``equilibrium``,
        i.e., the same solver and iteration count as before).

    Returns
    -------
    Tensor
        Solution ``y*`` of the fixed-point equation. If gradients are
        required, it is connected to the graph of all parameters of ``fcn``
        (through one evaluation ``fcn(y*)``) and carries the
        implicit-differentiation hook.
    """
    opts = dict(bck_options or {})

    with torch.no_grad():
        x_star = xto.equilibrium(fcn=fcn, y0=y0.detach(), **fwd_options)
    x_star = x_star.detach()

    if on_converged is not None:
        on_converged()

    if not torch.is_grad_enabled():
        return x_star

    # NOTE: `f_z` must be evaluated first. The last evaluation of `fcn` defines
    # to which graph state-holding objects (`_Data`) are connected afterwards.
    z = x_star.clone().requires_grad_(True)
    f_z = fcn(z)

    x_new = fcn(x_star)
    if not x_new.requires_grad:
        return x_star

    # Value is exactly `x_star` (as before), gradient flows through `x_new`
    x_out = x_star + (x_new - x_new.detach())

    eps = torch.finfo(x_star.dtype).eps
    atol = float(opts.get("atol", 1e-10 if eps < 1e-10 else 1e-4))
    maxiter = int(opts.get("maxiter", 200))
    hist = int(opts.get("m", 5))

    def _hook(
        v: Tensor,
        f_z: Tensor = f_z,
        z: Tensor = z,
        atol: float = atol,
        maxiter: int = maxiter,
        hist: int = hist,
    ) -> Tensor:
        cg = torch.is_grad_enabled()  # True for create_graph=True

        def jtu(u: Tensor) -> Tensor:
            return torch.autograd.grad(
                f_z, z, u, retain_graph=True, create_graph=cg
            )[0]

        # J couples all entries (batch dimension included), so the whole
        # tensor is treated as one system
        shape = v.shape

        def step(u: Tensor) -> Tensor:
            return (v + jtu(u.reshape(shape))).reshape(1, -1)

        try:
            u = _anderson(
                step,
                v.reshape(1, -1),
                maxiter=maxiter,
                tol=atol,
                m=hist,
            ).reshape(shape)
        except _AdjointConvergenceError:
            u = _dense_adjoint(jtu, v, cg)
        return u

    x_out.register_hook(_hook)
    return x_out

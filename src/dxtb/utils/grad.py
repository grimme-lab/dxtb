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
Utility: Autograd
=================

This module contains utility functions for automatic differentiation.

Important! Before PyTorch 2.0.0, `functorch` does not work together with custom
autograd functions, which we definitely require. Additionally, `functorch`
imposes the implementation of a `forward` **and** `setup_context` method, i.e.,
the traditional way of using `forward` with the `ctx` argument does not work
"""

from __future__ import annotations

import torch
from tad_mctc.autograd import jac, jacrev

from dxtb.typing import Any, Callable, Tensor

from ..__version__ import __tversion__

__all__ = ["_hessian", "hessian"]


def hessian(
    f: Callable[..., Tensor],
    inputs: tuple[Any, ...],
    argnums: int = 0,
    create_graph: bool | None = None,
    retain_graph: bool = True,
    is_batched: bool = False,
) -> Tensor:
    """
    Wrapper for Hessian. The Hessian is the Jacobian of the gradient.

    PyTorch, however, suggests calculating the Jacobian of the Jacobian, which
    does not yield the correct shape in this case.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    inputs : tuple[Any, ...]
        The input parameters of `f`.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Tensor
        The Hessian.
    """
    if is_batched:
        raise NotImplementedError("Batched Hessian not available.")

    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    assert create_graph is not None

    def _grad(*inps: tuple[Any, ...]) -> Tensor:
        e = f(*inps).sum()

        if not isinstance(inps[argnums], Tensor):
            raise RuntimeError(f"The {argnums}'th input parameter must be a tensor.")

        # catch missing gradients (e.g., halogen bond correction evaluates to
        # zero if no donors/acceptors are present)
        if e.grad_fn is None:
            return torch.zeros_like(inps[argnums])  # type: ignore

        (g,) = torch.autograd.grad(
            e,
            inps[argnums],
            create_graph=create_graph,
            retain_graph=retain_graph,
        )
        return g

    grad = _grad(*inputs)
    hess = jac(
        grad.flatten(),
        inputs[argnums],
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return hess


def _hessian(
    f: Callable[..., Tensor],
    inputs: tuple[Any, ...],
    argnums: int = 0,
    create_graph: bool | None = None,
    retain_graph: bool = True,
    is_batched: bool = False,
) -> Tensor:
    """
    Wrapper for Hessian. The Hessian is the Jacobian of the gradient.

    PyTorch, however, suggests calculating the Jacobian of the Jacobian, which
    does not yield the correct shape in this case.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    inputs : tuple[Any, ...]
        The input parameters of `f`.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Tensor
        The Hessian.
    """
    if is_batched:
        raise NotImplementedError("Batched Hessian not available.")

    if create_graph is None:
        create_graph = torch.is_grad_enabled()
    assert create_graph is not None

    def _grad(*inps: tuple[Any, ...]) -> Tensor:
        e = f(*inps).sum()

        if not isinstance(inps[argnums], Tensor):
            raise RuntimeError(f"The {argnums}'th input parameter must be a tensor.")

        # catch missing gradients (e.g., halogen bond correction evaluates to
        # zero if no donors/acceptors are present)
        if e.grad_fn is None:
            return torch.zeros_like(inps[argnums])  # type: ignore

        (g,) = torch.autograd.grad(
            e,
            inps[argnums],
            create_graph=create_graph,
            retain_graph=retain_graph,
        )
        return g

    # NOTE: This is a (non-vectorized, slow) workaround that probably only
    # works for the nuclear Hessian! The use of functorch causes issues!
    def _jacobian(a: Tensor, b: Tensor) -> Tensor:
        # catch missing gradients (e.g., halogen bond correction evaluates to
        # zero if no donors/acceptors are present)
        if a.grad_fn is None:
            return torch.zeros(
                (*b.shape, *b.shape),
                dtype=b.dtype,
                device=b.device,
            )

        aflat = a.reshape(-1)
        res = torch.empty(
            (a.numel(), b.numel()),
            dtype=a.dtype,
            device=a.device,
        )

        for i in range(aflat.numel()):
            (g,) = torch.autograd.grad(
                aflat[i],
                b,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )
            res[i] = g.reshape(-1)

        return res.reshape((*b.shape, *b.shape))

    grad = _grad(*inputs)
    hess = _jacobian(grad, inputs[argnums])
    return hess


#####################################
# functorch-reliant implementations #
#####################################


def hessian_functorch(
    f: Callable[..., Tensor],
    inputs: tuple[Any, ...],
    argnums: int = 0,
    is_batched: bool = False,
) -> Tensor:
    """
    Wrapper for Hessian. The Hessian is the Jacobian of the gradient.

    PyTorch, however, suggests calculating the Jacobian of the Jacobian, which
    does not yield the correct shape in this case.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    inputs : tuple[Any, ...]
        The input parameters of `f`.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.

    Returns
    -------
    Tensor
        The Hessian.
    """

    def _grad(*inps: tuple[Any, ...]) -> Tensor:
        e = f(*inps).sum()

        if not isinstance(inps[argnums], Tensor):
            raise RuntimeError(f"The {argnums}'th input parameter must be a tensor.")

        # catch missing gradients (e.g., halogen bond correction evaluates to
        # zero if no donors/acceptors are present)
        if e.grad_fn is None:
            return torch.zeros_like(inps[argnums])  # type: ignore

        (g,) = torch.autograd.grad(
            e,
            inps[argnums],
            create_graph=True,
        )
        return g

    _jacobian = jacrev(_grad, argnums=argnums)

    if is_batched:
        raise NotImplementedError("Batched Hessian not available.")
        # dims = tuple(None if x != argnums else 0 for x in range(len(inputs)))
        # _jacobian = torch.func.vmap(_jacobian, in_dims=dims)

    return _jacobian(*inputs)  # type: ignore

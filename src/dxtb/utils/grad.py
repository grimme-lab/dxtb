"""
Autograd utility
================

This module contains utility functions for automaic differentiation.

Important! Before PyTorch 2.0.0, `functorch` does not work together with custom
autograd functions, which we definitely require. Additionally, `functorch`
imposes the implementation of a `forward` **and** `setup_context` method, i.e.,
the traditional way of using `forward` with the `ctx` argument does not work
"""
from __future__ import annotations

import torch

from ..__version__ import __torch_version__
from .._types import Any, Callable, Tensor

if __torch_version__ < (2, 0, 0):  # type: ignore, pragma: no cover
    try:
        from functorch import jacrev  # type: ignore
    except ModuleNotFoundError:
        jacrev = None
        from torch.autograd.functional import jacobian  # type: ignore

else:  # pragma: no cover
    from torch.func import jacrev  # type: ignore


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

    # NOTE: This is a (non-vectorized, slow) workaround that probably only
    # works for the nuclear Hessian! The use of functorch causes issues!
    def _jac(a: Tensor, b: Tensor) -> Tensor:
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
    hess = _jac(grad, inputs[argnums])
    return hess


########################################
# `torch.func`-reliant implementations #
########################################


def jac(f: Callable[..., Tensor], argnums: int = 0) -> Any:
    """
    Wrapper for Jacobian calcluation.

    Parameters
    ----------
    f : Callable[[Any], Tensor]
        The function whose result is differentiated.
    argnums : int, optional
        The variable w.r.t. which will be differentiated. Defaults to 0.
    """
    if jacrev is None:  # pragma: no cover

        def wrap(*args) -> Any:
            return jacobian(f, *args)  # type: ignore. pylint: disable=used-before-assignment

        return wrap

    return jacrev(f, argnums=argnums)  # type: ignore


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

    _jac = jac(_grad, argnums=argnums)

    if is_batched:
        raise NotImplementedError("Batched Hessian not available.")
        # dims = tuple(None if x != argnums else 0 for x in range(len(inputs)))
        # _jac = torch.func.vmap(_jac, in_dims=dims)

    return _jac(*inputs)  # type: ignore

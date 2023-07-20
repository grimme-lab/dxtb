"""
Collection of utility functions for testing.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from dxtb._types import Any, Callable, Protocol, Tensor, TensorOrTensors

from .conftest import FAST_MODE

coordfile = Path(Path(__file__).parent, "test_singlepoint/mols/H2/coord").resolve()
"""Path to coord file of H2."""


def get_device_from_str(s: str) -> torch.device:
    """
    Convert device name to `torch.device`. Critically, this also sets the index
    for CUDA devices to `torch.cuda.current_device()`.

    Parameters
    ----------
    s : str
        Name of the device as string.

    Returns
    -------
    torch.device
        Device as torch class.

    Raises
    ------
    KeyError
        Unknown device name is given.
    """
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Torch not compiled with CUDA or no CUDA device available."
            )
        return torch.device("cuda", index=torch.cuda.current_device())

    raise KeyError(f"Unknown device '{s}' given.")


def load_from_npz(
    npzfile: Any,
    name: str,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Get torch tensor from npz file

    Parameters
    ----------
    npzfile : Any
        Loaded npz file.
    name : str
        Name of the tensor in the npz file.
    dtype : torch.dtype
        Data type of the tensor.
    device : torch.device | None
        Device of the tensor. Defaults to `None`.

    Returns
    -------
    Tensor
        Tensor from the npz file.
    """
    name = name.replace("-", "").replace("+", "").lower()
    return torch.from_numpy(npzfile[name]).to(device=device, dtype=dtype)


def nth_derivative(f: Tensor, x: Tensor, n: int = 1) -> Tensor:
    """
    Calculate the *n*th derivative of a tensor.

    Parameters
    ----------
    f : Tensor
        Input tensor of which the gradient should be calculated.
    x : Tensor
        Dependent variable (must have `requires_grad`)
    n : int, optional
        Order of the derivative. Defaults to 1.

    Returns
    -------
    Tensor
        The *n*th order derivative of `f` w.r.t. `x`.

    Raises
    ------
    ValueError
        Order of derivative is smaller than 1 or not an integer.
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError("Order of derivative must be an integer and larger 1.")

    grads = None
    for _ in range(n):
        grads = torch.autograd.grad(f, x, create_graph=True)[0]
        f = grads.sum()

    assert grads is not None
    return grads


class _GradcheckFunction(Protocol):
    """
    Type annotation for gradcheck function.
    """

    def __call__(
        self,
        func: Callable[..., TensorOrTensors],
        inputs: TensorOrTensors,
        *,
        eps: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        raise_exception: bool = True,
        check_sparse_nnz: bool = False,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
        check_batched_grad: bool = False,
        check_batched_forward_grad: bool = False,
        check_forward_ad: bool = False,
        check_backward_ad: bool = True,
        fast_mode: bool = False,
    ) -> bool:
        ...


class _GradgradcheckFunction(Protocol):
    """
    Type annotation for gradgradcheck function.
    """

    def __call__(
        self,
        func: Callable[..., TensorOrTensors],
        inputs: TensorOrTensors,
        grad_outputs: TensorOrTensors | None = None,
        *,
        eps: float = 1e-6,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        gen_non_contig_grad_outputs: bool = False,
        raise_exception: bool = True,
        nondet_tol: float = 0.0,
        check_undefined_grad: bool = True,
        check_grad_dtypes: bool = False,
        check_batched_grad: bool = False,
        check_fwd_over_rev: bool = False,
        check_rev_over_rev: bool = True,
        fast_mode: bool = False,
    ) -> bool:
        ...


def _wrap_gradcheck(
    gradcheck_func: _GradcheckFunction | _GradgradcheckFunction,
    func: Callable[..., TensorOrTensors],
    diffvars: TensorOrTensors,
    **kwargs,
) -> bool:
    fast_mode = kwargs.pop("fast_mode", FAST_MODE)
    try:
        assert gradcheck_func(func, diffvars, fast_mode=fast_mode, **kwargs)
    finally:
        if isinstance(diffvars, Tensor):
            diffvars.detach_()
        else:
            for diffvar in diffvars:
                diffvar.detach_()

    return True


def dgradcheck(
    func: Callable[..., TensorOrTensors], diffvars: TensorOrTensors, **kwargs
) -> bool:
    """
    Wrapper for `torch.autograd.gradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradcheck, func, diffvars, **kwargs)


def dgradgradcheck(
    func: Callable[..., TensorOrTensors], diffvars: TensorOrTensors, **kwargs
) -> bool:
    """
    Wrapper for `torch.autograd.gradgradcheck` that detaches the differentiated
    variables after the check.

    Parameters
    ----------
    func : Callable[..., TensorOrTensors]
        Forward function.
    diffvars : TensorOrTensors
        Variables w.r.t. which we differentiate.

    Returns
    -------
    bool
        Status of check.
    """
    return _wrap_gradcheck(gradgradcheck, func, diffvars, **kwargs)

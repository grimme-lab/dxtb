import pytest
import numpy as np
import functools
import torch
import scipy.linalg
from torch.autograd import gradcheck

from xtbml.exlibs.tbmalt import batch
from xtbml.scf import solver


def fix_seed(func):
    """Sets torch's & numpy's random number generator seed"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Set both numpy's and pytorch's seed to zero
        np.random.seed(0)
        torch.manual_seed(0)

        return func(*args, **kwargs)

    return wrapper


@fix_seed
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eighb_standard_single(dtype, n=10):
    """eighb accuracy on a single standard eigenvalue problem."""
    tol = 1.0e-6
    a = solver.symmetrize(torch.rand(n, n))

    w_ref = torch.tensor(scipy.linalg.eigh(a)[0])

    w_calc, v_calc = solver.eighb(a)

    assert torch.allclose(w_calc.cpu(), w_ref, atol=tol)
    assert torch.allclose(v_calc @ v_calc.T, torch.eye(*v_calc.shape), atol=tol)


@fix_seed
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eighb_standard_batch(dtype):
    """eighb accuracy on a batch of standard eigenvalue problems."""
    tol = 1.0e-6
    sizes = torch.randint(2, 10, (11,))
    a = [solver.symmetrize(torch.rand(s, s, dtype=dtype)) for s in sizes]
    a_batch = batch.pack(a)

    w_ref = batch.pack(
        [torch.tensor(scipy.linalg.eigh(i.cpu())[0], dtype=dtype) for i in a]
    )

    w_calc = solver.eighb(a_batch)[0]

    assert torch.allclose(w_calc.cpu(), w_ref, atol=tol)


@fix_seed
@pytest.mark.parametrize("cholesky", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eighb_general_single(dtype, cholesky, n=8):
    """eighb accuracy on a single general eigenvalue problem."""

    tol = 1.0e-6
    a = solver.symmetrize(torch.rand(n, n, dtype=dtype))
    b = solver.symmetrize(torch.eye(n, dtype=dtype) * torch.rand(n, dtype=dtype))

    w_ref = torch.tensor(scipy.linalg.eigh(a, b)[0], dtype=dtype)

    w_calc, v_calc = solver.eighb(a, b, cholesky=cholesky)

    assert torch.allclose(w_calc.cpu(), w_ref, atol=tol)
    assert torch.allclose(v_calc @ v_calc.T, torch.inverse(b), atol=tol)


@fix_seed
@pytest.mark.parametrize("cholesky", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_eighb_general_batch(dtype, cholesky):
    """eighb accuracy on a batch of general eigenvalue problems."""

    tol = 1.0e-6
    sizes = torch.randint(2, 10, (11,))
    a = [solver.symmetrize(torch.rand(s, s, dtype=dtype)) for s in sizes]
    b = [
        solver.symmetrize(torch.eye(s, dtype=dtype) * torch.rand(s, dtype=dtype))
        for s in sizes
    ]
    a_batch, b_batch = batch.pack(a), batch.pack(b)

    w_ref = batch.pack(
        [torch.tensor(scipy.linalg.eigh(i, j)[0], dtype=dtype) for i, j in zip(a, b)]
    )

    w_calc = solver.eighb(a_batch, b_batch, cholesky=cholesky)[0]

    assert torch.allclose(w_calc.cpu(), w_ref, atol=tol)


@pytest.mark.grad
@fix_seed
def test_standard_grad(dtype=torch.float64):
    """eighb gradient stability on standard, broadened, eigenvalue problems."""

    def eighb_proxy(m):
        m = solver.symmetrize(m)
        return solver.eighb(m)

    # Generate a single standard eigenvalue test instance
    a1 = solver.symmetrize(torch.rand(8, 8, dtype=dtype))
    a1.requires_grad = True

    assert gradcheck(eighb_proxy, (a1,), raise_exception=False)


@pytest.mark.grad
@fix_seed
def test_general_grad(dtype=torch.float64):
    """eighb gradient stability on general eigenvalue problems."""

    def eighb_proxy(m, n, size_data=None):
        m, n = solver.symmetrize(m), solver.symmetrize(n)
        return solver.eighb(m, n)

    # Generate a single general eigenvalue test instance
    a1 = solver.symmetrize(torch.rand(8, 8, dtype=dtype))
    b1 = solver.symmetrize(torch.eye(8, dtype=dtype) * torch.rand(8, dtype=dtype))
    a1.requires_grad, b1.requires_grad = True, True

    assert gradcheck(eighb_proxy, (a1, b1), raise_exception=False)

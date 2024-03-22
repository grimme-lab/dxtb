"""
Test diagonalization.
"""

from __future__ import annotations

import pytest
import torch
from torch.autograd.gradcheck import gradcheck

from dxtb._types import Literal, Tensor
from dxtb.exlibs.xitorch import LinearOperator
from dxtb.exlibs.xitorch.linalg import symeig
from dxtb.utils import eighb, symmetrize

# Hamiltonian of LiH from last step
hamiltonian = torch.tensor(
    [
        [
            -0.27474006548256,
            -0.00000000000000,
            -0.00000000000000,
            -0.00000000000000,
            -0.22679941570507,
            0.07268461913372,
        ],
        [
            -0.00000000000000,
            -0.17641725918816,
            -0.00000000000000,
            -0.00000000000000,
            0.00000000000000,
            0.00000000000000,
        ],
        [
            -0.00000000000000,
            -0.00000000000000,
            -0.17641725918816,
            -0.00000000000000,
            -0.28474359171632,
            0.02385107216679,
        ],
        [
            -0.00000000000000,
            -0.00000000000000,
            -0.00000000000000,
            -0.17641725918816,
            0.00000000000000,
            0.00000000000000,
        ],
        [
            -0.22679941570507,
            0.00000000000000,
            -0.28474359171632,
            0.00000000000000,
            -0.33620576141638,
            0.00000000000000,
        ],
        [
            0.07268461913372,
            0.00000000000000,
            0.02385107216679,
            0.00000000000000,
            0.00000000000000,
            -0.01268791523447,
        ],
    ],
    dtype=torch.float64,
)


@pytest.mark.parametrize("broadening", ["none", "cond", "lorn"])
def test_eighb(broadening: Literal["cond", "lorn", "none"]) -> None:
    a = torch.rand(8, 8, dtype=torch.double)
    a.requires_grad_(True)

    def eigen_proxy(m: Tensor):
        m = symmetrize(m, force=True)
        return eighb(a=m, broadening_method=broadening)

    assert gradcheck(eigen_proxy, a)


@pytest.mark.xfail
@pytest.mark.parametrize("broadening", ["none", "cond", "lorn"])
def test_eighb_degen(broadening: Literal["cond", "lorn", "none"]) -> None:
    hamiltonian.detach_().requires_grad_(True)

    def eigen_proxy(m: Tensor):
        m = symmetrize(m, force=True)
        return eighb(a=m, broadening_method=broadening)

    assert gradcheck(eigen_proxy, hamiltonian)


def test_xtlsymeig() -> None:
    a = torch.rand(8, 8, dtype=torch.double)
    a.requires_grad_(True)

    def eigen_proxy(m: Tensor):
        m = symmetrize(m, force=True)
        m_op = LinearOperator.m(m, is_hermitian=True)
        return symeig(m_op)

    assert gradcheck(eigen_proxy, a)


@pytest.mark.xfail
def test_xtlsymeig_degen() -> None:
    hamiltonian.detach_().requires_grad_(True)

    def eigen_proxy(m: Tensor):
        m = symmetrize(m, force=True)
        m_op = LinearOperator.m(m, is_hermitian=True)
        return symeig(m_op, bck_options={"degen_rtol": 1e-1})

    assert gradcheck(eigen_proxy, hamiltonian)

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
Test diagonalization.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import symmetrize
from tad_mctc.storch.linalg import eighb
from torch.autograd.gradcheck import gradcheck

from dxtb._src.exlibs.xitorch import LinearOperator
from dxtb._src.exlibs.xitorch.linalg import symeig
from dxtb._src.typing import Literal, Tensor

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


@pytest.mark.parametrize("broadening", [None, "cond", "lorn"])
def test_eighb(broadening: Literal["cond", "lorn"] | None) -> None:
    a = torch.rand(8, 8, dtype=torch.double)
    a.requires_grad_(True)

    def eigen_proxy(m: Tensor):
        m = symmetrize(m, force=True)
        return eighb(a=m, broadening_method=broadening)

    assert gradcheck(eigen_proxy, a)


@pytest.mark.xfail
@pytest.mark.parametrize("broadening", [None, "cond", "lorn"])
def test_eighb_degen(broadening: Literal["cond", "lorn"] | None) -> None:
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

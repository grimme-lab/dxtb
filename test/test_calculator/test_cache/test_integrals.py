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
Test caching integrals.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Tensor
from dxtb.calculators import GFN1Calculator

from ...conftest import DEVICE

opts = {"cache_enabled": True, "verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_deleted(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    calc = GFN1Calculator(numbers, opts={"verbosity": 0}, **dd)
    assert calc._ncalcs == 0

    # overlap should not be cached
    assert calc.opts.cache.store.overlap == False

    # one successful calculation
    energy = calc.get_energy(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    # cache should be empty
    assert calc.cache.overlap is None

    # ... but also the tensors in the calculator should be deleted
    assert calc.integrals.overlap is not None
    assert calc.integrals.overlap._matrix is None
    assert calc.integrals.overlap._norm is None
    assert calc.integrals.overlap._gradient is None


def overlap_retained_for_grad(dtype: torch.dtype, intdriver: int) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd, requires_grad=True
    )

    calc = GFN1Calculator(
        numbers, opts={"verbosity": 0, "int_driver": intdriver}, **dd
    )
    assert calc._ncalcs == 0

    # overlap should not be cached
    assert calc.opts.cache.store.overlap == False

    # one successful calculation
    energy = calc.get_energy(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    # cache should still be empty ...
    assert calc.cache.overlap is None

    # ... but the tensors in the calculator should still be there
    assert calc.integrals.overlap is not None
    assert calc.integrals.overlap._matrix is not None
    assert calc.integrals.overlap._norm is not None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_retained_for_grad_pytorch(dtype: torch.dtype) -> None:
    overlap_retained_for_grad(dtype, labels.INTDRIVER_AUTOGRAD)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_retained_for_grad_libcint(dtype: torch.dtype) -> None:
    overlap_retained_for_grad(dtype, labels.INTDRIVER_LIBCINT)

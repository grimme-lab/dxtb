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
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.autograd import dgradcheck

from dxtb.basis import IndexHelper
from dxtb.components.interactions.coulomb import thirdorder as es3
from dxtb.param import GFN1_XTB, get_elem_param
from dxtb.typing import DD, Tensor
from dxtb.utils import batch

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02", "SiH4_atom"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test ES3 for some samples from MB16_43."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    qat = sample["q"].to(**dd)
    ref = sample["es3"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    es = es3.new_es3(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(numbers=numbers, ihelp=ihelp)
    e = es.get_atom_energy(qat, cache)
    assert pytest.approx(torch.sum(e, dim=-1)) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Test batched calculation of ES3."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    qat = batch.pack(
        (
            sample1["q"].to(**dd),
            sample2["q"].to(**dd),
        )
    )
    ref = torch.stack(
        [
            sample1["es3"].to(**dd),
            sample2["es3"].to(**dd),
        ],
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    es = es3.new_es3(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(numbers=numbers, ihelp=ihelp)
    e = es.get_atom_energy(qat, cache)
    assert torch.allclose(torch.sum(e, dim=-1), ref)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name: str) -> None:
    """Test autograd for ES3 parameters."""
    dd: DD = {"device": device, "dtype": torch.double}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    qat = sample["q"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    hd = get_elem_param(
        torch.unique(numbers),
        GFN1_XTB.element,
        "gam3",
        **dd,
    )

    # variable to be differentiated
    hd.requires_grad_(True)

    def func(hubbard_derivs: Tensor):
        es = es3.ES3(hubbard_derivs, **dd)
        cache = es.get_cache(numbers=numbers, ihelp=ihelp)
        return es.get_atom_energy(qat, cache)

    assert dgradcheck(func, hd)

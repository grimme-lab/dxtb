# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
from math import sqrt

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper
from dxtb._src.components.interactions.spin import factory
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples


# test if the spin constants of the right elements are pulled
@pytest.mark.parametrize("name", ["LiH"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_spinconstants(dtype: torch.dtype, name: str) -> None:

    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"dtype": dtype, "device": DEVICE}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample["spconst"].to(**dd)

    spinconst = factory._load_spin_constants(**dd)[numbers]

    assert pytest.approx(ref.cpu(), abs=tol) == spinconst.cpu()


# test if the wll matrix is set up correctly for both GFN1_XTB and GFN2_XTB
@pytest.mark.parametrize("name", ["LiH", "SiH4"])
@pytest.mark.parametrize(
    "model_cls, ref_key",
    [
        (GFN1_XTB, "wllgfn1"),
        (GFN2_XTB, "wllgfn2"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_wll(dtype: torch.dtype, name: str, model_cls, ref_key) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    ref = sample[ref_key].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, model_cls)
    spin = factory.new_spinpolarisation(numbers, **dd)

    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    assert pytest.approx(ref.cpu(), abs=tol) == cache.wll.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("model_cls", [GFN1_XTB, GFN2_XTB])
def test_wll_batch(dtype: torch.dtype, model_cls) -> None:
    tol = sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    names = ("LiH", "SiH4")

    numbers = pack(tuple(samples[name]["numbers"].to(DEVICE) for name in names))
    ihelp = IndexHelper.from_numbers(numbers, model_cls)
    spin = factory.new_spinpolarisation(numbers, **dd)
    cache = spin.get_cache(numbers=numbers, ihelp=ihelp)

    assert cache.wll.shape == torch.Size((len(names), ihelp.nsh, ihelp.nsh))

    for batch_idx, name in enumerate(names):
        numbers_single = samples[name]["numbers"].to(DEVICE)
        ihelp_single = IndexHelper.from_numbers(numbers_single, model_cls)
        spin_single = factory.new_spinpolarisation(numbers_single, **dd)
        cache_single = spin_single.get_cache(
            numbers=numbers_single, ihelp=ihelp_single
        )

        nsh_single = ihelp_single.nsh
        wll_block = cache.wll[batch_idx, :nsh_single, :nsh_single]

        assert pytest.approx(cache_single.wll.cpu(), abs=tol) == wll_block.cpu()

        if nsh_single < ihelp.nsh:
            assert (
                torch.count_nonzero(cache.wll[batch_idx, nsh_single:, :]) == 0
            )
            assert (
                torch.count_nonzero(cache.wll[batch_idx, :, nsh_single:]) == 0
            )

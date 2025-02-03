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
Test caches.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, IndexHelper, Param
from dxtb._src.typing import Callable, Tensor
from dxtb.components.base import Classical, ComponentCache
from dxtb.components.dispersion import new_dispersion
from dxtb.components.halogen import new_halogen
from dxtb.components.repulsion import new_repulsion

from ..conftest import DEVICE


@pytest.mark.parametrize(
    "comp_factory_par",
    [
        (new_repulsion, GFN1_XTB),
        (new_halogen, GFN1_XTB),
        (new_dispersion, GFN1_XTB),
    ],
)
def test_fail_overwritten_cache(
    comp_factory_par: tuple[Callable[[Tensor, Param], Classical], Param],
) -> None:
    numbers = torch.tensor([3, 1], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    comp_factory, par = comp_factory_par
    comp = comp_factory(numbers, par)
    assert comp is not None

    # create cache
    comp.cache_enable()
    _ = comp.get_cache(numbers=numbers, ihelp=ihelp)

    # manually overwrite cache
    comp.cache = ComponentCache()

    with pytest.raises(TypeError):
        comp.get_cache(numbers=numbers, ihelp=ihelp)


def test_fail_overwritten_cache_d4() -> None:
    numbers = torch.tensor([3, 1], device=DEVICE)

    par = GFN2_XTB.model_copy(deep=True)
    par.dispersion.d4.sc = False  # type: ignore

    d4 = new_dispersion(numbers, par, charge=torch.tensor(0.0, device=DEVICE))
    assert d4 is not None

    # create cache
    d4.cache_enable()
    _ = d4.get_cache(numbers=numbers)

    # manually overwrite cache
    d4.cache = ComponentCache()

    with pytest.raises(TypeError):
        d4.get_cache(numbers=numbers)

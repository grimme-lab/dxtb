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
from dxtb.components.base import Interaction, InteractionCache
from dxtb.components.coulomb import new_es2, new_es3
from dxtb.components.dispersion import new_d4sc
from dxtb.components.field import new_efield, new_efield_grad
from dxtb.components.solvation import new_solvation

from ..conftest import DEVICE


@pytest.mark.parametrize(
    "comp_factory_par",
    [
        (new_d4sc, GFN2_XTB),
        (new_es2, GFN1_XTB),
        (new_es3, GFN1_XTB),
    ],
)
def test_fail_overwritten_cache(
    comp_factory_par: tuple[Callable[[Tensor, Param], Interaction], Param]
) -> None:
    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    comp_factory, par = comp_factory_par
    comp = comp_factory(numbers, par)
    assert comp is not None

    # create cache
    comp.cache_enable()
    _ = comp.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    # manually overwrite cache
    comp.cache = InteractionCache()

    with pytest.raises(TypeError):
        comp.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)


def test_fail_overwritten_cache_ef() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)

    ef = new_efield(torch.tensor([0.0, 0.0, 0.0]), device=DEVICE)
    assert ef is not None

    # create cache
    ef.cache_enable()
    _ = ef.get_cache(positions=positions)

    # manually overwrite cache
    ef.cache = InteractionCache()

    with pytest.raises(TypeError):
        ef.get_cache(positions=positions)


def test_fail_overwritten_cache_efg() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)

    efg = new_efield_grad(torch.zeros((3, 3)), device=DEVICE)
    assert efg is not None

    # create cache
    efg.cache_enable()
    _ = efg.get_cache(positions=positions)

    # manually overwrite cache
    efg.cache = InteractionCache()

    with pytest.raises(TypeError):
        efg.get_cache(positions=positions)


def test_fail_overwritten_cache_solvation() -> None:
    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=DEVICE)
    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    # manually create solvation
    from dxtb._src.param.solvation import ALPB, Solvation

    par = GFN1_XTB.model_copy(deep=True)
    par.solvation = Solvation(
        alpb=ALPB(alpb=True, kernel="p16", born_scale=1.0, born_offset=0.0)
    )

    solv = new_solvation(numbers, par)
    assert solv is not None

    # create cache
    solv.cache_enable()
    _ = solv.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

    # manually overwrite cache

    solv.cache = InteractionCache()

    with pytest.raises(TypeError):
        solv.get_cache(numbers=numbers, positions=positions, ihelp=ihelp)

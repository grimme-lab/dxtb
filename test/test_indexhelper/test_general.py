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
General tests for IndexHelper covering instantiation, changing dtypes, and
moving to devices.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.convert import str_to_device

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, IndexHelper, ParamModule
from dxtb._src.param import Param
from dxtb._src.param.element import get_elem_angular

from ..conftest import DEVICE


def test_fail_init_dtype() -> None:
    numbers = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    with pytest.raises(ValueError):
        IndexHelper(
            ihelp.unique_angular.type(torch.float),
            ihelp.angular,
            ihelp.atom_to_unique,
            ihelp.ushells_to_unique,
            ihelp.ushells_per_unique,
            ihelp.shells_to_ushell,
            ihelp.shells_per_atom,
            ihelp.shell_index,
            ihelp.shells_to_atom,
            ihelp.orbitals_per_shell,
            ihelp.orbital_index,
            ihelp.orbitals_to_shell,
            batch_mode=0,
            device=DEVICE,
        )


@pytest.mark.cuda
def test_fail_init_device() -> None:
    numbers = torch.tensor([1], device=torch.device("cpu"))
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    with pytest.raises(ValueError):
        IndexHelper(
            ihelp.unique_angular.to(str_to_device("cuda")),
            ihelp.angular,
            ihelp.atom_to_unique,
            ihelp.ushells_to_unique,
            ihelp.ushells_per_unique,
            ihelp.shells_to_ushell,
            ihelp.shells_per_atom,
            ihelp.shell_index,
            ihelp.shells_to_atom,
            ihelp.orbitals_per_shell,
            ihelp.orbital_index,
            ihelp.orbitals_to_shell,
            batch_mode=0,
        )


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_change_type(dtype: torch.dtype) -> None:
    numbers = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    ihelp = ihelp.type(dtype)
    assert ihelp.dtype == dtype


def test_change_type_fail() -> None:
    numbers = torch.tensor([1], device=DEVICE)
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]})

    # trying to use setter
    with pytest.raises(AttributeError):
        ihelp.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        ihelp.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = str_to_device(device_str)

    numbers = torch.tensor([1])
    ihelp = IndexHelper.from_numbers_angular(numbers, {1: [0]}).to(device)
    assert ihelp.device == device


def test_change_device_fail() -> None:
    ihelp = IndexHelper.from_numbers_angular(torch.tensor([1]), {1: [0]})

    # trying to use setter
    with pytest.raises(AttributeError):
        ihelp.device = "cpu"


@pytest.mark.parametrize("par", [GFN0_XTB, GFN1_XTB, GFN2_XTB])
def test_from_numbers_param_matches_module(par: Param) -> None:
    """Raw `Param` and `ParamModule` yield the same index helper."""
    numbers = torch.tensor([1, 6, 8, 14, 17, 26, 79], device=DEVICE)

    parmod = ParamModule(par)
    assert get_elem_angular(par.element) == parmod.get_elem_angular()

    ihelp_par = IndexHelper.from_numbers(numbers, par)
    ihelp_mod = IndexHelper.from_numbers(numbers, parmod)

    assert (ihelp_par.angular == ihelp_mod.angular).all()
    assert (ihelp_par.shells_to_atom == ihelp_mod.shells_to_atom).all()
    assert (ihelp_par.orbitals_to_shell == ihelp_mod.orbitals_to_shell).all()


@pytest.mark.parametrize("module", [False, True])
def test_from_numbers_unknown_shell(module: bool) -> None:
    """Raw `Param` and `ParamModule` reject unknown shell types alike."""
    par = GFN1_XTB.model_copy(deep=True)
    par.element["H"].shells = ["1s", "5h"]

    numbers = torch.tensor([1], device=DEVICE)
    with pytest.raises(ValueError, match="Unknown shell type 'h'"):
        IndexHelper.from_numbers(numbers, ParamModule(par) if module else par)


# def test_cache() -> None:
#     ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})

#     # run a memoized function
#     _ = ihelp.orbitals_to_shell_cart

#     # get cache
#     fcn = ihelp._orbitals_to_shell_cart  # pylint: disable=protected-access
#     cache = fcn.get_cache()

#     # cache should only have one entry
#     assert len(cache) == 1

#     # the key is created from the function name, so check if it is really there
#     assert fcn.__name__ in tuple(*cache.keys())

#     # clear cache and check if it is really empty
#     ihelp.clear_cache()
#     cache = fcn.get_cache()
#     assert len(cache) == 0

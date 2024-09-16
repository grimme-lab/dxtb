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
Test Calculator usage.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB
from dxtb._src.calculators.properties.vibration import VibResult
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import DD, Literal, Tensor
from dxtb.calculators import (
    AnalyticalCalculator,
    AutogradCalculator,
    GFN1Calculator,
)

from ...conftest import DEVICE

opts = {"cache_enabled": True, "verbosity": 0}


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_energy(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)

    calc = GFN1Calculator(numbers, opts=opts, **dd)
    assert calc._ncalcs == 0

    energy = calc.get_energy(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    # cache is used
    energy = calc.get_energy(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    # different name for energy getter
    energy = calc.get_potential_energy(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    # get other properties
    energy = calc.get_iterations(positions)
    assert calc._ncalcs == 1
    assert isinstance(energy, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("grad_mode", ["functorch", "row"])
def test_forces(
    dtype: torch.dtype, grad_mode: Literal["functorch", "row"]
) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_forces(pos, grad_mode=grad_mode)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for same calc
    prop = calc.get_forces(pos, grad_mode=grad_mode)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for energy
    prop = calc.get_energy(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_forces_analytical(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    calc = AnalyticalCalculator(numbers, GFN1_XTB, opts=opts, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_forces(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for same calc
    prop = calc.get_forces(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for energy
    prop = calc.get_energy(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("use_functorch", [False, True])
def test_hessian(dtype: torch.dtype, use_functorch: bool) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_hessian(pos, use_functorch=use_functorch)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for same calc
    assert "hessian" in calc.cache.list_cached_properties()
    prop = calc.get_hessian(pos, use_functorch=use_functorch)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for energy
    assert "energy" in calc.cache.list_cached_properties()
    prop = calc.get_energy(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for forces (needs `functorch` to be equivalent)
    assert "forces" in calc.cache.list_cached_properties()
    prop = calc.get_forces(pos, grad_mode="functorch")
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("use_functorch", [False, True])
def test_vibration(dtype: torch.dtype, use_functorch: bool) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_normal_modes(pos, use_functorch=use_functorch)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)
    assert "normal_modes" in calc.cache.list_cached_properties()

    # cache is used for freqs
    assert "frequencies" in calc.cache.list_cached_properties()
    prop = calc.get_frequencies(pos, use_functorch=use_functorch)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for full vibration result
    assert "vibration" in calc.cache.list_cached_properties()
    prop = calc.get_vibration(pos, use_functorch=use_functorch)
    assert calc._ncalcs == 1
    assert isinstance(prop, VibResult)

    # cache is used for forces (needs `functorch` to be equivalent)
    assert "forces" in calc.cache.list_cached_properties()
    grad_mode = "autograd" if use_functorch is False else "functorch"
    prop = calc.get_forces(pos, grad_mode=grad_mode)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for hessian (needs `matrix=False` bo be equivalent)
    assert "hessian" in calc.cache.list_cached_properties()
    prop = calc.get_hessian(pos, use_functorch=use_functorch, matrix=False)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_dipole(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_dipole(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for same calc
    prop = calc.get_dipole_moment(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for energy
    prop = calc.get_energy(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_ir(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_ir(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for same calc
    prop = calc.get_ir(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for energy
    prop = calc.get_energy(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for IR intensities
    prop = calc.get_ir_intensities(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_raman(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], **dd)
    pos = positions.clone().requires_grad_(True)

    options = dict(opts, **{"scf_mode": "full", "mixer": "anderson"})
    calc = AutogradCalculator(numbers, GFN1_XTB, opts=options, **dd)
    assert calc._ncalcs == 0

    prop = calc.get_raman(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for same calc
    prop = calc.get_raman(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for energy
    prop = calc.get_energy(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for Raman depolarization ratio
    prop = calc.get_raman_depol(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # cache is used for IR intensities
    prop = calc.get_raman_intensities(pos)
    assert calc._ncalcs == 1
    assert isinstance(prop, Tensor)

    # check reset
    calc.cache.reset_all()
    assert len(calc.cache.list_cached_properties()) == 0

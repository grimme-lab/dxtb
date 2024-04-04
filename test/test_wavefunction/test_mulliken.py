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
Test for Mulliken population analysis and charges.
Reference values obtained with xTB 6.5.1 and tblite 0.2.1.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.basis.indexhelper import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD
from dxtb.utils import batch
from dxtb.wavefunction import mulliken
from dxtb.xtb import GFN1Hamiltonian as Hamiltonian

from .samples import samples

sample_list = ["H2", "LiH", "SiH4"]

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_number_electrons(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["n_electrons"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)

    pop = mulliken.get_atomic_populations(overlap, density, ihelp)
    assert pytest.approx(ref, rel=1e-7, abs=tol) == torch.sum(pop, dim=-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_number_electrons(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    density = batch.pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["n_electrons"].to(**dd),
            sample2["n_electrons"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, par)

    pop = mulliken.get_atomic_populations(overlap, density, ihelp)
    assert pytest.approx(ref, rel=1e-7, abs=tol) == torch.sum(pop, dim=-1)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_pop_shell(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["mulliken_pop"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    pop = mulliken.get_shell_populations(overlap, density, ihelp)

    assert pytest.approx(ref, abs=1e-5) == pop


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_pop_shell(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    density = batch.pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["mulliken_pop"].to(**dd),
            sample2["mulliken_pop"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    pop = mulliken.get_shell_populations(overlap, density, ihelp)

    assert pytest.approx(ref, abs=1e-5) == pop


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-5

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["mulliken_charges"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref, abs=tol) == pop


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_charges(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    density = batch.pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["mulliken_charges"].to(**dd),
            sample2["mulliken_charges"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref, abs=tol) == pop


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges_shell(dtype: torch.dtype, name: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-5

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["mulliken_charges_shell"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_shell(h0.get_occupation())

    pop = mulliken.get_mulliken_shell_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref, abs=tol) == pop


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_charges_shell(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": device, "dtype": dtype}
    tol = 1e-5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    density = batch.pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = batch.pack(
        (
            sample1["mulliken_charges_shell"].to(**dd),
            sample2["mulliken_charges_shell"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_shell(h0.get_occupation())

    pop = mulliken.get_mulliken_shell_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref, abs=tol) == pop

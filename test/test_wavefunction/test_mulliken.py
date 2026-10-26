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
from tad_mctc.batch import pack

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.typing import DD
from dxtb._src.wavefunction import mulliken
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian

from ..conftest import DEVICE
from ..utils import get_param_module
from .samples import samples

sample_list = ["H2", "LiH", "SiH4"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_number_electrons(dtype: torch.dtype, name: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["n_electrons"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    pop = mulliken.get_atomic_populations(overlap, density, ihelp)
    assert pytest.approx(ref.cpu(), rel=1e-7, abs=tol) == pop.sum(-1).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_number_electrons(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = torch.finfo(dtype).eps ** 0.5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    density = pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["n_electrons"].to(**dd),
            sample2["n_electrons"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)

    pop = mulliken.get_atomic_populations(overlap, density, ihelp)
    assert pytest.approx(ref.cpu(), rel=1e-7, abs=tol) == pop.sum(-1).cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_pop_shell(dtype: torch.dtype, name: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["mulliken_pop"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    pop = mulliken.get_shell_populations(overlap, density, ihelp)

    assert pytest.approx(ref.cpu(), abs=1e-5) == pop.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_pop_shell(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    density = pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["mulliken_pop"].to(**dd),
            sample2["mulliken_pop"].to(**dd),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, GFN1_XTB)
    pop = mulliken.get_shell_populations(overlap, density, ihelp)

    assert pytest.approx(ref.cpu(), abs=1e-5) == pop.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges(dtype: torch.dtype, name: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["mulliken_charges"].to(**dd)

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref.cpu(), abs=tol) == pop.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_charges(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    density = pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["mulliken_charges"].to(**dd),
            sample2["mulliken_charges"].to(**dd),
        ),
    )

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref.cpu(), abs=tol) == pop.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges_shell(dtype: torch.dtype, name: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    density = sample["density"].to(**dd)
    overlap = sample["overlap"].to(**dd)
    ref = sample["mulliken_charges_shell"].to(**dd)

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_shell(h0.get_occupation())

    pop = mulliken.get_mulliken_shell_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref.cpu(), abs=tol) == pop.cpu()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_charges_shell(dtype: torch.dtype, name1: str, name2: str):
    dd: DD = {"device": DEVICE, "dtype": dtype}
    tol = 1e-5

    sample1, sample2 = samples[name1], samples[name2]

    numbers = pack(
        (
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        )
    )
    density = pack(
        (
            sample1["density"].to(**dd),
            sample2["density"].to(**dd),
        )
    )
    overlap = pack(
        (
            sample1["overlap"].to(**dd),
            sample2["overlap"].to(**dd),
        )
    )
    ref = pack(
        (
            sample1["mulliken_charges_shell"].to(**dd),
            sample2["mulliken_charges_shell"].to(**dd),
        ),
    )

    par = get_param_module("gfn1", **dd)
    ihelp = IndexHelper.from_numbers(numbers, par)
    h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    n0 = ihelp.reduce_orbital_to_shell(h0.get_occupation())

    pop = mulliken.get_mulliken_shell_charges(overlap, density, ihelp, n0)
    assert pytest.approx(ref.cpu(), abs=tol) == pop.cpu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_orbital_charges_uhf(dtype: torch.dtype) -> None:
    """Alpha/beta populations are collected to charge/magnetization."""
    dd: DD = {"device": DEVICE, "dtype": dtype}

    overlap = torch.eye(3, **dd)
    density = torch.diag_embed(
        torch.tensor([[0.8, 0.2, 0.0], [0.3, 0.1, 0.0]], **dd)
    )
    n0 = torch.tensor([1.0, 1.0, 0.0], **dd)

    q = mulliken.get_mulliken_orbital_charges(overlap, density, n0, nspin=2)

    # total: n0 - p_alpha - p_beta; magnetization: p_beta - p_alpha
    ref = torch.tensor([[-0.1, 0.7, 0.0], [-0.5, -0.1, 0.0]], **dd)
    tol = 1e-3 if dtype == torch.float16 else 1e-6
    assert pytest.approx(ref.cpu(), abs=tol) == q.cpu()

    # restricted
    q = mulliken.get_mulliken_orbital_charges(overlap, density.sum(-3), n0)
    assert pytest.approx(ref[0].cpu(), abs=tol) == q.cpu()

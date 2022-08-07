"""
Test for Mulliken population analysis and charges.
Reference values obtained with xTB 6.5.1 and tblite 0.2.1.
"""

import pytest
import torch

from xtbml.exlibs.tbmalt import batch
from xtbml.basis.indexhelper import IndexHelper
from xtbml.param import GFN1_XTB as par
from xtbml.param import get_elem_angular
from xtbml.wavefunction import mulliken
from xtbml.xtb.h0 import Hamiltonian

from .samples import samples

sample_list = ["H2", "LiH", "SiH4"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_number_electrons(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["n_electrons"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    pop = mulliken.get_atomic_populations(overlap, density, ihelp)
    assert torch.allclose(ref, torch.sum(pop, dim=-1))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_number_electrons(dtype: torch.dtype, name1: str, name2: str):
    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    density = batch.pack(
        (
            sample1["density"].type(dtype),
            sample2["density"].type(dtype),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].type(dtype),
            sample2["overlap"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["n_electrons"].type(dtype),
            sample2["n_electrons"].type(dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

    pop = mulliken.get_atomic_populations(overlap, density, ihelp)
    assert torch.allclose(ref, torch.sum(pop, dim=-1))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_pop_shell(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["mulliken_pop"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    pop = mulliken.get_shell_populations(overlap, density, ihelp)

    assert torch.allclose(ref, pop, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_pop_shell(dtype: torch.dtype, name1: str, name2: str):
    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    density = batch.pack(
        (
            sample1["density"].type(dtype),
            sample2["density"].type(dtype),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].type(dtype),
            sample2["overlap"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["mulliken_pop"].type(dtype),
            sample2["mulliken_pop"].type(dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    pop = mulliken.get_shell_populations(overlap, density, ihelp)

    assert torch.allclose(ref, pop, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["mulliken_charges"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(overlap, density, ihelp, n0)
    assert torch.allclose(ref, pop, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_charges(dtype: torch.dtype, name1: str, name2: str):
    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"],
            sample2["positions"],
        )
    )
    density = batch.pack(
        (
            sample1["density"].type(dtype),
            sample2["density"].type(dtype),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].type(dtype),
            sample2["overlap"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["mulliken_charges"].type(dtype),
            sample2["mulliken_charges"].type(dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(overlap, density, ihelp, n0)
    assert torch.allclose(ref, pop, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges_shell(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["mulliken_charges_shell"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)
    n0 = ihelp.reduce_orbital_to_shell(h0.get_occupation())

    pop = mulliken.get_mulliken_shell_charges(overlap, density, ihelp, n0)
    assert torch.allclose(ref, pop, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch_charges_shell(dtype: torch.dtype, name1: str, name2: str):
    sample1, sample2 = samples[name1], samples[name2]

    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"],
            sample2["positions"],
        )
    )
    density = batch.pack(
        (
            sample1["density"].type(dtype),
            sample2["density"].type(dtype),
        )
    )
    overlap = batch.pack(
        (
            sample1["overlap"].type(dtype),
            sample2["overlap"].type(dtype),
        )
    )
    ref = batch.pack(
        (
            sample1["mulliken_charges_shell"].type(dtype),
            sample2["mulliken_charges_shell"].type(dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)
    n0 = ihelp.reduce_orbital_to_shell(h0.get_occupation())

    pop = mulliken.get_mulliken_shell_charges(overlap, density, ihelp, n0)
    assert torch.allclose(ref, pop, atol=1e-5)

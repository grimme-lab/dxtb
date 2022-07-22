"""
Test for Mulliken population analysis and charges.
Reference values obtained with xTB 6.5.1.
"""

import pytest
import torch

from xtbml.exlibs.tbmalt import batch
from xtbml.basis.indexhelper import IndexHelper
from xtbml.param import GFN1_XTB as par
from xtbml.param import get_element_angular
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

    ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))

    pop = mulliken.get_atomic_populations(ihelp, density, overlap)
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

    ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))

    pop = mulliken.get_atomic_populations(ihelp, density, overlap)
    assert torch.allclose(ref, torch.sum(pop, dim=-1))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_pop_shell(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["mulliken_pop"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
    pop = mulliken.get_shell_populations(ihelp, density, overlap)

    # manually reduce 1s and 2s of hydrogen as xtb does not resolve this
    if name == "H2":
        idx = torch.tensor([0, 0, 1, 1])
    elif name == "LiH":
        idx = torch.tensor([0, 1, 2, 2])
    elif name == "SiH4":
        idx = torch.tensor([0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    else:
        raise ValueError(f"Unknown sample name '{name}'.")

    mod_pop = torch.scatter_reduce(pop, -1, idx, reduce="sum")
    assert torch.allclose(ref, mod_pop, atol=1e-3)


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

    ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
    pop = mulliken.get_shell_populations(ihelp, density, overlap)

    # manually reduce 1s and 2s of hydrogen as xtb does not resolve this
    if name1 == "H2":
        idx1 = torch.tensor([0, 0, 1, 1])
    elif name1 == "LiH":
        idx1 = torch.tensor([0, 1, 2, 2])
    elif name1 == "SiH4":
        idx1 = torch.tensor([0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    else:
        raise ValueError(f"Unknown sample name '{name1}'.")

    if name2 == "H2":
        idx2 = torch.tensor([0, 0, 1, 1])
    elif name2 == "LiH":
        idx2 = torch.tensor([0, 1, 2, 2])
    elif name2 == "SiH4":
        idx2 = torch.tensor([0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    else:
        raise ValueError(f"Unknown sample name '{name2}'.")

    idx = [idx1, idx2]

    mod_pop = batch.pack(
        [
            torch.scatter_reduce(
                pop[_batch][pop[_batch].ne(0)], -1, idx[_batch], reduce="sum"
            )
            for _batch in range(pop.shape[0])
        ]
    )

    assert torch.allclose(ref, mod_pop, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single_charges(dtype: torch.dtype, name: str):
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    density = sample["density"].type(dtype)
    overlap = sample["overlap"].type(dtype)
    ref = sample["mulliken_charges"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(ihelp, density, overlap, n0)
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

    ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
    h0 = Hamiltonian(numbers, positions, par, ihelp)
    n0 = ihelp.reduce_orbital_to_atom(h0.get_occupation())

    pop = mulliken.get_mulliken_atomic_charges(ihelp, density, overlap, n0)
    assert torch.allclose(ref, pop, atol=1e-5)

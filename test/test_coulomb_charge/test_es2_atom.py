"""
Run tests for energy contribution from isotropic second-order
electrostatic energy (ES2).
"""

import pytest
import torch

from xtbml.basis import IndexHelper
from xtbml.coulomb import averaging_function
from xtbml.coulomb import secondorder as es2
from xtbml.param import GFN1_XTB, get_elem_param, get_elem_angular
from xtbml.typing import Tensor
from xtbml.utils import batch

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02", "SiH4_atom"]


def test_none() -> None:
    dummy = torch.tensor(0.0)
    par = GFN1_XTB.copy(deep=True)

    par.charge = None
    assert es2.new_es2(dummy, dummy, par) is None

    del par.charge
    assert es2.new_es2(dummy, dummy, par) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test ES2 for some samples from MB16_43."""

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)
    ref = sample["es2"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, positions, GFN1_XTB, shell_resolved=False)
    if es is None:
        assert False, es

    cache = es.get_cache(numbers, positions, ihelp)
    e = es.get_atom_energy(qat, ihelp, cache)
    assert torch.allclose(torch.sum(e, dim=-1), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].type(dtype),
            sample2["positions"].type(dtype),
        )
    )
    qat = batch.pack(
        (
            sample1["q"].type(dtype),
            sample2["q"].type(dtype),
        )
    )
    ref = torch.stack(
        [
            sample1["es2"].type(dtype),
            sample2["es2"].type(dtype),
        ],
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es2.new_es2(numbers, positions, GFN1_XTB, shell_resolved=False)
    if es is None:
        assert False

    cache = es.get_cache(numbers, positions, ihelp)
    e = es.get_atom_energy(qat, ihelp, cache)
    assert torch.allclose(torch.sum(e, dim=-1), ref)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_positions(name: str) -> None:
    dtype = torch.float64

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    # variable to be differentiated
    positions.requires_grad_(True)

    def func(positions: Tensor):
        es = es2.new_es2(numbers, positions, GFN1_XTB, shell_resolved=False)
        if es is None:
            assert False

        cache = es.get_cache(numbers, positions, ihelp)
        return es.get_atom_energy(qat, ihelp, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, positions)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name: str) -> None:
    dtype = torch.float64

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    if GFN1_XTB.charge is None:
        assert False

    hubbard = get_elem_param(
        torch.unique(numbers),
        GFN1_XTB.element,
        "gam",
        device=positions.device,
        dtype=positions.dtype,
    )
    gexp = torch.tensor(GFN1_XTB.charge.effective.gexp).type(dtype)
    average = averaging_function[GFN1_XTB.charge.effective.average]

    # variables to be differentiated
    gexp.requires_grad_(True)
    hubbard.requires_grad_(True)

    def func(gexp: Tensor, hubbard: Tensor):
        es = es2.ES2(
            positions=positions,
            hubbard=hubbard,
            average=average,
            gexp=gexp,
        )
        cache = es.get_cache(numbers, positions, ihelp)
        return es.get_atom_energy(qat, ihelp, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (gexp, hubbard))

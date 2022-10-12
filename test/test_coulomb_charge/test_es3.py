"""
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""

import pytest
import torch

from xtbml.basis import IndexHelper
from xtbml.coulomb import thirdorder as es3
from xtbml.utils import batch
from xtbml.param import GFN1_XTB, get_elem_param, get_elem_angular
from xtbml.typing import Tensor

from .samples import mb16_43

sample_list = ["01", "02", "SiH4"]


def test_none() -> None:
    dummy = torch.tensor(0.0)
    par = GFN1_XTB.copy(deep=True)

    par.thirdorder = None
    assert es3.new_es3(dummy, dummy, par) is None

    del par.thirdorder
    assert es3.new_es3(dummy, dummy, par) is None


def test_fail() -> None:
    dummy = torch.tensor(0.0)

    par = GFN1_XTB.copy(deep=True)
    if par.thirdorder is None:
        assert False

    with pytest.raises(NotImplementedError):
        par.thirdorder.shell = True
        es3.new_es3(dummy, dummy, par)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", sample_list)
def test_mb16_43(dtype: torch.dtype, name: str) -> None:
    """Test ES3 for some samples from MB16_43."""

    sample = mb16_43[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)
    ref = sample["es3"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es3.new_es3(numbers, positions, GFN1_XTB)
    if es is None:
        assert False

    cache = es.get_cache(numbers, positions, ihelp)  # cache is empty
    e = es.get_atom_energy(qat, ihelp, cache)
    assert torch.allclose(torch.sum(e, dim=-1), ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Test batched calculation of ES3."""
    sample1, sample2 = mb16_43[name1], mb16_43[name2]
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
            sample1["es3"].type(dtype),
            sample2["es3"].type(dtype),
        ],
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es3.new_es3(numbers, positions, GFN1_XTB)
    if es is None:
        assert False

    e = es.get_atom_energy(qat, ihelp, None)
    assert torch.allclose(torch.sum(e, dim=-1), ref)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name: str) -> None:
    """Test autograd for ES3 parameters."""
    dtype = torch.float64

    sample = mb16_43[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    qat = sample["q"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))

    hd = get_elem_param(
        torch.unique(numbers),
        GFN1_XTB.element,
        "gam3",
        device=numbers.device,
        dtype=dtype,
    )

    # variable to be differentiated
    hd.requires_grad_(True)

    def func(hubbard_derivs: Tensor):
        es = es3.ES3(positions, hubbard_derivs)
        return es.get_atom_energy(qat, ihelp, None)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, hd)

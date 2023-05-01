"""
Run tests for energy contribution from on-site third-order
electrostatic energy (ES3).
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import Tensor
from dxtb.basis import IndexHelper
from dxtb.coulomb import thirdorder as es3
from dxtb.param import GFN1_XTB, get_elem_angular, get_elem_param
from dxtb.utils import batch

from .samples import samples

sample_list = ["MB16_43_01", "MB16_43_02", "SiH4_atom"]


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    """Test ES3 for some samples from MB16_43."""
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
    qat = sample["q"].type(dtype)
    ref = sample["es3"].type(dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
    es = es3.new_es3(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(ihelp=ihelp)
    e = es.get_atom_energy(qat, cache)
    assert pytest.approx(torch.sum(e, dim=-1)) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", sample_list)
@pytest.mark.parametrize("name2", sample_list)
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Test batched calculation of ES3."""
    dd = {"dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        (
            sample1["numbers"],
            sample2["numbers"],
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
    es = es3.new_es3(numbers, GFN1_XTB, **dd)
    assert es is not None

    cache = es.get_cache(ihelp=ihelp)
    e = es.get_atom_energy(qat, cache)
    assert torch.allclose(torch.sum(e, dim=-1), ref)


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_param(name: str) -> None:
    """Test autograd for ES3 parameters."""
    dtype = torch.double
    dd = {"dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"]
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
        es = es3.ES3(hubbard_derivs, **dd)
        cache = es.get_cache(ihelp=ihelp)
        return es.get_atom_energy(qat, cache)

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, hd)

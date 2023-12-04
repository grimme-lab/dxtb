"""
Testing shape of multipole integrals.
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD
from dxtb.basis import Basis, IndexHelper
from dxtb.integral.driver.libcint import impls as intor
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch, is_basis_list

from .samples import samples

sample_list = ["H2", "HHe", "LiH", "Li2", "S2", "H2O", "SiH4"]
mp_ints = ["j", "jj"]  # dipole, quadrupole

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", sample_list)
@pytest.mark.parametrize("intstr", mp_ints)
def test_single(dtype: torch.dtype, intstr: str, name: str) -> None:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)

    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    wrapper = intor.LibcintWrapper(atombases, ihelp, spherical=True)
    i = intor.int1e(intstr, wrapper)

    mpdim = 3 ** len(intstr)
    assert i.shape == torch.Size((mpdim, ihelp.nao, ihelp.nao))


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name1", ["LiH"])
@pytest.mark.parametrize("name2", sample_list)
@pytest.mark.parametrize("intstr", mp_ints)
def test_batch(dtype: torch.dtype, name1: str, name2: str, intstr: str) -> None:
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = batch.pack(
        [
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        ]
    )
    positions = batch.pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    _ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, _ihelp, **dd)
    atombases = bas.create_dqc(positions)

    # batched IndexHelper does not yet work with LibcintWrapper
    ihelp = [
        IndexHelper.from_numbers(batch.deflate(number), get_elem_angular(par.element))
        for number in numbers
    ]

    wrappers = [
        intor.LibcintWrapper(ab, ihelp)
        for ab, ihelp in zip(atombases, ihelp)
        if is_basis_list(ab)
    ]
    i = batch.pack([intor.int1e(intstr, wrapper) for wrapper in wrappers])

    mpdim = 3 ** len(intstr)
    assert i.shape == torch.Size((2, mpdim, _ihelp.nao, _ihelp.nao))

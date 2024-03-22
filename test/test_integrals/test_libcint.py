"""
Test overlap build from integral container.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import integral as ints
from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

from .samples import samples

device = None


@pytest.mark.parametrize("name", ["H2"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype, name: str):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, **dd)

    i.setup_driver(positions)
    assert isinstance(i.driver, ints.driver.IntDriverLibcint)
    assert isinstance(i.driver.drv, ints.driver.libcint.LibcintWrapper)

    ################################################

    i.overlap = ints.Overlap(**dd)
    i.build_overlap(positions)

    o = i.overlap
    assert o is not None
    assert o.matrix is not None

    ################################################

    i.dipole = ints.Dipole(**dd)
    i.build_dipole(positions)

    d = i.dipole
    assert d is not None
    assert d.matrix is not None

    ################################################

    i.quadrupole = ints.Quadrupole(**dd)
    i.build_quadrupole(positions)

    q = i.quadrupole
    assert q is not None
    assert q.matrix is not None


@pytest.mark.parametrize("name1", ["H2"])
@pytest.mark.parametrize("name2", ["LiH"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype, name1: str, name2: str):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    sample1, sample2 = samples[name1], samples[name2]
    numbers = sample1["numbers"].to(device)
    positions = sample2["positions"].to(**dd)

    numbers = batch.pack(
        (
            sample1["numbers"].to(device),
            sample2["numbers"].to(device),
        )
    )
    positions = batch.pack(
        (
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        )
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    i = ints.Integrals(numbers, par, ihelp, **dd)

    i.setup_driver(positions)
    assert isinstance(i.driver, ints.driver.IntDriverLibcint)
    assert isinstance(i.driver.drv, list)
    assert isinstance(i.driver.drv[0], ints.driver.libcint.LibcintWrapper)
    assert isinstance(i.driver.drv[1], ints.driver.libcint.LibcintWrapper)

    ################################################

    i.overlap = ints.Overlap(**dd)
    i.build_overlap(positions)

    o = i.overlap
    assert o is not None
    assert o.matrix is not None

    ################################################

    i.dipole = ints.Dipole(**dd)
    i.build_dipole(positions)

    d = i.dipole
    assert d is not None
    assert d.matrix is not None

    ################################################

    i.quadrupole = ints.Quadrupole(**dd)
    i.build_quadrupole(positions)

    q = i.quadrupole
    assert q is not None
    assert q.matrix is not None

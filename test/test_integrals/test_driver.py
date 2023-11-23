"""
Test overlap build from integral container.
"""
from __future__ import annotations

import pytest
import torch

from dxtb import integral as ints
from dxtb._types import DD
from dxtb.basis import IndexHelper
from dxtb.integral.driver.labels import DRIVER_LIBCINT, DRIVER_PYTORCH
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch

device = None


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_single(dtype: torch.dtype):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([3, 1], device=device)
    positions = torch.zeros((2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    ipy = ints.Integrals(numbers, par, ihelp, driver=DRIVER_PYTORCH, **dd)
    ilc = ints.Integrals(numbers, par, ihelp, driver=DRIVER_LIBCINT, **dd)

    ipy.setup_driver(positions)
    assert isinstance(ipy.driver, ints.driver.IntDriverPytorch)
    ilc.setup_driver(positions)
    assert isinstance(ilc.driver, ints.driver.IntDriverLibcint)

    assert ipy.driver.is_latest(positions) is True
    assert ilc.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert ipy.driver.is_latest(positions) is False
    assert ilc.driver.is_latest(positions) is False


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_batch(dtype: torch.dtype):
    """Overlap matrix for monoatomic molecule should be unity."""
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([[3, 1], [1, 0]], device=device)
    positions = torch.zeros((2, 2, 3), **dd)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    ipy = ints.Integrals(numbers, par, ihelp, driver=DRIVER_PYTORCH, **dd)
    ilc = ints.Integrals(numbers, par, ihelp, driver=DRIVER_LIBCINT, **dd)

    ipy.setup_driver(positions)
    assert isinstance(ipy.driver, ints.driver.IntDriverPytorch)
    ilc.setup_driver(positions)
    assert isinstance(ilc.driver, ints.driver.IntDriverLibcint)

    assert ipy.driver.is_latest(positions) is True
    assert ilc.driver.is_latest(positions) is True

    # upon changing the positions, the driver should become outdated
    positions[0, 0] += 1e-4
    assert ipy.driver.is_latest(positions) is False
    assert ilc.driver.is_latest(positions) is False

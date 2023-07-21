"""
General tests for IndexHelper covering instantiation, changing dtypes, and
moving to devices.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.basis import IndexHelper

from ..utils import get_device_from_str


def test_fail_init_dtype() -> None:
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})
    with pytest.raises(ValueError):
        IndexHelper(
            ihelp.unique_angular.type(torch.float),
            ihelp.angular,
            ihelp.atom_to_unique,
            ihelp.ushells_to_unique,
            ihelp.ushells_per_unique,
            ihelp.shells_to_ushell,
            ihelp.shells_per_atom,
            ihelp.shell_index,
            ihelp.shells_to_atom,
            ihelp.orbitals_per_shell,
            ihelp.orbital_index,
            ihelp.orbitals_to_shell,
        )


@pytest.mark.cuda
def test_fail_init_device() -> None:
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})
    with pytest.raises(ValueError):
        IndexHelper(
            ihelp.unique_angular.to(get_device_from_str("cuda")),
            ihelp.angular,
            ihelp.atom_to_unique,
            ihelp.ushells_to_unique,
            ihelp.ushells_per_unique,
            ihelp.shells_to_ushell,
            ihelp.shells_per_atom,
            ihelp.shell_index,
            ihelp.shells_to_atom,
            ihelp.orbitals_per_shell,
            ihelp.orbital_index,
            ihelp.orbitals_to_shell,
        )


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_change_type(dtype: torch.dtype) -> None:
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})
    ihelp = ihelp.type(dtype)
    assert ihelp.dtype == dtype


def test_change_type_fail() -> None:
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})

    # trying to use setter
    with pytest.raises(AttributeError):
        ihelp.dtype = torch.float64

    # passing disallowed dtype
    with pytest.raises(ValueError):
        ihelp.type(torch.bool)


@pytest.mark.cuda
@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_change_device(device_str: str) -> None:
    device = get_device_from_str(device_str)
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]}).to(device)
    assert ihelp.device == device


def test_change_device_fail() -> None:
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})

    # trying to use setter
    with pytest.raises(AttributeError):
        ihelp.device = "cpu"


def test_cache() -> None:
    ihelp = IndexHelper.from_numbers(torch.tensor([1]), {1: [0]})

    # run a memoized function
    _ = ihelp.orbitals_to_shell_cart

    # get cache
    fcn = ihelp._orbitals_to_shell_cart  # pylint: disable=protected-access
    cache = fcn.get_cache()

    # cache should only have one entry
    assert len(cache) == 1

    # the key is created from the function name, so check if it is really there
    assert fcn.__name__ in tuple(*cache.keys())

    # clear cache and check if it is really empty
    ihelp.clear_cache()
    cache = fcn.get_cache()
    assert len(cache) == 0

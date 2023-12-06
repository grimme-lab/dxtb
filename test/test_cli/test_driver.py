"""
Test command line driver.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.cli import Driver, parser
from dxtb.timing import timer

from ..utils import coordfile


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_driver(dtype: torch.dtype) -> None:
    ref = torch.tensor(-1.0362714373390, dtype=dtype)

    dtype_str = "float32" if dtype == torch.float else "double"
    opts = f"--verbosity 0 --grad --chrg 0 --dtype {dtype_str} {coordfile}"
    args = parser().parse_args(opts.split())
    d = Driver(args)
    result = d.singlepoint()

    energy = result.total.sum(-1).detach()
    assert pytest.approx(energy) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_empty(dtype: torch.dtype) -> None:
    dtype_str = "float32" if dtype == torch.float else "double"
    opts = f"--verbosity 0 --exclude all --dtype {dtype_str} {coordfile}"
    args = parser().parse_args(opts.split())
    d = Driver(args)
    result = d.singlepoint()

    energy = result.total.sum(-1).detach()
    assert pytest.approx(0.0) == energy


# TODO: "Exclude all" and "grad" do not work together! Fix this!
@pytest.mark.xfail
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_empty_grad(dtype: torch.dtype) -> None:
    timer.disable()

    dtype_str = "float32" if dtype == torch.float else "double"
    opts = f"--verbosity 0 --grad --exclude all --dtype {dtype_str} {coordfile}"
    args = parser().parse_args(opts.split())
    d = Driver(args)
    result = d.singlepoint()

    energy = result.total.sum(-1).detach()

    timer.enable()
    assert pytest.approx(0.0) == energy


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_empty_interactions(dtype: torch.dtype) -> None:
    dtype_str = "float32" if dtype == torch.float else "double"
    opts = f"--verbosity 0 --grad --exclude es2 es3 --dtype {dtype_str} {coordfile}"
    args = parser().parse_args(opts.split())
    d = Driver(args)
    result = d.singlepoint()

    energy = result.total.sum(-1).detach()
    assert pytest.approx(-1.036271443341644) == energy


def test_fail() -> None:
    status = timer._enabled
    if status is True:
        timer.disable()

    args = parser().parse_args([str(coordfile)])

    with pytest.raises(ValueError):
        setattr(args, "method", "xtb")
        Driver(args).singlepoint()

    with pytest.raises(NotImplementedError):
        setattr(args, "method", "gfn2")
        Driver(args).singlepoint()

    with pytest.raises(ValueError):
        setattr(args, "method", "gfn1")
        setattr(args, "guess", "random")
        Driver(args).singlepoint()

    if status is True:
        timer.enable()

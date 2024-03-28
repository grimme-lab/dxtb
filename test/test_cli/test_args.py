"""
Test command line options.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
import torch

from dxtb.cli import parser
from dxtb.constants import defaults

from ..utils import coordfile as dummy


def test_defaults() -> None:
    args = parser().parse_args([str(dummy)])

    assert isinstance(args.chrg, int)
    assert args.chrg == defaults.CHRG

    # assert isinstance(args.spin, int)
    assert args.spin == defaults.SPIN

    assert isinstance(args.verbosity, int)
    assert args.verbosity == defaults.VERBOSITY

    assert isinstance(args.maxiter, int)
    assert args.maxiter == defaults.MAXITER

    assert isinstance(args.guess, str)
    assert args.guess == defaults.GUESS

    assert isinstance(args.fermi_etemp, float)
    assert args.fermi_etemp == defaults.FERMI_ETEMP

    assert isinstance(args.fermi_maxiter, int)
    assert args.fermi_maxiter == defaults.FERMI_MAXITER

    assert isinstance(args.fermi_partition, str)
    assert args.fermi_partition == defaults.FERMI_PARTITION

    # integral settings
    assert isinstance(args.int_cutoff, (float, int))
    assert args.int_cutoff == defaults.INTCUTOFF

    assert isinstance(args.int_driver, str)
    assert args.int_driver == defaults.INTDRIVER

    assert isinstance(args.int_level, int)
    assert args.int_level == defaults.INTLEVEL

    assert isinstance(args.int_uplo, str)
    assert args.int_uplo == defaults.INTUPLO


@pytest.mark.parametrize(
    "option", ["chrg", "spin", "maxiter", "verbosity", "fermi_maxiter"]
)
def test_int(option: str) -> None:
    value = 1
    args = parser().parse_args(f"--{option} {value}".split())

    assert isinstance(getattr(args, option), int)
    assert getattr(args, option) == value


@pytest.mark.parametrize("option", ["chrg", "spin"])
def test_int_multiple(option: str) -> None:
    args = parser().parse_args(f"--{option} 1 1".split())

    assert isinstance(getattr(args, option), list)
    assert getattr(args, option) == [1, 1]


@pytest.mark.parametrize("option", ["fermi_etemp"])
def test_float(option: str) -> None:
    value = 200.0
    args = parser().parse_args(f"--{option} {value}".split())

    assert isinstance(getattr(args, option), float)
    assert getattr(args, option) == value


@pytest.mark.parametrize("value", ["float16", "float32", "float64"])
def test_torch_dtype(value: str) -> None:
    option = "dtype"
    args = parser().parse_args(f"--{option} {value}".split())

    ref = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    assert isinstance(getattr(args, option), torch.dtype)
    assert getattr(args, option) == ref[value]


def test_dir() -> None:
    option = "dir"
    args = parser().parse_args(f"--{option} {Path(__file__).parent}".split())

    d = getattr(args, option)
    assert isinstance(d, list)

    d = d[0]
    assert isinstance(d, str)
    assert Path(d).is_dir()


@pytest.mark.parametrize("value", ["cpu", "cpu:0"])
def test_torch_device_cpu(value: str) -> None:
    option = "device"
    args = parser().parse_args(f"--{option} {value}".split())

    ref = {
        "cpu": torch.device("cpu"),
        "cpu:0": torch.device("cpu:0"),
    }

    assert isinstance(getattr(args, option), torch.device)
    assert getattr(args, option) == ref[value]


@pytest.mark.cuda
@pytest.mark.parametrize("value", ["cuda", "cuda:0"])
def test_torch_device(value: str) -> None:
    option = "device"
    args = parser().parse_args(f"--{option} {value}".split())

    ref = {
        "cuda": torch.device("cuda", index=torch.cuda.current_device()),
        "cuda:0": torch.device("cuda:0"),
    }

    assert isinstance(getattr(args, option), torch.device)
    assert getattr(args, option) == ref[value]


def test_fail_type():
    """Test behavior if wrong data type is given."""

    f = StringIO()
    with redirect_stderr(f):
        with pytest.raises(SystemExit):
            parser().parse_args("--chrg 2.0".split())

        with pytest.raises(SystemExit):
            parser().parse_args("--method 2.0".split())

        with pytest.raises(SystemExit):
            parser().parse_args("--dtype int64".split())

        with pytest.raises(SystemExit):
            parser().parse_args("--device laptop".split())

        with pytest.raises(SystemExit):
            parser().parse_args("--device laptop:0".split())

        with pytest.raises(SystemExit):
            parser().parse_args("--device cuda:zero".split())


def test_fail_value():
    """Test behavior if disallowed values are given."""
    p = parser()

    f = StringIO()
    with redirect_stderr(f):
        with pytest.raises(SystemExit):
            p.parse_args("--spin -1".split())

        with pytest.raises(SystemExit):
            p.parse_args("--spin -1 -1".split())

        with pytest.raises(SystemExit):
            p.parse_args("--method dftb3".split())

        with pytest.raises(SystemExit):
            p.parse_args("--guess zero".split())

        with pytest.raises(SystemExit):
            p.parse_args("--etemp -1.0".split())

        with pytest.raises(SystemExit):
            p.parse_args(["non-existing-coord-file"])

        with pytest.raises(SystemExit):
            directory = Path(__file__).parent.resolve()
            p.parse_args([str(directory)])

        with pytest.raises(SystemExit):
            p.parse_args("--dir non-existing-dir".split())

        with pytest.raises(SystemExit):
            p.parse_args(f"--dir {dummy}".split())

    with redirect_stdout(f):
        with pytest.raises(SystemExit):
            p.parse_args(["--help"])

"""
Test command line options.
"""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import pytest

from dxtb.cli import argparser
from dxtb.constants import defaults


def test_defaults() -> None:
    dummy = Path(Path(__file__).parent.parent, "test_singlepoint/H2/coord").resolve()
    args = argparser().parse_args([str(dummy)])

    assert isinstance(args.chrg, int)
    assert args.chrg == defaults.CHRG

    assert args.spin is None
    assert args.spin == defaults.SPIN

    assert isinstance(args.verbosity, int)
    assert args.verbosity == defaults.VERBOSITY

    assert isinstance(args.maxiter, int)
    assert args.maxiter == defaults.MAXITER

    assert isinstance(args.etemp, float)
    assert args.etemp == defaults.ETEMP

    assert isinstance(args.guess, str)
    assert args.guess == defaults.GUESS

    assert isinstance(args.fermi_maxiter, int)
    assert args.fermi_maxiter == defaults.FERMI_MAXITER

    assert isinstance(args.fermi_energy_partition, str)
    assert args.fermi_energy_partition == defaults.FERMI_FENERGY_PARTITION


@pytest.mark.parametrize(
    "option", ["chrg", "spin", "maxiter", "verbosity", "fermi_maxiter"]
)
def test_int(option: str) -> None:
    value = 1
    args = argparser().parse_args(f"--{option} {value}".split())

    assert isinstance(getattr(args, option), int)
    assert getattr(args, option) == value


@pytest.mark.parametrize("option", ["etemp"])
def test_float(option: str) -> None:
    value = 200.0
    args = argparser().parse_args(f"--{option} {value}".split())

    assert isinstance(getattr(args, option), float)
    assert getattr(args, option) == value


def test_fail_type():
    """Test behavior if wrong data type is given."""
    parser = argparser()

    f = StringIO()
    with redirect_stderr(f):
        with pytest.raises(SystemExit):
            parser.parse_args("--chrg 2.0".split())

        with pytest.raises(SystemExit):
            parser.parse_args("--method 2.0".split())


def test_fail_value():
    """Test behavior if disallowed values are given."""
    parser = argparser()

    f = StringIO()
    with redirect_stderr(f):
        with pytest.raises(SystemExit):
            parser.parse_args("--spin -1".split())

        with pytest.raises(SystemExit):
            parser.parse_args("--method dftb3".split())

        with pytest.raises(SystemExit):
            parser.parse_args("--guess zero".split())

        with pytest.raises(SystemExit):
            parser.parse_args(["non-existing-coord-file"])

        with pytest.raises(SystemExit):
            directory = Path(__file__).parent.resolve()
            parser.parse_args([str(directory)])

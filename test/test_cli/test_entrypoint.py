"""
Test for the actual command line entrypint function.
"""

import pytest

from dxtb import __version__
from dxtb.cli import console_entry_point

from ..utils import coordfile


def test_version(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        ret = console_entry_point(["--version"])
        assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert out == f"dxtb {__version__}\n"


def test_entrypoint(capsys: pytest.CaptureFixture) -> None:
    ret = console_entry_point([str(coordfile)])
    assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert len(out) != 0

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
    assert out == f"{__version__}\n"


def test_no_file(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.INFO)

    with pytest.raises(SystemExit):
        ret = console_entry_point([])
        assert ret == 1

    # empty because message goes to logs
    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert "No coordinate file given." in caplog.text


def test_entrypoint(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture
) -> None:
    ret = console_entry_point([str(coordfile)])
    assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert len(caplog.text) != 0

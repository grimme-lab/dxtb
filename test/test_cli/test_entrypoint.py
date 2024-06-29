# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test for the actual command line entrypint function.
"""

import pytest

from dxtb import __version__
from dxtb._src.cli import console_entry_point

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
    # avoid pollution from previous tests
    caplog.clear()

    ret = console_entry_point([str(coordfile), "--verbosity", "0"])
    assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert out == ""
    assert len(caplog.text) == 0

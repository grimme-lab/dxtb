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
Test integral configuration.
"""

from __future__ import annotations

import pytest

from dxtb._src.constants import defaults, labels
from dxtb._src.exlibs.available import has_libcint
from dxtb.config import ConfigIntegrals as Cfg


def test_default() -> None:
    cfg = Cfg()
    assert cfg.cutoff == defaults.INTCUTOFF
    assert cfg.level == defaults.INTLEVEL
    assert cfg.uplo == defaults.INTUPLO


def test_default_driver() -> None:
    cfg = Cfg()

    if has_libcint is True:
        assert cfg.driver == defaults.INTDRIVER
    else:
        assert cfg.driver == labels.INTDRIVER_ANALYTICAL


def test_driver_pytorch() -> None:
    cfg = Cfg(driver=labels.INTDRIVER_ANALYTICAL_STRS[0])
    assert cfg.driver == labels.INTDRIVER_ANALYTICAL

    cfg = Cfg(driver=labels.INTDRIVER_AUTOGRAD_STRS[0])
    assert cfg.driver == labels.INTDRIVER_AUTOGRAD

    cfg = Cfg(driver=labels.INTDRIVER_LEGACY_STRS[0])
    assert cfg.driver == labels.INTDRIVER_LEGACY


def test_driver_libcint() -> None:

    if has_libcint is False:
        with pytest.raises(ValueError):
            Cfg(driver=labels.INTDRIVER_LIBCINT_STRS[0])
    else:
        cfg = Cfg(driver=labels.INTDRIVER_LIBCINT_STRS[0])
        assert cfg.driver == labels.INTDRIVER_LIBCINT


def test_fail_driver() -> None:
    with pytest.raises(ValueError):
        Cfg(driver=-999)

    with pytest.raises(ValueError):
        Cfg(driver="-999")

    with pytest.raises(TypeError):
        Cfg(driver=1.0)  # type: ignore


def test_fail_level() -> None:
    with pytest.raises(TypeError):
        Cfg(level="overlap")  # type: ignore


def test_fail_uplo() -> None:
    with pytest.raises(ValueError):
        Cfg(uplo="symmetric")  # type: ignore

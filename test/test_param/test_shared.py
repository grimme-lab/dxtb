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
Test the guard for parametrizations shared between tests.
"""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from dxtb import ParamModule

from .. import utils
from ..utils import check_param_modules, get_param_module


@pytest.fixture(autouse=True)
def _private_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the deliberately misused modules out of the real cache."""
    monkeypatch.setattr(utils, "_PARAM_MODULES", {})
    monkeypatch.setattr(utils, "_PARAM_MODULES_USED", set())


def _float_params(par: ParamModule) -> list[torch.nn.Parameter]:
    return [p for p in par.parameters() if p.is_floating_point()]


def _element(par: ParamModule) -> torch.nn.Module:
    return par.get_submodule("parameter_tree.element")


def _misuse_check(misuse: Callable[[ParamModule], object]) -> None:
    par = get_param_module("gfn1", dtype=torch.float32)
    misuse(par)

    with pytest.raises(AssertionError, match="gfn1"):
        check_param_modules()

    # The modified module is discarded and rebuilt on next use.
    fresh = get_param_module("gfn1", dtype=torch.float32)
    assert fresh is not par
    assert fresh.dtype == torch.float32
    assert fresh.device == torch.device("cpu")


def test_unchanged() -> None:
    par = get_param_module("gfn1", dtype=torch.float32)
    check_param_modules()
    assert get_param_module("gfn1", dtype=torch.float32) is par


def test_requires_grad() -> None:
    _misuse_check(lambda par: _float_params(par)[0].requires_grad_(True))


def test_value() -> None:
    _misuse_check(lambda par: _float_params(par)[0].data.add_(1.0))


def test_dtype() -> None:
    _misuse_check(lambda par: par.to(torch.float64))


def test_dtype_partial() -> None:
    # Promotion in the value comparison would hide a partial conversion.
    _misuse_check(lambda par: _element(par).to(torch.float64))


@pytest.mark.cuda
def test_device() -> None:
    _misuse_check(lambda par: par.to("cuda"))


@pytest.mark.cuda
def test_device_partial() -> None:
    _misuse_check(lambda par: _element(par).to("cuda"))

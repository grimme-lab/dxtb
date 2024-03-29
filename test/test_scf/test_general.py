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
General tests for SCF setup.
"""

from __future__ import annotations

import pytest
import torch

from dxtb.integral import IntegralMatrices
from dxtb.scf.iterator import SelfConsistentField


def test_properties() -> None:
    d = torch.randn((3, 3))  # dummy

    ints = IntegralMatrices()
    with pytest.raises(RuntimeError):
        SelfConsistentField(d, d, d, d, d, d, integrals=ints)  # type: ignore

    ints.hcore = torch.randn((3, 3))
    with pytest.raises(RuntimeError):
        SelfConsistentField(d, d, d, d, d, d, integrals=ints)  # type: ignore

    ints.overlap = torch.randn((3, 3))
    scf = SelfConsistentField(d, d, d, d, d, d, integrals=ints)  # type: ignore
    assert scf.shape == d.shape
    assert scf.device == d.device
    assert scf.dtype == d.dtype

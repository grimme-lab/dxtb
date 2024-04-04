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
Testing dispersion gradient (autodiff).
"""

from __future__ import annotations

import pytest
import torch

from dxtb.components.classicals.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD

from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "PbH4-BiH3"]

device = None


@pytest.mark.grad
@pytest.mark.parametrize("name", sample_list)
def test_grad_fail(name: str) -> None:
    dtype = torch.double
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)
    energy = disp.get_energy(positions, cache)

    with pytest.raises(RuntimeError):
        disp.get_gradient(energy, positions)

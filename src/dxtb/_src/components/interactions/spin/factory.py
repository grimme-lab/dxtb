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
Spin Polarisation: Factory
===========================

A factory function to create instances of the
:class.components.SpinPolarisation class.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import DD, Tensor, get_default_dtype
from dxtb._src.typing.exceptions import DeviceError, DtypeError

from .constants import _load_spin_constants
from .spinpolarisation import SpinPolarisation

__all__ = ["new_spinpolarisation"]


def new_spinpolarisation(
    numbers: Tensor,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> SpinPolarisation | None:

    dd: DD = {
        "device": device,
        "dtype": dtype if dtype is not None else get_default_dtype(),
    }

    spinconst = _load_spin_constants(**dd)[numbers]

    return SpinPolarisation(
        spinconst,
        **dd,
    )

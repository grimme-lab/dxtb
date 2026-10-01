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
Spin Polarization: Constants
============================

Element-specific spin constants for the on-site spin-polarization
interaction. The values (in Hartree) are taken from tblite 0.7.0
(``src/tblite/data/spin.f90``) and cover the elements H to Rn.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import tomli as toml
import torch

from dxtb._src.typing import (
    DD,
    Tensor,
    get_default_device,
    get_default_dtype,
)

__all__ = ["ANGULAR_PAIR_INDEX", "MAX_SPIN_ELEMENT", "load_spin_constants"]


MAX_SPIN_ELEMENT = 86
"""Largest atomic number covered by the spin constants."""

ANGULAR_PAIR_INDEX = (
    (0, 1, 3),
    (1, 2, 4),
    (3, 4, 5),
)
"""
Column of the spin constants table (ss, sp, pp, sd, pd, dd) for a pair of
angular momenta (s, p, d).
"""


@lru_cache(maxsize=1)
def _read_spin_constants() -> list[list[float]]:
    path = Path(__file__).parent / "data" / "spin-constants.toml"
    with open(path, "rb") as fd:
        return toml.load(fd)["constants"]


def load_spin_constants(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Load the spin constants of all elements.

    Parameters
    ----------
    device : torch.device | None, optional
        Device to store the tensor on. Defaults to the default device.
    dtype : torch.dtype | None, optional
        Floating point data type. Defaults to the default dtype.

    Returns
    -------
    Tensor
        Spin constants of shape ``(MAX_SPIN_ELEMENT + 1, 6)``. The first row
        corresponds to padding (``Z=0``) and is zero, so that the table can
        directly be indexed with atomic numbers.
    """
    dd: DD = {
        "device": get_default_device() if device is None else device,
        "dtype": get_default_dtype() if dtype is None else dtype,
    }
    table = torch.tensor(_read_spin_constants(), **dd)
    return torch.cat((torch.zeros((1, table.shape[-1]), **dd), table), dim=0)

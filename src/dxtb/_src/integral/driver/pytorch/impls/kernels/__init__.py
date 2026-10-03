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
Integral algorithms
===================

Registry of the interchangeable 1D kernels (``compute_1d`` contract) of the
pair builder, which computes all integrals of the PyTorch driver, selected by
name through the ``int_algorithm`` configuration option.
"""

from __future__ import annotations

from typing import Callable

from dxtb._src.constants.labels import (
    INTALGORITHM_CHOICES,
    INTALGORITHM_DEFAULT,
)
from dxtb._src.typing import Tensor

__all__ = ["ALGORITHMS", "DEFAULT_ALGORITHM", "get_kernel"]

ALGORITHMS = INTALGORITHM_CHOICES
"""Names of the available kernels."""

DEFAULT_ALGORITHM = INTALGORITHM_DEFAULT
"""Kernel used when none is requested."""


def get_kernel(name: str) -> Callable[..., Tensor]:
    """
    Return the ``compute_1d``-contract kernel registered under ``name``.

    Raises
    ------
    ValueError
        Unknown algorithm name.
    """
    # pylint: disable=import-outside-toplevel
    name = name.casefold()

    if name == "md":
        from .md import compute_1d_md_hermite as kernel
    elif name == "os":
        from .os import compute_1d_os as kernel
    else:
        raise ValueError(
            f"Unknown integral algorithm '{name}'. "
            f"Choose one of: {', '.join(ALGORITHMS)}."
        )

    return kernel

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
Broyden mixing
==============
"""

from __future__ import annotations

from dxtb._src.typing import Any

from .base import Mixer

__all__ = ["Broyden"]


default_opts = {
    "method": "broyden1",
    "alpha": -0.5,
    "f_tol": 1.0e-6,
    "x_tol": 1.0e-6,
    "f_rtol": float("inf"),
    "x_rtol": float("inf"),
    "maxiter": 50,
    "verbose": False,
    "line_search": False,
}


class Broyden(Mixer):
    """
    Broyden mixing using xitorch.
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        if options is not None:
            default_opts.update(options)
        super().__init__(default_opts)

    def iter(self):
        raise NotImplementedError

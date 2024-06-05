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
Driver: PyTorch
===============

Collection of PyTorch-based integral drivers.
"""

from __future__ import annotations

from .base_driver import BaseIntDriverPytorch
from .impls import (
    OverlapAG_V1,
    OverlapAG_V2,
    OverlapFunction,
    overlap,
    overlap_gradient,
)

__all__ = [
    "IntDriverPytorch",
    "IntDriverPytorchNoAnalytical",
    "IntDriverPytorchLegacy",
]


class IntDriverPytorch(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver.

    The overlap evaluation function implements a custom backward function
    containing the analytical overlap derivative.

    Note
    ----
    Currently, only the overlap integral is implemented.
    """

    eval_ovlp: OverlapFunction
    """Function for overlap calculation."""

    eval_ovlp_grad: OverlapFunction
    """Function for overlap gradient calculation."""

    def setup_eval_funcs(self) -> None:
        # pylint: disable=import-outside-toplevel
        from tad_mctc._version import __tversion__

        OverlapAG = OverlapAG_V1 if __tversion__ < (2, 0, 0) else OverlapAG_V2
        self.eval_ovlp = OverlapAG.apply  # type: ignore
        self.eval_ovlp_grad = overlap_gradient


class IntDriverPytorchNoAnalytical(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver without analytical derivatives.

    Note
    ----
    Currently, only the overlap integral is implemented.
    """

    def setup_eval_funcs(self) -> None:
        self.eval_ovlp = overlap
        self.eval_ovlp_grad = overlap_gradient


class IntDriverPytorchLegacy(BaseIntDriverPytorch):
    """
    PyTorch-based integral driver with old loop-based version of the full
    matrix build. The newer version partially vectorizes over the centers of
    the orbitals (unique pair algorithm).

    Note
    ----
    Currently, only the overlap integral is implemented.
    """

    def setup_eval_funcs(self) -> None:
        # pylint: disable=import-outside-toplevel
        from .impls.overlap_legacy import overlap_gradient_legacy, overlap_legacy

        self.eval_ovlp = overlap_legacy
        self.eval_ovlp_grad = overlap_gradient_legacy

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

from abc import abstractmethod

import torch

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Any, Tensor

from ...base import IntDriver
from .base import PytorchImplementation
from .impls import (
    OverlapAG_V1,
    OverlapAG_V2,
    OverlapFunction,
    overlap,
    overlap_gradient,
)

__all__ = [
    "BaseIntDriverPytorch",
    "IntDriverPytorch",
    "IntDriverPytorchNoAnalytical",
    "IntDriverPytorchLegacy",
]


class BaseIntDriverPytorch(PytorchImplementation, IntDriver):
    """
    PyTorch-based integral driver.

    Note
    ----
    Currently, only the overlap integral is implemented.
    """

    eval_ovlp: OverlapFunction
    """Function for overlap calculation."""

    eval_ovlp_grad: OverlapFunction
    """Function for overlap gradient calculation."""

    def setup(self, positions: Tensor, **kwargs: Any) -> None:
        """
        Run the `libcint`-specific driver setup.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        """
        if self.ihelp.batch_mode == 0:
            # setup `Basis` class if not already done
            if self._basis is None:
                self.basis = Basis(
                    torch.unique(self.numbers),
                    self.par,
                    self.ihelp,
                    device=self.device,
                    dtype=self.dtype,
                )

            self._positions_single = positions
        else:

            self._positions_batch: list[Tensor] = []
            self._basis_batch: list[Basis] = []
            self._ihelp_batch: list[IndexHelper] = []
            for _batch in range(self.numbers.shape[0]):
                # POSITIONS
                if self.ihelp.batch_mode == 1:
                    # pylint: disable=import-outside-toplevel
                    from tad_mctc.batch import deflate

                    mask = kwargs.pop("mask", None)
                    if mask is not None:
                        pos = torch.masked_select(
                            positions[_batch],
                            mask[_batch],
                        ).reshape((-1, 3))
                    else:
                        pos = deflate(positions[_batch])

                    nums = deflate(self.numbers[_batch])

                elif self.ihelp.batch_mode == 2:
                    pos = positions[_batch]
                    nums = self.numbers[_batch]

                else:
                    raise ValueError(
                        f"Unknown batch mode '{self.ihelp.batch_mode}'."
                    )

                self._positions_batch.append(pos)

                # INDEXHELPER
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `calc_overlap`
                ihelp = IndexHelper.from_numbers(nums, self.par)

                self._ihelp_batch.append(ihelp)

                # BASIS
                bas = Basis(
                    torch.unique(nums),
                    self.par,
                    ihelp,
                    dtype=self.dtype,
                    device=self.device,
                )

                self._basis_batch.append(bas)

        self.setup_eval_funcs()

        # setting positions signals successful setup; save current positions to
        # catch new positions and run the required re-setup of the driver
        self._positions = positions.detach().clone()

    @abstractmethod
    def setup_eval_funcs(self) -> None:
        """
        Specification of the overlap (gradient) evaluation functions
        (`eval_ovlp` and `eval_ovlp_grad`).
        """


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
        from .impls.overlap_legacy import (
            overlap_gradient_legacy,
            overlap_legacy,
        )

        self.eval_ovlp = overlap_legacy
        self.eval_ovlp_grad = overlap_gradient_legacy

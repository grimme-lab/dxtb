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
Base class for PyTorch-based integrals.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from ...._types import Any, Tensor
from ....basis import Basis, IndexHelper
from ....constants import labels
from ...base import BaseIntegralImplementation, IntDriver
from .impls import OverlapFunction


class PytorchImplementation:
    """
    Simple label for `PyTorch`-based integral implementations.
    """

    family: int = labels.INTDRIVER_ANALYTICAL
    """Label for integral implementation family"""


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
            Cartesian coordinates of all atoms in the system (nat, 3).
        """
        if not self.ihelp.batched:
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
            from tad_mctc.batch import deflate

            self._positions_batch: list[Tensor] = []
            self._basis_batch: list[Basis] = []
            self._ihelp_batch: list[IndexHelper] = []
            for _batch in range(self.numbers.shape[0]):
                # POSITIONS
                mask = kwargs.pop("mask", None)
                if mask is not None:
                    pos = torch.masked_select(
                        positions[_batch],
                        mask[_batch],
                    ).reshape((-1, 3))
                else:
                    pos = deflate(positions[_batch])

                self._positions_batch.append(pos)

                # INDEXHELPER
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `calc_overlap`
                nums = deflate(self.numbers[_batch])
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


class IntegralImplementationPytorch(
    PytorchImplementation,
    BaseIntegralImplementation,
):
    """PyTorch-based integral implementation"""

    def checks(self, driver: BaseIntDriverPytorch) -> None:
        """
        Check if the type of integral driver is correct.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.
        """
        super().checks(driver)

        if not isinstance(driver, BaseIntDriverPytorch):
            raise RuntimeError("Wrong integral driver selected.")

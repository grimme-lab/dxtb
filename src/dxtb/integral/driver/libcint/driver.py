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
Integral driver for `libcint`.
"""

from __future__ import annotations

from dxtb.timing import timer

from ...._types import Tensor
from ....basis import Basis, IndexHelper
from ....constants import labels
from ....utils import is_basis_list
from ...base import IntDriver
from .impls import LibcintWrapper

__all__ = ["IntDriverLibcint"]


class IntDriverLibcint(IntDriver):
    """
    Implementation of `libcint`-based integral driver.
    """

    family: int = labels.INTDRIVER_LIBCINT
    """Label for integral implementation family"""

    def setup(self, positions: Tensor, **kwargs) -> None:
        """
        Run the `libcint`-specific driver setup.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        """
        timer.start("Setup Overlap")
        # setup `Basis` class if not already done
        if self._basis is None:
            self.basis = Basis(self.numbers, self.par, self.ihelp, **self.dd)

        timer.start("OverlapDQC", parent_uid="Setup Overlap")
        # create atomic basis set in libcint format
        mask = kwargs.pop("mask", None)
        atombases = self.basis.create_dqc(positions, mask=mask)
        timer.stop("OverlapDQC")

        if self.ihelp.batch_mode > 0:

            timer.start("IndexHelper", parent_uid="Setup Overlap")
            # integrals do not work with a batched IndexHelper
            if self.ihelp.batch_mode == 1:
                # pylint: disable=import-outside-toplevel
                from tad_mctc.batch import deflate

                _ihelp = [
                    IndexHelper.from_numbers(deflate(number), self.par)
                    for number in self.numbers
                ]
            elif self.ihelp.batch_mode == 2:
                _ihelp = [
                    IndexHelper.from_numbers(number, self.par)
                    for number in self.numbers
                ]
            timer.stop("IndexHelper")

            timer.start("LibcintWrapper", parent_uid="Setup Overlap")
            assert isinstance(atombases, list)
            self.drv = [
                LibcintWrapper(ab, ihelp)
                for ab, ihelp in zip(atombases, _ihelp)
                if is_basis_list(ab)
            ]
            timer.stop("LibcintWrapper")
        else:
            timer.start("LibcintWrapper", parent_uid="Setup Overlap")
            assert is_basis_list(atombases)
            self.drv = LibcintWrapper(atombases, self.ihelp)
            timer.stop("LibcintWrapper")

        # setting positions signals successful setup; save current positions to
        # catch new positions and run the required re-setup of the driver
        self._positions = positions.detach().clone()
        timer.stop("Setup Overlap")

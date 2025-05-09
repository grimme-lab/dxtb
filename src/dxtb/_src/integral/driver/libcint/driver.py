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
Driver: Libcint
===============

Base class for a `libcint`-based integral implementation
Calculation and modification of multipole integrals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.typing import Tensor, TypeAlias, overload
from dxtb._src.utils import is_basis_list

from ...base import IntDriver
from .base import LibcintImplementation

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint

    ListAtomCGTOBasis: TypeAlias = list[libcint.AtomCGTOBasis]


__all__ = ["BaseIntDriverLibcint", "IntDriverLibcint"]


class BaseIntDriverLibcint(LibcintImplementation, IntDriver):
    """
    Implementation of `libcint`-based integral driver.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.drv = None

    def setup(self, positions: Tensor, **kwargs) -> None:
        """
        Run the `libcint`-specific driver setup.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        """

        # setup `Basis` class if not already done
        if self._basis is None:
            self.basis = Basis(self.numbers, self.par, self.ihelp, **self.dd)

        # create atomic basis set in libcint format
        mask = kwargs.pop("mask", None)
        atombases = self.basis.create_libcint(positions, mask=mask)
        self.drv = self._get_driver(atombases)

        # setting positions signals successful setup; save current positions to
        # catch new positions and run the required re-setup of the driver
        self._positions = positions.detach().clone()

    @overload
    def _get_driver(
        self, atombases: ListAtomCGTOBasis
    ) -> libcint.LibcintWrapper: ...

    @overload
    def _get_driver(
        self,
        atombases: list[ListAtomCGTOBasis],
    ) -> list[libcint.LibcintWrapper]: ...

    def _get_driver(
        self,
        atombases: ListAtomCGTOBasis | list[ListAtomCGTOBasis],
    ) -> libcint.LibcintWrapper | list[libcint.LibcintWrapper]:
        """
        Wrapper for getting the `libcint` driver in single or batched mode.
        """
        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        if self.ihelp.batch_mode == 0:
            assert is_basis_list(atombases)
            return libcint.LibcintWrapper(atombases, self.ihelp)

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
        else:
            raise ValueError(f"Unknown batch mode '{self.ihelp.batch_mode}'.")

        assert isinstance(atombases, list)
        return [
            libcint.LibcintWrapper(ab, ihelp)
            for ab, ihelp in zip(atombases, _ihelp)
            if is_basis_list(ab)
        ]


class IntDriverLibcint(BaseIntDriverLibcint):
    """
    Implementation of ``libcint``-based integral driver.
    """

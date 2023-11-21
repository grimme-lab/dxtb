"""
Integral driver for `libcint`.
"""
from __future__ import annotations

from ...._types import Tensor
from ....basis import Basis, IndexHelper
from ....utils import is_basis_list
from ...base import IntDriver
from .impls import LibcintWrapper


class IntDriverLibcint(IntDriver):
    """
    Implementation of `libcint`-based integral driver.
    """

    def setup(self, positions: Tensor) -> None:
        """
        Run the `libcint`-specific driver setup.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system (nat, 3).
        """
        # setup `Basis` class if not already done
        if self._basis is None:
            self.basis = Basis(
                self.numbers,
                self.par,
                self.ihelp,
                device=self.device,
                dtype=self.dtype,
            )

        # create atomic basis set in libcint format
        atombases = self.basis.create_dqc(positions)

        if self.ihelp.batched:
            from ....param.util import get_elem_angular
            from ....utils import batch

            # integrals do not work with a batched IndexHelper
            _ihelp = [
                IndexHelper.from_numbers(
                    batch.deflate(number), get_elem_angular(self.par.element)
                )
                for number in self.numbers
            ]

            assert isinstance(atombases, list)
            self.drv = [
                LibcintWrapper(ab, ihelp)
                for ab, ihelp in zip(atombases, _ihelp)
                if is_basis_list(ab)
            ]
        else:
            assert is_basis_list(atombases)
            self.drv = LibcintWrapper(atombases, self.ihelp)

        # setting positions signals complete setup
        self._positions = positions

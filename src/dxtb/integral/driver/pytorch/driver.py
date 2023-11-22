"""
Collection of PyTorch-based integral drivers.
"""
from __future__ import annotations

import torch

from ...._types import Any, Tensor
from ....basis import Basis, IndexHelper
from ...base import IntDriver
from .base import PytorchImplementation
from .impls import OverlapAG, OverlapFunction, overlap_gradient


class IntDriverPytorch(PytorchImplementation, IntDriver):
    """
    PyTorch-based integral driver.
    Currently, only the overlap integral is implemented.
    """

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
            from ....param import get_elem_angular
            from ....utils import batch

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
                    pos = batch.deflate(positions[_batch])

                self._positions_batch.append(pos)

                # INDEXHELPER
                # unfortunately, we need a new IndexHelper for each batch,
                # but this is much faster than `calc_overlap`
                nums = batch.deflate(self.numbers[_batch])
                ihelp = IndexHelper.from_numbers(
                    nums, get_elem_angular(self.par.element)
                )

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

    def setup_eval_funcs(self) -> None:
        self.eval_ovlp: OverlapFunction = OverlapAG.apply  # type: ignore
        self.eval_ovlp_grad: OverlapFunction = overlap_gradient

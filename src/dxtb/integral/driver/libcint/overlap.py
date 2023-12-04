"""
Overlap implementation based on `libcint`.
"""
from __future__ import annotations

import torch

from ...._types import Tensor
from ....utils import batch
from ...base import BaseIntegralImplementation, IntDriver
from .base import LibcintImplementation
from .driver import IntDriverLibcint
from .impls import LibcintWrapper, int1e, overlap

__all__ = ["OverlapLibcint"]


def snorm(ovlp: Tensor) -> Tensor:
    return torch.pow(ovlp.diagonal(dim1=-1, dim2=-2), -0.5)


class OverlapLibcint(BaseIntegralImplementation, LibcintImplementation):
    """
    Overlap integral from atomic orbitals.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self._gradient = None

    @property
    def gradient(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("Overlap gradient has not been calculated.")
        return self._gradient

    @gradient.setter
    def gradient(self, mat: Tensor) -> None:
        self._gradient = mat

    def build(self, driver: IntDriverLibcint) -> Tensor:
        """
        Calculation of overlap integral using libcint.

        Returns
        -------
        driver : IntDriverLibcint
            The integral driver for the calculation.
        """
        super().checks(driver)

        def fcn(driver: LibcintWrapper) -> tuple[Tensor, Tensor]:
            s = overlap(driver)
            norm = snorm(s)
            mat = torch.einsum("...ij,...i,...j->...ij", s, norm, norm)
            return mat, norm

        # batched mode
        if driver.ihelp.batched:
            assert isinstance(driver.drv, list)

            slist = []
            nlist = []

            for d in driver.drv:
                mat, norm = fcn(d)
                slist.append(mat)
                nlist.append(norm)

            self.norm = batch.pack(nlist)
            self.matrix = batch.pack(slist)
            return self.matrix

        # single mode
        assert isinstance(driver.drv, LibcintWrapper)

        mat, norm = fcn(driver.drv)
        self.norm = norm
        self.matrix = mat
        return self.matrix

    def get_gradient(self, driver: IntDriverLibcint) -> Tensor:
        """
        Overlap gradient calculation using libcint.

        Parameters
        ----------
        driver : IntDriverLibcint
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Overlap gradient of shape `(nb, norb, norb, 3)`.
        """
        super().checks(driver)

        def fcn(driver: LibcintWrapper, norm: Tensor) -> Tensor:
            # (3, norb, norb)
            grad = int1e("ipovlp", driver)

            # normalize and move xyz dimension to last, which is required for
            # the reduction (only works with extra dimension in last)
            grad = -torch.einsum("...xij,...i,...j->...ijx", grad, norm, norm)
            return grad

        # build norm if not already available
        if self.norm is None:
            if driver.ihelp.batched:
                assert isinstance(driver.drv, list)
                self.norm = batch.pack([snorm(overlap(d)) for d in driver.drv])
            else:
                assert isinstance(driver.drv, LibcintWrapper)
                self.norm = snorm(overlap(driver.drv))

        # batched mode
        if driver.ihelp.batched:
            assert isinstance(driver.drv, list)

            glist = []
            for i, d in enumerate(driver.drv):
                norm = batch.deflate(self.norm[i])
                grad = fcn(d, norm)
                glist.append(grad)

            self.grad = batch.pack(glist)
            return self.grad

        # single mode
        assert isinstance(driver.drv, LibcintWrapper)

        self.grad = fcn(driver.drv, self.norm)
        return self.grad

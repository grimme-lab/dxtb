"""
Base class for Multipole Integrals
==================================

Calculation and modification of multipole integrals.
"""

from __future__ import annotations

import torch

from ...._types import Tensor
from ....constants import labels
from ....utils import batch
from ...base import BaseIntegralImplementation
from .driver import IntDriverLibcint
from .impls import LibcintWrapper, int1e


class LibcintImplementation:
    """
    Simple label for `libcint`-based integral implementations.
    """

    family: int = labels.INTDRIVER_LIBCINT
    """Label for integral implementation family"""

    def checks(self, driver: IntDriverLibcint) -> None:
        """
        Check if the type of integral driver is correct.

        Parameters
        ----------
        driver : IntDriverLibcint
            Integral driver for the calculation.
        """
        if not isinstance(driver, IntDriverLibcint):
            raise RuntimeError("Wrong integral driver selected.")


class IntegralImplementationLibcint(
    LibcintImplementation,
    BaseIntegralImplementation,
):
    """PyTorch-based integral implementation"""

    def checks(self, driver: IntDriverLibcint) -> None:
        """
        Check if the type of integral driver is correct.

        Parameters
        ----------
        driver : BaseIntDriverPytorch
            Integral driver for the calculation.
        """
        super().checks(driver)

        if not isinstance(driver, IntDriverLibcint):
            raise RuntimeError("Wrong integral driver selected.")

    def get_gradient(self, _: IntDriverLibcint) -> Tensor:
        """
        Create the nuclear integral derivative matrix.

        Parameters
        ----------
        driver : IntDriver
            Integral driver for the calculation.

        Returns
        -------
        Tensor
            Nuclear integral derivative matrix.
        """
        raise NotImplementedError(
            "The `get_gradient` method is not implemented for libcint "
            "integrals as it is not explicitly required."
        )


class MultipoleLibcint(IntegralImplementationLibcint):
    """
    Base class for multipole integrals calculated with `libcint`.
    """

    def multipole(self, driver: IntDriverLibcint, intstring: str) -> Tensor:
        """
        Calculation of multipole integral. The integral is normalized, using
        the diagonal of the overlap integral.

        Parameters
        ----------
        driver : IntDriver
            The integral driver for the calculation.
        intstring : str
            String for `libcint` integral engine.

        Returns
        -------
        Tensor
            Normalized multipole integral.
        """
        super().checks(driver)

        allowed_mps = ("r0", "r0r0", "r0r0r0")
        if intstring not in allowed_mps:
            raise ValueError(
                f"Unknown integral string '{intstring}' provided.\n"
                f"Only '{', '.join(allowed_mps)} are allowed.\n'"
                "Other integrals can be added to dxtblibs."
            )

        if self.norm is None:
            raise RuntimeError("Norm must be set before building.")

        def _mpint(driver: LibcintWrapper, norm: Tensor) -> Tensor:
            return torch.einsum(
                "...ij,i,j->...ij", int1e(intstring, driver), norm, norm
            )

        if driver.ihelp.batched:
            if not isinstance(driver.drv, list):
                raise RuntimeError(
                    "IndexHelper on integral driver is batched, but the driver "
                    "instance itself not."
                )

            self.matrix = batch.pack(
                [
                    _mpint(d, batch.deflate(self.norm[_batch]))
                    for _batch, d in enumerate(driver.drv)
                ]
            )
        else:
            if not isinstance(driver.drv, LibcintWrapper):
                raise RuntimeError(
                    "IndexHelper on integral driver is not batched, but the "
                    "driver instance itself seems to be batched."
                )

            self.matrix = _mpint(driver.drv, self.norm)

        return self.matrix

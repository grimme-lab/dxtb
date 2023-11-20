"""
Dipole integral implementation based on `libcint`.
"""
from __future__ import annotations

import torch

from ...._types import Tensor
from .base_multipole import MultipoleLibcint
from .driver import IntDriver

__all__ = ["DipoleLibcint"]


class DipoleLibcint(MultipoleLibcint):
    """
    Dipole integral from atomic orbitals.
    """

    def build(self, driver: IntDriver) -> Tensor:
        """
        Overlap calculation using libcint.

        Parameters
        ----------
        driver : IntDriver
            The integral driver for the calculation.

        Returns
        -------
        Tensor
            Dipole integral.
        """
        self.matrix = self.multipole(driver, "r0")
        return self.matrix

    def shift_r0_rj(self, overlap: Tensor, pos: Tensor) -> Tensor:
        r"""
        Shift the centering of the dipole integral (moment operator) from the origin
        (`r0 = r - (0, 0, 0)`) to atoms (ket index, `rj = r - r_j`).

        .. math::
            D &= D^{r_j}
            &= \langle i | r_j | j \rangle
            &= \langle i | r | j \rangle - r_j \langle i | j \rangle \\
            &= \langle i | r_0 | j \rangle - r_j S_{ij}
            &= D^{r_0} - r_j S_{ij}

        Parameters
        ----------
        r0 : Tensor
            Origin centered dipole integral.
        overlap : Tensor
            Overlap integral.
        pos : Tensor
            Orbital-resolved atomic positions.

        Raises
        ------
        RuntimeError
            Shape mismatch between `positions` and `overlap`.
            The positions must be orbital-resolved.

        Returns
        -------
        Tensor
            Second-index (ket) atom-centered dipole integral.
        """
        if pos.shape[-2] != overlap.shape[-1]:
            raise RuntimeError(
                "Shape mismatch between positions and overlap integral. "
                "The position tensor must be spread to orbital-resolution."
            )

        shift = torch.einsum("...jx,...ij->...xij", pos, overlap)
        self.matrix = self.matrix - shift
        return self.matrix

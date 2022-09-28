from __future__ import annotations

from .type import Dispersion
from ..typing import Tensor


class DispersionD4(Dispersion):
    """Representation of the DFT-D4 dispersion correction."""

    numbers: Tensor
    """Atomic numbers of all atoms."""

    param: dict[str, float]
    """Dispersion parameters."""

    def __init__(self, *args):
        raise NotImplementedError("D4 dispersion scheme not implemented.")

    def get_energy(self, positions: Tensor, **kwargs) -> Tensor:
        """
        Get D4 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms.

        Returns
        -------
        Tensor
            Atom-resolved D4 dispersion energy.
        """

        raise NotImplementedError("D4 dispersion scheme not implemented.")
    
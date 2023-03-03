"""
DFT-D4 dispersion model.
"""
from __future__ import annotations

from .._types import Any, NoReturn, Tensor
from .base import Dispersion


class DispersionD4(Dispersion):
    """
    Representation of the DFT-D4 dispersion correction.

    Note:
    -----
    DispersionD4 should be an `Interaction` as D4 can be self-consistent.
    However, this requires a different setup starting with the base class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError("D4 dispersion scheme not implemented.")

    def get_energy(self, positions: Tensor, **kwargs: Any) -> Tensor:
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

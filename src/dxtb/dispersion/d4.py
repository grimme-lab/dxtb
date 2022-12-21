"""
DFT-D4 dispersion model.
"""
from __future__ import annotations

from .._types import Tensor
from ..interaction import Interaction
from .abc import Dispersion


class DispersionD4(Dispersion, Interaction):
    """
    Representation of the DFT-D4 dispersion correction.

    Note:
    -----
    DispersionD4 should be an `Interaction` as D4 can be self-consistent.
    """

    def __init__(self, *args, **kwargs):
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

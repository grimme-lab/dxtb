"""
DFT-D3(BJ) dispersion model.
"""


import tad_dftd3 as d3

from .abc import Dispersion
from ..typing import Tensor


class DispersionD3(Dispersion):
    """Representation of the DFT-D3(BJ) dispersion correction."""

    numbers: Tensor
    """Atomic numbers of all atoms."""

    param: dict[str, float]
    """Dispersion parameters."""

    def get_energy(self, positions: Tensor, **kwargs) -> Tensor:
        """
        Get D3 dispersion energy.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms.

        Returns
        -------
        Tensor
            Atom-resolved D3 dispersion energy.
        """

        return d3.dftd3(self.numbers, positions, self.param, **kwargs)

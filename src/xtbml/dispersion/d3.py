from __future__ import annotations

import tad_dftd3 as d3


from .type import Dispersion
from ..typing import Tensor


class DispersionD3(Dispersion):
    """Representation of the DFT-D3(BJ) dispersion correction."""

    def __init__(
        self, numbers: Tensor, positions: Tensor, param: dict[str, float]
    ) -> None:
        self.numbers = numbers
        self.positions = positions
        self.param = param

    def get_energy(self) -> Tensor:
        """
        Get dispersion energy.

        Returns
        -------
        Tensor
            Atom-resolved dispersion energy.
        """

        return d3.dftd3(self.numbers, self.positions, self.param)

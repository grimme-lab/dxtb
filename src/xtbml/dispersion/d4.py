from __future__ import annotations

from .type import Dispersion
from ..typing import Tensor


class DispersionD4(Dispersion):
    """Representation of the DFT-D4 dispersion correction."""

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

        raise NotImplementedError()

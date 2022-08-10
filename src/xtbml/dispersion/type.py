from ..param import Param
from ..typing import Tensor


class Dispersion:
    """
    Base class for dispersion correction.
    """

    def __init__(self, numbers: Tensor, positions: Tensor, param: Param) -> None:
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
        ...

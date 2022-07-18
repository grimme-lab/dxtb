# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from ..basis.type import Basis
from ..param import Param
from ..typing import Tensor
from ..xtb.h0 import Hamiltonian


class Calculator:
    """
    Parametrized calculator defining the extended tight-binding model.

    The calculator holds the atomic orbital basis set for defining the Hamiltonian
    and the overlap matrix.
    """

    basis: Basis
    """Atomic orbital basis set definition."""

    hamiltonian: Hamiltonian
    """Core Hamiltonian definition."""

    def __init__(
        self, numbers: Tensor, positions: Tensor, par: Param, acc: float = 1.0
    ) -> None:

        self.basis = Basis(numbers, par, acc)
        self.hamiltonian = Hamiltonian(numbers, positions, par)

# This file is part of xtbml.

"""
Definition of energy terms as abstract base class for classical interactions.
"""

from typing import Literal, Optional, Tuple, Union, overload
from ..typing import Tensor

from pydantic import BaseModel
from abc import ABC, abstractmethod
import torch

# TODO: allow for usability with base Params object
class EnergyContribution(BaseModel, ABC):
    """
    Abstract base class for calculation of classical contributions, like repulsion interactions.
    This class provides a method to retrieve the contributions to the energy, gradient and virial
    within a given cutoff.
    """  # TODOC

    numbers: Tensor
    """The atomic numbers of the atoms in the system."""

    positions: Tensor
    """The positions of the atoms in the system."""

    trans: Optional[Tensor] = None
    """Lattice points"""

    cutoff: Tensor = torch.tensor(25.0)
    """Real space cutoff"""

    energy: Optional[Tensor] = None  # single value, needs gradient
    """Repulsion energy"""

    gradient: Optional[Tensor] = None
    """Molecular gradient of the repulsion energy"""

    sigma: Optional[Tensor] = None
    """Strain derivatives of the repulsion energy"""

    req_grad: Optional[bool] = False
    """Flag for autograd computation. Defaults to `False`"""

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True

    @overload
    @abstractmethod
    def get_engrad(self, calc_gradient: Literal[False] = False) -> Tensor:
        ...

    @overload
    @abstractmethod
    def get_engrad(self, calc_gradient: Literal[True] = True) -> Tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def get_engrad(
        self, calc_gradient: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Obtain energy and gradient for classical contributions.
        """
        ...

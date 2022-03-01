# This file is part of xtbml.


"""
Definition of energy terms as abstract base class for classical interactions.

"""

from typing import Optional
from pydantic import BaseModel
from abc import ABC
from torch import Tensor

from tbmalt.structures.geometry import Geometry

# TODO: allow for usability with base Params object
class Energy_Contribution(BaseModel, ABC):
    """
    Abstract base class for calculation of classical contributions, like repulsion interactions.
    This class provides a method to retrieve the contributions to the energy, gradient and virial
    within a given cutoff.
    """  # TODOC

    """Molecular structure data"""
    geo: Geometry
    """Lattice points"""
    trans: Optional[Tensor] = None
    """Real space cutoff"""
    cutoff: Optional[float] = None
    """Repulsion energy"""
    energy: Optional[Tensor] = None  # single value, needs gradient
    """Molecular gradient of the repulsion energy"""
    gradient: Optional[Tensor] = None
    """Strain derivatives of the repulsion energy"""
    sigma: Optional[Tensor] = None

    @abstractmethod
    def get_energy(self) -> Tensor:
        """
        Obtain energy for classical contribution
        """
        return

    @abstractmethod
    def get_gradient(self) -> Tensor:
        """
        Obtain gradient for classical contribution
        """
        return

    @abstractmethod
    def get_virial(self) -> Tensor:
        """
        Obtain virial for classical contribution
        """
        return

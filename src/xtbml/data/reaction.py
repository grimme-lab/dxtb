from typing import Dict, List
from pydantic import BaseModel
from torch import Tensor


class Reaction(BaseModel):
    """Representation for reaction involving multiple samples."""

    uid: str
    """Unique identifier for reaction"""
    reactants: List[str]  # reactants, participants, partner, ...
    """List of reactants uids"""
    nu: List[int]
    """Stoichiometry coefficient for respective participant"""
    egfn1: Tensor
    """Reaction energies given by GFN1-xtb"""
    eref: Tensor
    """Reaction energies given by reference method"""

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True

    def from_json(path: str) -> List["Reaction"]:
        """
        Create reaction from json.
        """
        raise NotImplementedError

    def to_json(self) -> Dict:
        """
        Convert reaction to json.
        """
        raise NotImplementedError

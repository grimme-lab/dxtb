from typing import Dict, List
from pydantic import BaseModel
from torch import Tensor


class Reaction(BaseModel):
    """Representation for reaction involving multiple samples."""

    # supported features
    uid: str
    """Unique identifier for reaction"""
    reactant: List[str]
    """List of reagent uids"""
    products: List[str]
    """List of products uids"""
    nu_r: List[int]
    """Stoichiometry coefficient for respective reactants"""
    nu_p: List[int]
    """Stoichiometry coefficient for respective products"""
    e: Tensor
    """Reaction energies"""

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True

    def from_json() -> "Reaction":
        """
        Create reaction from json.
        """
        raise NotImplementedError

    def to_json(self) -> Dict:
        """
        Convert reaction to json.
        """
        raise NotImplementedError

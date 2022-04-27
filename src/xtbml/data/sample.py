from typing import Dict
from pydantic import BaseModel
from torch import Tensor


class Sample(BaseModel):
    """Representation for single sample information."""

    # supported features
    uid: str
    """Unique identifier for sample"""
    positions: Tensor
    """Atomic positions"""
    atomic_numbers: Tensor
    """Atomic numbers"""
    egfn1: Tensor
    """Energy calculated by GFN1-xtb"""
    overlap: Tensor
    """Overlap matrix"""
    hamiltonian: Tensor
    """Hamiltonian matrix"""
    # ...
    # TODO: add further QM features

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True

    def from_json() -> "Sample":
        """
        Create sample from json.
        """
        raise NotImplementedError

    def to_json(self) -> Dict:
        """
        Convert sample to json.
        """
        raise NotImplementedError

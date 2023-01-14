"""
Definition of the anisotropic second-order multipolar interactions.
"""
from __future__ import annotations

from pydantic import BaseModel


class Multipole(BaseModel):
    """
    Representation of the anisotropic second-order multipolar interactions
    for a parametrization.
    """

"""
Definition of the isotropic third-order onsite correction.
"""
from __future__ import annotations

from pydantic import BaseModel


class ThirdOrder(BaseModel):
    """
    Representation of the isotropic third-order onsite correction.
    """

    shell: bool = False
    """Whether the third order contribution is shell-dependent or only atomwise."""

"""
Definition of the isotropic third-order onsite correction.
"""
from __future__ import annotations

from pydantic import BaseModel

from typing import Optional


class ThirdOrder(BaseModel):
    """
    Representation of the isotropic third-order onsite correction.
    """

    shell: Optional[bool] = False
    """Whether the third order contribution is shell-dependent or only atomwise."""

"""
Definition of the isotropic third-order onsite correction.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ThirdOrder(BaseModel):
    """
    Representation of the isotropic third-order onsite correction.
    """

    shell: Optional[bool] = False
    """Whether the third order contribution is shell-dependent or only atomwise."""

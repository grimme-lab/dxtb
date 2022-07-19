"""
Definition of the isotropic third-order onsite correction.
"""

from pydantic import BaseModel


class ThirdOrder(BaseModel):
    """
    Representation of the isotropic third-order onsite correction.
    """

    shell: bool = False
    """Whether the third order contribution is shell-dependent or only atomwise."""

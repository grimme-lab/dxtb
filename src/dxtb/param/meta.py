# This file is part of xtbml.
"""
Meta data associated with a parametrization. Mainly used for identification of data format.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Meta(BaseModel):
    """
    Representation of the meta data for a parametrization.
    """

    name: Optional[str]
    """Name of the represented method"""
    reference: Optional[str]
    """References relevant for the parametrization records"""
    version: int = 0
    """Version of the represented method"""

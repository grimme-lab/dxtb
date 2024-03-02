"""
GFN1-xTB
========

This module loads the standard GFN1-xTB parametrization (lazily).
"""

from __future__ import annotations

from pathlib import Path

from ...loader import LazyLoaderParam as Lazy
from ..base import Param

GFN1_XTB: Param = Lazy(Path(__file__).parent / "gfn1-xtb.toml")  # type: ignore

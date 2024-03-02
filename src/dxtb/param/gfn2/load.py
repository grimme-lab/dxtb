"""
GFN2-xTB
========

This module loads the standard GFN2-xTB parametrization (lazily).
"""

from __future__ import annotations

from pathlib import Path

from ...loader import LazyLoaderParam as Lazy
from ..base import Param

GFN2_XTB: Param = Lazy(Path(__file__).parent / "gfn2-xtb.toml")  # type: ignore

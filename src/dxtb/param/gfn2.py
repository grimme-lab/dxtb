"""
GFN2-xTB
========

This module loads the GFN2-xTB parametrization.

.. warning::

    GFN2-xTB is not implemented. The parameters should only be used for testing.
"""
from __future__ import annotations

import os.path as op

import tomli as toml

from .base import Param

with open(op.join(op.dirname(__file__), "gfn2-xtb.toml"), "rb") as fd:
    GFN2_XTB = Param(**toml.load(fd))
